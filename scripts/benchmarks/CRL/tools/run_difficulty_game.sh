#!/usr/bin/env bash
# =============================================================================
# Run the task-difficulty study for ONE game over all four mod families,
# pinned to a single GPU and executed strictly one after the other.
#
# Each sequence runs as its own process, so the VRAM of the previous one is
# fully released before the next starts. The GPU stays occupied for the whole
# duration, which is the point: pick one free GPU, hand it this script, leave
# the others to everyone else.
#
# Usage:
#   ./tools/run_difficulty_game.sh <gpu> <game> [options] [-- hydra overrides]
#   ./tools/run_difficulty_game.sh 2 pong
#
# Options:
#   --seqs a,b,c   mod families to run, in order   (default: dyn4,rew4,vis4,mag4)
#   --modality m   oc | pixel                      (default: oc)
#   --method m     ft | ewc | agem | packnet       (default: ft; only affects EXP_NAME here)
#   --seed n       SEED passed to every sequence   (default: 0)
#   --force        re-run sequences whose runs/ dir already exists (overwrites it)
#   --dry-run      print the commands without running them
# Everything after `--` is forwarded verbatim to ppo_crl_difficulty.py.
# =============================================================================

set -uo pipefail

CRL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEQS="dyn4,rew4,vis4,mag4"
MODALITY="oc"
METHOD="ft"
SEED="0"
FORCE=0
DRY_RUN=0

usage() { sed -n '3,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -lt 2 ]] && usage 1
case "$1" in -h|--help) usage 0 ;; esac

GPU="$1"; GAME="$2"; shift 2

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seqs)     SEQS="$2"; shift 2 ;;
    --modality) MODALITY="$2"; shift 2 ;;
    --method)   METHOD="$2"; shift 2 ;;
    --seed)     SEED="$2"; shift 2 ;;
    --force)    FORCE=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  usage 0 ;;
    --)         shift; EXTRA_ARGS=("$@"); break ;;
    *)          echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "Error: <gpu> must be a GPU index, got '$GPU'." >&2; exit 1; }

IFS=',' read -r -a SEQ_LIST <<< "$SEQS"

# Fail before burning hours on a typo'd game/mod-family combination.
for fam in "${SEQ_LIST[@]}"; do
  [[ -f "$CRL_DIR/config/sequence/${GAME}_${fam}.yaml" ]] || {
    echo "Error: no config/sequence/${GAME}_${fam}.yaml — check the game name and --seqs." >&2
    exit 1
  }
done

# ---------------------------------------------------------------------------
# Pre-flight: is the requested GPU actually free?
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  used_mib="$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  if [[ -z "$used_mib" ]]; then
    echo "Warning: nvidia-smi could not query GPU $GPU; continuing anyway." >&2
  elif [[ "$used_mib" -gt 1024 ]]; then
    echo "=============================================================="
    echo "WARNING: GPU $GPU already has ${used_mib} MiB in use:"
    nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null
    echo "Someone else is probably on it. Continuing in 15s — Ctrl-C to abort."
    echo "=============================================================="
    sleep 15
  fi
fi

LOG_DIR="$CRL_DIR/runs/difficulty_logs"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo "Task-difficulty sweep"
echo "  game      : $GAME"
echo "  sequences : ${SEQ_LIST[*]}  (sequential, one process each)"
echo "  gpu       : $GPU   (CUDA_VISIBLE_DEVICES=$GPU)"
echo "  modality  : $MODALITY | method: $METHOD | seed: $SEED"
echo "  extra     : ${EXTRA_ARGS[*]:-none}"
echo "  logs      : $LOG_DIR"
echo "=============================================================="

declare -a STATUS_LINES=()
overall_rc=0

for fam in "${SEQ_LIST[@]}"; do
  sequence="${GAME}_${fam}"
  # Mirrors the run_dir that ppo_crl_difficulty.py builds:
  #   runs/{ENV_ID}_{EXP_NAME}_{oc|pixel}_difficulty_{SEED},  EXP_NAME = {CL_METHOD}_{SEQUENCE}
  run_dir="$CRL_DIR/runs/${GAME}_${METHOD}_${fam}_${MODALITY}_difficulty_${SEED}"
  log_file="$LOG_DIR/${sequence}_${MODALITY}_${METHOD}_seed${SEED}.log"

  if [[ -d "$run_dir" && $FORCE -eq 0 ]]; then
    echo ""
    echo ">>> SKIP $sequence — $run_dir already exists (use --force to overwrite)."
    STATUS_LINES+=("$(printf '%-18s %s' "$sequence" "SKIPPED (existing results)")")
    continue
  fi

  cmd=(uv run python tools/ppo_crl_difficulty.py
       "sequence=$sequence" "method=$METHOD" "modality=$MODALITY" "SEED=$SEED"
       "${EXTRA_ARGS[@]}")

  echo ""
  echo ">>> [$(date '+%F %T')] $sequence"
  echo "    CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
  echo "    log: $log_file"

  if [[ $DRY_RUN -eq 1 ]]; then
    STATUS_LINES+=("$(printf '%-18s %s' "$sequence" "DRY-RUN")")
    continue
  fi

  start=$SECONDS
  ( cd "$CRL_DIR" && CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" ) 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  elapsed=$(( SECONDS - start ))
  hms=$(printf '%02dh%02dm%02ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))

  if [[ $rc -eq 0 ]]; then
    echo "<<< $sequence finished in $hms"
    STATUS_LINES+=("$(printf '%-18s %s' "$sequence" "OK ($hms)")")
  else
    echo "<<< $sequence FAILED (exit $rc) after $hms — see $log_file" >&2
    STATUS_LINES+=("$(printf '%-18s %s' "$sequence" "FAILED exit $rc ($hms)")")
    overall_rc=1
  fi

  # Small pause so the driver has fully reclaimed the VRAM before the next process
  # starts probing it (JAX preallocates aggressively on init).
  sleep 10
done

echo ""
echo "=============================================================="
echo "Summary — game=$GAME modality=$MODALITY method=$METHOD seed=$SEED"
for line in "${STATUS_LINES[@]}"; do echo "  $line"; done
echo ""
echo "Rankings: $CRL_DIR/runs/${GAME}_${METHOD}_<fam>_${MODALITY}_difficulty_${SEED}/difficulty.json"
echo "=============================================================="
exit "$overall_rc"
