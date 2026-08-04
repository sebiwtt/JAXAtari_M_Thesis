# =============================================================================
# Run a whole CRL campaign from one YAML manifest, spread across GPUs.
# =============================================================================
# Where run_all_crl_seeds.py runs ONE composition over several seeds, this takes a
# manifest describing the full cross-product
#
#     sequences x methods x modalities x seeds
#
# expands it into individual ppo_crl_continual.py runs, and feeds them to a pool of
# workers (default: one run at a time per GPU, since NUM_ENVS=8192 fills a GPU).
#
#   python tools/run_campaign.py tools/campaigns/final_eval.yaml
#
# =============================================================================

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

CRL_DIR = Path(__file__).resolve().parent.parent  # scripts/benchmarks/CRL
CONFIG_DIR = CRL_DIR / "config"

from config_groups import sequence_yaml_path  # sibling module in tools/

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _fmt_override(key: str, value) -> str:
    """One hydra CLI override. Bools must be lower-case; lists need no spaces."""
    if isinstance(value, bool):
        value = str(value).lower()
    elif isinstance(value, (list, tuple)):
        value = "[" + ",".join(str(v) for v in value) + "]"
    return f"{key}={value}"


def _group_yaml(group: str, name: str) -> dict:
    # sequence/ is nested one folder per game; "pong_dyn4" and "pong/pong_dyn4" both work.
    path = sequence_yaml_path(name) if group == "sequence" else CONFIG_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"unknown {group} '{name}': {path} does not exist")
    return yaml.safe_load(path.read_text()) or {}


def _run_dir_name(sequence: str, method: str, modality: str, seed: int, overrides: dict) -> str:
    """Mirror ppo_crl_continual's run-dir naming so finished runs can be detected."""
    seq_cfg = _group_yaml("sequence", sequence)
    env_id = overrides.get("ENV_ID", seq_cfg.get("ENV_ID", sequence))
    exp_name = overrides.get("EXP_NAME") or f"{method}_{seq_cfg.get('SEQUENCE', sequence)}"
    pixel = bool(_group_yaml("modality", modality).get("PIXEL_BASED", False))
    return f"{env_id}_{exp_name}_{'pixel' if pixel else 'oc'}_{seed}"


def expand_jobs(manifest: dict) -> list[dict]:
    """Cross-product every block in the manifest into a flat, de-duplicated job list."""
    defaults = {
        "sequences": manifest.get("sequences"),
        "methods": manifest.get("methods", ["ft"]),
        "modalities": manifest.get("modalities", ["oc"]),
        "seeds": manifest.get("seeds", [0, 1, 2]),
        "overrides": manifest.get("overrides", {}) or {},
    }
    blocks = manifest.get("groups") or [{}]
    jobs, seen = [], set()
    for block in blocks:
        sequences = block.get("sequences", defaults["sequences"])
        if not sequences:
            raise SystemExit("manifest must list at least one sequence (top-level 'sequences' or per-group)")
        methods = block.get("methods", defaults["methods"])
        modalities = block.get("modalities", defaults["modalities"])
        seeds = block.get("seeds", defaults["seeds"])
        overrides = {**defaults["overrides"], **(block.get("overrides") or {})}
        for sequence, method, modality, seed in itertools.product(sequences, methods, modalities, seeds):
            key = (sequence, method, modality, int(seed))
            if key in seen:  # same run reachable from two blocks: keep the first
                continue
            seen.add(key)
            name = _run_dir_name(sequence, method, modality, int(seed), overrides)
            resolvable = "${" not in name
            jobs.append({
                "name": name if resolvable else f"{sequence}_{method}_{modality}_seed{seed}",
                "resolvable": resolvable,
                "sequence": sequence,
                "method": method,
                "modality": modality,
                "seed": int(seed),
                "overrides": overrides,
            })
    return jobs


def build_cmd(job: dict, python_cmd: list[str]) -> list[str]:
    return [
        *python_cmd,
        str(CRL_DIR / "ppo_crl_continual.py"),
        f"sequence={job['sequence']}",
        f"method={job['method']}",
        f"modality={job['modality']}",
        f"SEED={job['seed']}",
        *(_fmt_override(k, v) for k, v in job["overrides"].items()),
    ]


def worker(gpu: str, task_queue: "queue.Queue", python_cmd: list[str], extra_env: dict,
           log_dir: Path, results: list, total: int) -> None:
    """Pull jobs off the shared queue and run them one at a time on this GPU."""
    while True:
        try:
            idx, job = task_queue.get_nowait()
        except queue.Empty:
            return
        try:
            env = {**os.environ, **{k: str(v) for k, v in extra_env.items()}}
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = log_dir / f"{job['name']}.log"
            _log(f"[gpu {gpu}] ({idx}/{total}) START {job['name']}  -> {log_path}")
            started = time.time()
            with open(log_path, "w") as log_file:
                log_file.write(" ".join(build_cmd(job, python_cmd)) + "\n\n")
                log_file.flush()
                proc = subprocess.run(
                    build_cmd(job, python_cmd), env=env, cwd=CRL_DIR,
                    stdout=log_file, stderr=subprocess.STDOUT,
                )
            mins = (time.time() - started) / 60
            ok = proc.returncode == 0
            results.append({"job": job, "ok": ok, "code": proc.returncode, "log": log_path, "minutes": mins})
            status = "DONE " if ok else f"FAIL({proc.returncode})"
            _log(f"[gpu {gpu}] ({idx}/{total}) {status} {job['name']}  [{mins:.1f} min]")
        finally:
            task_queue.task_done()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="path to the campaign YAML")
    ap.add_argument("--dry-run", action="store_true", help="print the expanded job list and exit")
    ap.add_argument("--gpus", help="comma-separated GPU ids, overriding the manifest")
    ap.add_argument("--force", action="store_true", help="re-run jobs whose matrix.json already exists")
    args = ap.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text()) or {}
    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else [str(g) for g in manifest.get("gpus", [0])]
    workers_per_gpu = int(manifest.get("workers_per_gpu", 1))
    python_cmd = manifest.get("python_cmd") or [sys.executable]
    if isinstance(python_cmd, str):
        python_cmd = python_cmd.split()
    runs_dir = CRL_DIR / manifest.get("runs_dir", "runs")
    log_dir = CRL_DIR / manifest.get("log_dir", "runs/campaign_logs")

    jobs = expand_jobs(manifest)
    skipped = []
    if not args.force and not manifest.get("force", False):
        pending = []
        for job in jobs:
            if job["resolvable"] and (runs_dir / job["name"] / "matrix.json").exists():
                skipped.append(job)
            else:
                pending.append(job)
        jobs = pending

    print(f"campaign: {args.manifest}")
    print(f"  {len(jobs)} run(s) to launch, {len(skipped)} already finished (skipped), "
          f"{len(gpus)} GPU(s) {gpus} x {workers_per_gpu} worker(s)")
    for job in skipped:
        print(f"    skip  {job['name']}")
    for i, job in enumerate(jobs, 1):
        print(f"    {i:>3}.  {job['name']}")
    if args.dry_run:
        print("\ncommands:")
        for job in jobs:
            print("   ", " ".join(build_cmd(job, python_cmd)))
        return
    if not jobs:
        print("nothing to do.")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    task_queue: "queue.Queue" = queue.Queue()
    for i, job in enumerate(jobs, 1):
        task_queue.put((i, job))

    results: list = []
    threads = [
        threading.Thread(target=worker, args=(gpu, task_queue, python_cmd, manifest.get("env", {}) or {},
                                              log_dir, results, len(jobs)), daemon=True)
        for gpu in gpus for _ in range(workers_per_gpu)
    ]
    started = time.time()
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\ninterrupted - running jobs keep going until they exit; re-run to resume the rest.")
        raise SystemExit(130)

    failed = [r for r in results if not r["ok"]]
    print(f"\ncampaign finished in {(time.time() - started) / 60:.1f} min: "
          f"{len(results) - len(failed)} succeeded, {len(failed)} failed, {len(skipped)} skipped")
    for r in failed:
        print(f"  FAILED {r['job']['name']} (exit {r['code']}) - see {r['log']}")
    if failed:
        raise SystemExit(1)
    print(f"\nplot a sequence's methods together, e.g.:\n"
          f"  python tools/plot_crl_curve.py runs/<env>_{{ft,ewc,agem,packnet}}_<seq>_oc_*/ --out crl_methods.png")


if __name__ == "__main__":
    main()
