# Continual-RL benchmark (JAXtari PPO)

A benchmark for **continual reinforcement learning (CRL)**: one PPO agent is trained
sequentially over an ordered list of tasks: variants of a single JAXtari game, each
produced by applying one game modification, carrying its parameters forward from
task to task. After each task the agent is re-evaluated on every task seen so far, producing
a **retention/forgetting matrix** that measures how much of each task's skill survives (or
is lost during) subsequent training.

The framework is game-agnostic; a task sequence just names a game and its ordered mods.
Shipped: **6 games** (asteroids, breakout, freeway, kangaroo, pong, seaquest) × **4 mod
families** (`dyn4` dynamics, `vis4` visuals, `rew4` reward, `mag4` one parameter scaled
×2..×5) = 24 sequences of 5 tasks each.

For a full implementation-level write-up (wrappers, metric derivations, method deviations
from MEAL, caveats), see [`METHODOLOGY.md`](METHODOLOGY.md).

Four continual-learning methods are implemented behind a common interface, so they all run
through the same orchestrator and are directly comparable:

| method    | idea                                                          | config          |
|-----------|---------------------------------------------------------------|-----------------|
| `ft`      | naive finetuning — no mitigation (the baseline)               | `config/method/ft.yaml` |
| `ewc`     | Elastic Weight Consolidation (Kirkpatrick et al. 2017)        | `config/method/ewc.yaml` |
| `agem`    | Averaged Gradient Episodic Memory (Chaudhry et al. 2019)      | `config/method/agem.yaml` |
| `packnet` | PackNet iterative pruning (Mallya & Lazebnik 2018)            | `config/method/packnet.yaml` |

The methods are ported from [MEAL](https://github.com/TTomilin/MEAL) (which implements them
for IPPO) and adapted to this single-agent, single-head PPO trainer.

---

## Quickstart

Run one experiment by composing three config groups — **which tasks**, **which method**,
**which observation modality**:

```bash
cd scripts/benchmarks/CRL
uv run python ppo_crl_continual.py sequence=pong_dyn4 method=ewc modality=oc
```

Any individual key can be overridden on top of the composition:

```bash
uv run python ppo_crl_continual.py method=ewc EWC_COEF=1000 SEED=3 TRACK=False
```

That's the whole interface — `sequence` × `method` × `modality` plus overrides. Defaults
(if you omit a group) are `sequence=pong_dyn4 method=ft modality=oc`, set in
[`config/config.yaml`](config/config.yaml).

---

## What gets measured

For an ordered task list `T[0..n-1]` (`T[0]` is always the unmodified base task):

- **`R[i, j]`** — return of the agent *trained through task i*, evaluated on task j (`j ≤ i`).
- **`R_rand[j]`** — return of a fresh, untrained agent on task j. This is the "knows
  nothing" floor, which is generally far from 0 and game-dependent (e.g. Pong's
  random-policy floor is near −21), keyed off `EVAL_SEED`.
- **`Retention[i, j] = clip((R[i,j] − R_rand[j]) / (R[j,j] − R_rand[j]), 0, 1)`** (`j ≤ i`) —
  1.0 means task j's skill is fully retained after training through task i; 0.0 means
  at-or-below random. Clamped both directions — raw Atari returns can go negative (e.g.
  life-loss/miss penalties), so the unclamped ratio can exceed 1.0 whenever a later task
  happens to improve task j beyond its own post-training checkpoint (positive backward
  transfer), or go negative when performance craters below the random floor. Both read as
  "fully retained" / "fully forgotten" respectively once clamped, rather than a confusing
  value outside `[0, 1]`. If `R[j,j] ≤ R_rand[j]` (task j never learned above the floor) the
  denominator is non-positive, so the cell is left NaN with a printed warning.
- **`Drop[i, j] = 1 − Retention[i, j]`** (`j < i` only) — the same information, restricted
  to already-trained tasks and flipped so 0.0 = no forgetting, 1.0 = fully forgotten.
- **`Forgetting[j]`** — recency-weighted average of `Drop[i, j]` over every later checkpoint
  `i > j`, with `weight(i) = exp(-FORGETTING_LAMBDA * (i-j)/(last_task-j))`: degradation seen
  soon after task j counts more than degradation long after. `FORGETTING_LAMBDA=0` recovers
  the plain unweighted mean. The last task has no later checkpoint, so it's left undefined
  (NaN) — by design, not padded out with an extra "finale" task just to fill the cell.
  `mean_forgetting` averages this over all tasks except the last, and
  `mean_retention = 1 - mean_forgetting` (its exact complement — *not* a flat per-cell average
  of `Retention`, which would over-weight earlier tasks).

`Retention`/`Drop`/`Forgetting` are MEAL/COOM-style: the agent is compared to *itself* (its
own performance right after training task j), not to a converged reference. `R_rand` only
rescales/clamps the range here — it isn't something the metric assumes has been converged
to — so this needs no assumption that `R[j,j]` is a converged ceiling; it stays well-defined
even if base task performance was still improving when its training budget ran out. The
recency weighting follows MEAL, which doesn't publish the lambda it uses; `FORGETTING_LAMBDA`
defaults to 1.0 here.

Setting `EVAL_FULL_MATRIX=True` also fills the `j > i` cells (forward transfer to
not-yet-trained tasks), at roughly double the eval cost. These extra cells feed `Retention`
but aren't used by `Drop`/`Forgetting` (which only ever look at `j < i`).

> **Interpreting PackNet:** its retention is **1.0** / forgetting is **0.0**, both by
> construction: it freezes each task's subnetwork, so a completed task can never be
> disturbed. For PackNet the meaningful signal is the *diagonal* `R[j,j]` (how well each
> task learns under a shrinking capacity budget), not retention/forgetting. Compare methods
> on average final performance instead.

---

## Repository layout

```
CRL/
├── ppo_crl_continual.py     # MAIN entry point: the continual orchestrator
├── ppo_trainer.py           # single-task PPO (CL-agnostic; hooks in via cl_method/cl_state)
├── ppo_eval.py              # deterministic evaluation of a saved checkpoint
├── networks.py              # torsos (CNN / MLP) + Actor/Critic heads + AgentParams
├── envs.py                  # make_env: the wrapped JAXtari env factory
│
├── continual/               # continual-learning methods (one file each)
│   ├── base.py              #   CLMethod interface + default (finetuning) behavior
│   ├── ft.py  ewc.py  agem.py  packnet.py
│   └── __init__.py          #   make_cl_method() registry
│
├── config/                  # Hydra config, composed from three groups
│   ├── config.yaml          #   shared defaults + the `defaults:` list
│   ├── sequence/            #   which game + ordered task mods  (24: 6 games × dyn4/vis4/rew4/mag4)
│   ├── method/              #   which CL method + its hyperparams (ft, ewc, agem, packnet)
│   └── modality/            #   observation pipeline + budget    (oc, pixel)
│
├── tools/                   # auxiliary scripts (not part of the core pipeline)
│   ├── visualize_matrix.py  #   render a run's retention/forgetting matrix to PNG
│   ├── ppo_crl_difficulty.py#   rank tasks by adaptation difficulty (separate study)
│   ├── run_difficulty_game.sh#  all 4 mod families of one game, sequentially, on one GPU
│   └── video_utils.py       #   final-rollout video / obs-frame capture
│
└── runs/                    # outputs (git-ignored) — one dir per run
```

---

## The config system

Config is [Hydra](https://hydra.cc) with three **composition groups**. Because the run
matrix is a cross-product (game-sequence × method × modality) and the axes are independent,
each axis is one small file per value — adding a game is 1 sequence file, adding a method is
1 file, and a shared default changes in exactly one place.

- **`config/sequence/*`** — `ENV_ID`, a short `SEQUENCE` label, and `TASK_MODS` (the ordered
  list; index 0 must be `[]`, the base task; at most one mod per task).
- **`config/method/*`** — `CL_METHOD` and that method's hyperparameters.
- **`config/modality/*`** — `PIXEL_BASED` plus the compute budget it implies:
  `BASE_TIMESTEP_BUDGET` (base task) and `TASK_TIMESTEP_BUDGET` (each mod task).
  (`oc`: 8192 envs / 100M base / 50M per mod; `pixel`: 512 envs / 50M / 25M).
- **`config/config.yaml`** — everything shared: wandb, eval protocol, PPO hyperparameters,
  and the `defaults:` list. `EXP_NAME` is derived as `${CL_METHOD}_${SEQUENCE}`.

---

## Method hyperparameters

Edit the method file or override on the CLI. Key knobs:

- **EWC** — `EWC_COEF` (penalty strength, sweep ~`{1,10,100,1e3,1e4}`), `EWC_MODE`
  (`last` | `multi` | `online`), `EWC_DECAY` (online only), `EWC_NORMALIZE_FISHER`.
- **A-GEM** — `AGEM_MEMORY_PER_TASK` (transitions stored per finished task; lower it for
  `pixel`, whose obs are large), `AGEM_SAMPLE_SIZE` (reference-gradient batch per minibatch).
- **PackNet** — `PACKNET_FINETUNE_FRAC` (fraction of each task's budget spent on the
  post-prune finetune phase; MEAL uses 0.5), `PACKNET_FINETUNE_LR`.

---

## Outputs

Each run writes to `runs/{ENV_ID}_{EXP_NAME}_{oc|pixel}_{SEED}/`, e.g.
`runs/pong_ewc_dyn4_oc_0/`:

| file | contents |
|------|----------|
| `matrix.json` / `matrix.npz` | `R`, `R_rand`, `Retention`, `Drop`, `Forgetting`, `mean_forgetting`, `mean_retention`, labels, task mods, method name, wall-clock compute breakdown |
| `task_{i}.cleanrl_model`      | agent checkpoint after task i (full params) |
| `random_agent.cleanrl_model`  | the untrained floor agent |
| `packnet_owner.msgpack`       | (PackNet only) owner tree, needed to recover per-task subnetworks |

With `TRACK=True`, per-iteration metrics stream to Weights & Biases (charts and losses are
grouped per task; A-GEM logs `agem_projected`, EWC logs `cl_penalty`).

---

## Multi-seed runs

Report **mean ± std over seeds**, not single-seed point estimates — RL amplifies tiny
numeric differences (hardware, jax version) into divergent trajectories, so single runs are
noisy. The launcher runs one full sweep per seed, one process per GPU worker:

For a whole campaign (the final eval: many sequences x methods x seeds at once), describe
it in a YAML manifest instead and let `tools/run_campaign.py` expand and schedule it:

```bash
uv run python tools/run_campaign.py tools/campaigns/final_eval.yaml --dry-run   # list the runs
uv run python tools/run_campaign.py tools/campaigns/final_eval.yaml             # launch them
```

```yaml
gpus: [0, 1, 2]
seeds: [1, 2, 3]
sequences: [pong_dyn4, pong_vis4]
methods: [ft, ewc, agem, packnet]
modalities: [oc]
overrides:                 # hydra overrides applied to every run
  CRL_CURVE: true
```

Jobs are queued and each GPU pulls the next one as it frees up, each run's output goes to
`runs/campaign_logs/<run-name>.log`, and runs that already have a `matrix.json` are skipped
- so relaunching after a crash or a Ctrl-C resumes the rest (`--force` re-runs them).
Failures are listed at the end with their log path and the exit code is non-zero.
See `tools/campaigns/final_eval.yaml` for the documented schema (per-block `groups:`,
`workers_per_gpu`, extra `env:` vars).

`EVAL_SEED` is derived from `SEED` (`SEED * 12 + 1`, see `config/config.yaml`), so each
replicate gets its own reproducible-but-decorrelated eval seed rather than sharing one
fixed value across all of them.

---

## Visualization

```bash
uv run python tools/visualize_matrix.py runs/pong_ewc_dyn4_oc_0            # -> <run>/visualization.png
uv run python tools/visualize_matrix.py runs/pong_ewc_dyn4_oc_0 --out fig.png --show
```

---

## Task-difficulty study (separate)

`tools/ppo_crl_difficulty.py` answers a different question — "starting from a base agent, how
many steps does each single-mod task need to recover base-level performance?" — and ranks
tasks by adaptation cost. It shares the same config system:

```bash
uv run python tools/ppo_crl_difficulty.py sequence=pong_dyn4 modality=oc
```

To get the difficulty picture for one game — the `dyn4` and `vis4` mod families — use the
wrapper, which pins everything to a single GPU and runs the sequences strictly one after
the other (one process each, so VRAM is released in between):

```bash
./tools/run_difficulty_game.sh 2 pong            # GPU 2, dyn4 + vis4
./tools/run_difficulty_game.sh 0 seaquest --modality pixel --seed 1
nohup ./tools/run_difficulty_game.sh 2 pong > sweep_pong.log 2>&1 &
```

It warns if the requested GPU is already busy, skips sequences whose `runs/` dir exists
(unless `--force`), logs each sequence to `runs/difficulty_logs/`, and prints a summary.
`--help` lists all options; anything after `--` is forwarded to `ppo_crl_difficulty.py`.

`rew4` and `mag4` are not in the default list. `rew4` in particular is unsound for this
study as written: its mods replace the reward function, so the probe's return is measured
on a different scale than the base-derived target, and the resulting ranking reflects
reward scale rather than adaptation cost (`sparse_scoring` ranks "hardest" despite needing
no policy change at all, while `reward_per_hit` clears the target on the first probe).
Fixing it needs a task-relative target (normalize against `R_rand[j]` and a per-task
ceiling, as the retention matrix does), not just a different `--seqs`.

---

## Reproducibility note

Exact cross-hardware reproducibility is **not achievable** for JAX RL: identical seeds make
the *computation* deterministic but not the *floating-point arithmetic*, which differs across
GPU architectures, jax/jaxlib versions, and TF32 settings — and RL's feedback loop amplifies
those ~1e-7 differences into completely different trajectories. Pin the software stack
(commit a `uv.lock`, set `jax_default_matmul_precision="highest"`) for same-hardware
determinism, and otherwise report seed-averaged results.
