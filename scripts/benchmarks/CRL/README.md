# Continual-RL benchmark (JAXtari PPO)

A benchmark for **continual reinforcement learning (CRL)**: one PPO agent is trained
sequentially over an ordered list of tasks: variants of a single JAXtari game, each
produced by applying one game modification, carrying its parameters forward from
task to task. After each task the agent is re-evaluated on every task seen so far, producing
a **retention matrix** that measures how much of each task's skill survives subsequent
training.

The framework is game-agnostic; a task sequence just names a game and its ordered mods. The
sequences shipped right now are Pong variants (`config/sequence/pong_*`), with more games to
follow.

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
- **`Retention[i, j] = (R[i,j] − R_rand[j]) / (R[j,j] − R_rand[j])`** — 1.0 means task j's
  skill is fully retained after training through task i; 0.0 means it has decayed back to
  random.

Setting `EVAL_FULL_MATRIX=True` also fills the `j > i` cells (forward transfer to
not-yet-trained tasks), at roughly double the eval cost.

> **Interpreting PackNet:** its retention is **1.0 by construction**: it freezes each
> task's subnetwork, so a completed task can never be disturbed. For PackNet the meaningful
> signal is the *diagonal* `R[j,j]` (how well each task learns under a shrinking capacity
> budget), not retention. Compare methods on average final performance, not retention.

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
│   ├── sequence/            #   which game + ordered task mods  (pong_dyn4, pong_vis4, pong_rew4)
│   ├── method/              #   which CL method + its hyperparams (ft, ewc, agem, packnet)
│   └── modality/            #   observation pipeline + budget    (oc, pixel)
│
├── tools/                   # auxiliary scripts (not part of the core pipeline)
│   ├── run_all_crl_seeds.py #   launch N seeds across GPUs
│   ├── visualize_matrix.py  #   render a run's retention matrix to PNG
│   ├── ppo_crl_difficulty.py#   rank tasks by adaptation difficulty (separate study)
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
- **`config/modality/*`** — `PIXEL_BASED` plus the compute budget it implies
  (`oc`: 8192 envs / 100M steps-per-task; `pixel`: 512 envs / 10M).
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
| `matrix.json` / `matrix.npz` | `R`, `R_rand`, `Retention`, labels, task mods, method name |
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

```bash
# from the repo root
uv run python scripts/benchmarks/CRL/tools/run_all_crl_seeds.py \
    --gpus 0,1,2,3 --seeds 0,1,2,3,4 \
    --sequence pong_dyn4 --method ewc --modality oc

# anything after `--` is forwarded verbatim to ppo_crl_continual.py:
uv run python .../run_all_crl_seeds.py --gpus 0 --seeds 0,1,2 -- TOTAL_TIMESTEPS=1000000
```

`EVAL_SEED` is deliberately *not* varied across replicates, so every seed is scored under an
identical eval protocol.

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

---

## Reproducibility note

Exact cross-hardware reproducibility is **not achievable** for JAX RL: identical seeds make
the *computation* deterministic but not the *floating-point arithmetic*, which differs across
GPU architectures, jax/jaxlib versions, and TF32 settings — and RL's feedback loop amplifies
those ~1e-7 differences into completely different trajectories. Pin the software stack
(commit a `uv.lock`, set `jax_default_matmul_precision="highest"`) for same-hardware
determinism, and otherwise report seed-averaged results.
