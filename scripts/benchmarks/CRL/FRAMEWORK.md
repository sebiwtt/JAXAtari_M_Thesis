# Bring your own agent: the benchmark framework

The CRL benchmark is agent-agnostic. The harness owns everything that must be
identical across submissions for results to be comparable — the task sequences
(game + ordered mods), the per-task training budgets, the evaluation protocol,
and the metric definitions (`R`, `R_rand`, `Retention`, `Drop`, `Forgetting`,
see [README.md](README.md#what-gets-measured)). **You** own everything about
learning: the algorithm (PPO, PQN, DQN, world models, …), the networks, the
optimizer, and any continual-learning machinery.

To submit an agent you implement three functions against a fixed signature,
ship the file, and start the benchmark with a single call.

## The contract

Subclass [`framework.interface.ContinualAgent`](framework/interface.py) and
implement:

| method | you provide |
|---|---|
| `init_state(key)` | a freshly-initialized (untrained) agent state — any pytree: params, optimizer state, buffers. Deterministic in `key`. |
| `train_task(state, task, ctx)` | train on one task within `task.budget` env steps and return the new state. Plain Python; jit whatever you like inside. |
| `policy(state, eval_task, trained_task)` | the jit-safe act function evaluation runs: `act(obs, key) -> (action, key)`, obs shaped `(1, *obs_shape)`, action shaped `(1,)`. |

Optionally override `save_checkpoint` / `save_artifacts` / `collect_curve_points`
/ `describe` (all have working defaults). The docstrings in
[framework/interface.py](framework/interface.py) are the authoritative,
detailed version of this contract.

`task` is a [`TaskSpec`](framework/interface.py): index, label, mods, budget,
and two bound env factories — `make_train_env(seed, num_envs)` (episodic life,
clipped reward) and `make_eval_env(seed)` (true episodes, raw reward). Agents
never parse `TASK_MODS` or wire up wrappers themselves. The full task list
(budgets included) is available upfront as `self.tasks`.

The runner drives every agent identically:

```
floor_state = agent.init_state(PRNGKey(EVAL_SEED))          # R_rand floor
state       = agent.init_state(PRNGKey(SEED))
for task in tasks:
    state = agent.train_task(state, task, ctx)
    R[i, j] <- evaluate(agent.policy(state, j, trained_task=i))   for j <= i
```

## The smallest complete agent

[agents/random_policy.py](agents/random_policy.py) is a full working
submission in ~30 lines. A skeleton:

```python
import jax
from agents import register_agent
from framework.interface import ContinualAgent

@register_agent               # optional: makes it selectable as AGENT=my_dqn
class MyDQN(ContinualAgent):
    name = "my_dqn"

    def init_state(self, key):
        env = self.tasks[0].make_train_env(0, 1)   # shapes/spaces
        return {"params": ..., "opt_state": ..., "replay": ...}

    def train_task(self, state, task, ctx):
        env = task.make_train_env(self.config["SEED"], num_envs=...)
        # your training loop, <= task.budget env steps
        return new_state

    def policy(self, state, eval_task, trained_task):
        def act(obs, key):                          # obs: (1, *obs_shape)
            q = self._network.apply(state["params"], obs)
            return q.argmax(axis=-1), key           # action: (1,)
        return act
```

For a full-featured example — CL state threaded across tasks, per-task eval
policies (PackNet subnetworks), wandb offsets, the mid-training CRL curve —
read the reference implementation [agents/ppo/agent.py](agents/ppo/agent.py).

## Running it

One call, from Python:

```python
from framework import run_benchmark
from agents.my_dqn import MyDQN

result = run_benchmark(
    agent=MyDQN,                  # class, instance, or registry name ("ppo", "random")
    sequence="pong_dyn4",         # which tasks     (config/sequence/)
    modality="oc",                # which obs space + budget (config/modality/)
    overrides=["SEED=1", "EXP_NAME=my_dqn_dyn4"],
)
result["R"]; result["Retention"]; result["mean_forgetting"]; result["run_dir"]
```

or the CLI (registered agents only):

```bash
python ppo_crl_continual.py AGENT=my_dqn sequence=pong_dyn4 modality=oc
```

Everything the pre-framework benchmark wrote is still written: matrices to
`runs/<name>/matrix.{json,npz}`, per-task checkpoints, wandb metrics with
`TRACK=True`. Report **mean ± std over seeds** (see README): run the same call
with several `SEED`s, or use `tools/run_campaign.py`.

**Naming:** `EXP_NAME` defaults to `${CL_METHOD}_${SEQUENCE}`, which is right
for the reference PPO agent only — override it (as above) for custom agents so
run directories and wandb groups don't collide with PPO runs.

## Ground rules (fairness)

The harness cannot meter the env steps your training loop consumes, so these
are enforced by convention — but they are what makes results comparable:

1. `train_task` uses **at most `task.budget` env steps** (frame-skipped steps,
   the unit the reference PPO counts), and only envs from `task.make_train_env`.
2. Official numbers come **only** from the harness evaluating your `policy`;
   `make_eval_env` exists for your own probes, nothing more.
3. No peeking at future tasks' envs during training (the task list metadata —
   budgets, labels — is fair game for scheduling).
4. Everything inside `policy`'s act function must be jit-traceable JAX.
5. Don't change harness-owned config (eval protocol, budgets, sequences) in a
   submission; tune only your own hyperparameters.

## What goes where

```
framework/            the harness (don't change in a submission)
├── interface.py        the contract: ContinualAgent, TaskSpec, TrainContext
├── evaluation.py       evaluate_policy: the official rollout protocol
├── runner.py           task construction, benchmark loop, metrics, outputs
└── envs.py             make_env: the wrapped JAXtari env factory

agents/               submissions (add yours here)
├── __init__.py         @register_agent registry
├── random_policy.py    minimal example                        (AGENT=random)
└── ppo/                reference: PPO + ft/ewc/agem/packnet   (AGENT=ppo)
    ├── agent.py          the ContinualAgent adapter
    ├── trainer.py        the jitted single-task PPO loop
    ├── networks.py       CNN/MLP torsos + Actor/Critic heads
    ├── eval.py           checkpoint-based eval + curve eval
    └── continual/        CLMethod base + one file per method
```

The reference PPO agent's internals under `agents/ppo/` are the pre-framework
modules, moved but unchanged; the refactor is verified bit-exact against the
pre-framework orchestrator for ft, ewc, and packnet (matrices and checkpoint
params).
