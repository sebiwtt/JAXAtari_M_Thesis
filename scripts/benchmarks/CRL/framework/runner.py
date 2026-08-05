# =============================================================================
# Agent-agnostic continual-RL benchmark runner
# =============================================================================
# Trains ONE agent sequentially over ordered single-mod tasks and fills the
# retention/forgetting matrix. The agent is anything implementing
# framework.interface.ContinualAgent; the reference PPO(+ft/ewc/agem/packnet)
# lives in agents/ppo_crl.py. The runner owns what must be identical across
# submissions: task construction, budgets, the eval protocol, the metrics, and
# the output files.
#
#   R[i, j]         = return of the agent trained through task i, evaluated on task j  (j <= i)
#   R_rand[j]       = return of a fresh/untrained agent on task j - the "knows nothing" floor,
#                     not 0 (Pong's random-policy floor is close to -21)
#   Retention[i, j] = clip((R[i,j] - R_rand[j]) / (R[j,j] - R_rand[j]), 0, 1)         (j <= i)
#                     1.0 = matches post-task-j performance, 0.0 = at/below random.
#   Drop[i, j]      = 1 - Retention[i, j]                                            (j < i)
#                     MEAL/COOM-style: the agent is compared to itself, not to a fixed
#                     converged reference - R_rand only rescales/clamps the range.
#   Forgetting[j]   = recency-weighted mean of Drop[i, j] over later checkpoints i > j:
#                     weight(i) = exp(-FORGETTING_LAMBDA * (i-j)/(last_task-j)). The last
#                     task has no later checkpoint -> NaN by design.
#   mean_forgetting = nanmean(Forgetting); mean_retention = 1 - mean_forgetting (its exact
#                     complement - NOT a flat per-cell average of Retention, which would
#                     over-weight earlier tasks).
#
# Entry points:
#   run_benchmark(agent=..., sequence=..., method=..., modality=..., overrides=[...])
#       - the one-call API: composes the hydra config in-process and runs.
#   run_from_config(config, agent=None)
#       - same, from an already-composed flat config dict (the CLI path).
# =============================================================================

import json
import os
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np
import wandb

from framework.evaluation import evaluate_policy
from framework.interface import ContinualAgent, TaskSpec, TrainContext

CRL_ROOT = Path(__file__).resolve().parent.parent


def _task_label(mods) -> str:
    return "base" if len(mods) == 0 else str(mods[0])


def _print_matrix(name: str, M: np.ndarray, labels: "list[str]") -> None:
    print(f"\n{name}:")
    col_w = max(10, max(len(l) for l in labels) + 2)
    print(" " * col_w + "".join(f"{l:>{col_w}}" for l in labels))
    for i, row_label in enumerate(labels):
        row = "".join(
            f"{M[i, j]:>{col_w}.3f}" if not np.isnan(M[i, j]) else f"{'--':>{col_w}}"
            for j in range(M.shape[1])
        )
        print(f"{row_label:>{col_w}}" + row)


def _print_vector(name: str, v: np.ndarray, labels: "list[str]") -> None:
    print(f"\n{name}:")
    for label, value in zip(labels, v):
        print(f"  {label:>12}: {value:.3f}")


def _resolve_task_budgets(config: dict, num_tasks: int) -> "list[int]":
    """Per-task training budget: BASE_TIMESTEP_BUDGET for the base task
    (TASK_MODS[0]) and TASK_TIMESTEP_BUDGET for every mod task. A list of
    length len(TASK_MODS) sets every task explicitly (BASE_TIMESTEP_BUDGET is
    then unused). Algorithm-specific feasibility (e.g. a budget below one PPO
    batch) is checked by the agent itself, which knows its batching."""
    base_budget = int(config["BASE_TIMESTEP_BUDGET"])
    spec = config.get("TASK_TIMESTEP_BUDGET")
    if spec is None:
        budgets = [base_budget] * num_tasks
    elif isinstance(spec, (list, tuple)):
        assert len(spec) == num_tasks, (
            f"TASK_TIMESTEP_BUDGET has {len(spec)} entries but TASK_MODS has {num_tasks} tasks"
        )
        budgets = [int(s) for s in spec]
    else:
        budgets = [base_budget] + [int(spec)] * (num_tasks - 1)
    for i, b in enumerate(budgets):
        assert b >= 1, f"task {i} budget must be positive, got {b:,}"
    return budgets


def build_tasks(config: dict) -> "list[TaskSpec]":
    """The official task sequence for a config: one TaskSpec per TASK_MODS entry,
    with bound train/eval env factories so agents never touch env wiring."""
    from envs import make_env  # local import: envs pulls in jaxatari

    task_mods_list = [list(m) for m in config["TASK_MODS"]]
    assert len(task_mods_list) > 0, "TASK_MODS must contain at least one task"
    assert len(task_mods_list[0]) == 0, "TASK_MODS[0] must be the base task (no mods)"
    for i, mods in enumerate(task_mods_list):
        assert len(mods) <= 1, f"CRL tasks must use at most one mod each; TASK_MODS[{i}]={mods} has {len(mods)}"

    budgets = _resolve_task_budgets(config, len(task_mods_list))
    env_kwargs = dict(
        pixel_based=config["PIXEL_BASED"],
        native_downscaling=config["NATIVE_DOWNSCALING"],
        smooth_image=config["SMOOTH_IMAGE"],
        grayscale=config["GRAYSCALE"],
    )

    def make_train_env(mods, seed, num_envs):
        return make_env(config["ENV_ID"], seed, num_envs, mods=list(mods), eval=False, **env_kwargs)()

    def make_eval_env(mods, seed):
        return make_env(config["ENV_ID"], seed, 1, mods=list(mods), eval=True, **env_kwargs)()

    return [
        TaskSpec(
            index=i,
            label=_task_label(mods),
            mods=tuple(mods),
            budget=budgets[i],
            env_id=config["ENV_ID"],
            make_train_env=partial(make_train_env, tuple(mods)),
            make_eval_env=partial(make_eval_env, tuple(mods)),
        )
        for i, mods in enumerate(task_mods_list)
    ]


def _resolve_agent(agent, config: dict, tasks: "list[TaskSpec]") -> ContinualAgent:
    """Accept an agent instance, a ContinualAgent subclass, a registry name, or
    None (-> config's AGENT key, default "ppo")."""
    if isinstance(agent, ContinualAgent):
        return agent
    if isinstance(agent, type) and issubclass(agent, ContinualAgent):
        return agent(config, tasks)
    from agents import make_agent  # lazy: keeps framework importable without agents/

    name = agent if isinstance(agent, str) else str(config.get("AGENT", "ppo"))
    return make_agent(name, config, tasks)


def _eval_cell(agent, state, tasks, eval_task: int, trained_task: int, config: dict):
    """One official eval: the agent's policy for (eval_task, trained_task) rolled
    out on eval_task's env. Returns (mean_return, n_completed, n_episodes)."""
    env = tasks[eval_task].make_eval_env(config["EVAL_SEED"])
    act = agent.policy(state, eval_task, trained_task)
    episodic_returns, _, completed = evaluate_policy(
        act, env, eval_episodes=config["EVAL_EPISODES"], seed=config["EVAL_SEED"]
    )
    episodic_returns = np.asarray(jax.device_get(episodic_returns))
    completed = np.asarray(jax.device_get(completed))
    return float(episodic_returns.mean()), int(completed.sum()), int(completed.shape[0])


def run_from_config(config: dict, agent=None) -> dict:
    """Run the full benchmark for one composed config; returns the summary dict
    that is also written to <run_dir>/matrix.json."""
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    if config.get("EVAL_SEED") is None:
        # Derived (not literally 0/1/2/...) so replicate seeds don't share identical eval
        # noise, while staying reproducible from SEED alone. Still fixed across every
        # matrix cell within a single run.
        config["EVAL_SEED"] = config["SEED"] * 12 + 1

    tasks = build_tasks(config)
    num_tasks = len(tasks)
    labels = [t.label for t in tasks]
    budgets = [t.budget for t in tasks]
    env_step_offsets = [int(o) for o in np.cumsum([0] + budgets[:-1])]

    # Validates the agent (and, for the reference PPO, CL_METHOD + its config
    # keys and budget feasibility) before any training starts.
    agent = _resolve_agent(agent, config, tasks)

    group_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}'
    base_run_name = f'{group_name}_{config["SEED"]}'
    run_dir = str(CRL_ROOT / "runs" / base_run_name)
    os.makedirs(run_dir, exist_ok=True)

    if config["TRACK"]:
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=base_run_name,
            group=group_name,  # groups seed replicates of the same sweep in the wandb UI
            save_code=True,
        )

    # Random-agent floor R_rand[j]: a freshly-initialized (untrained) agent, one
    # eval pass per task, no training. Keyed off EVAL_SEED (not SEED) so it's
    # independent of the training seed.
    floor_state = agent.init_state(jax.random.PRNGKey(config["EVAL_SEED"]))
    rand_ckpt_path = agent.save_checkpoint(floor_state, run_dir, "random_agent")
    print(f"[CRL] random-agent baseline checkpoint saved to {rand_ckpt_path}")

    R_rand = np.full(num_tasks, np.nan)
    eval_time_rand = 0.0
    for j in range(num_tasks):
        eval_t0 = time.perf_counter()
        mean_ret, n_completed, n_episodes = _eval_cell(agent, floor_state, tasks, j, -1, config)
        eval_time_rand += time.perf_counter() - eval_t0
        if n_completed < n_episodes:
            print(
                f"[CRL] WARNING: R_rand[{j}] only {n_completed}/{n_episodes} eval episodes "
                f"completed within the eval scan window; this floor value may be inflated."
            )
        R_rand[j] = mean_ret
        print(f"[CRL] R_rand[{j}] (random agent on task {j}={labels[j]!r}) = {R_rand[j]:.3f}")

    R = np.full((num_tasks, num_tasks), np.nan)
    ckpt_paths: "list[str]" = []
    state = agent.init_state(jax.random.PRNGKey(config["SEED"]))
    train_time_per_task = np.full(num_tasks, np.nan)
    eval_time_matrix = 0.0

    for i, task in enumerate(tasks):
        ctx = TrainContext(
            run_name=f"{base_run_name}_task{i}",
            run_dir=run_dir,
            env_step_offset=env_step_offsets[i],
            track=bool(config["TRACK"]),
        )
        print(
            f"\n=== CRL task {i}/{num_tasks - 1}: mods={list(task.mods)} (label={task.label!r}, "
            f"agent={agent.name}, {task.budget:,} steps) ==="
        )
        train_t0 = time.perf_counter()
        state = agent.train_task(state, task, ctx)
        jax.block_until_ready(state)
        train_time_per_task[i] = time.perf_counter() - train_t0

        ckpt_path = agent.save_checkpoint(state, run_dir, f"task_{i}")
        print(f"[CRL] task {i} checkpoint saved to {ckpt_path}")
        ckpt_paths.append(ckpt_path)

        # j <= i: retention (tasks already trained on). EVAL_FULL_MATRIX also fills
        # j > i: forward transfer to tasks not yet trained on. Agents may substitute
        # task-specific eval policies (e.g. PackNet subnetworks) via policy().
        eval_js = range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)
        for j in eval_js:
            eval_t0 = time.perf_counter()
            mean_ret, n_completed, n_episodes = _eval_cell(agent, state, tasks, j, i, config)
            eval_time_matrix += time.perf_counter() - eval_t0
            if n_completed < n_episodes:
                print(
                    f"[CRL] WARNING: R[{i},{j}] only {n_completed}/{n_episodes} eval episodes "
                    f"completed within the eval scan window; this cell's mean return may be inflated."
                )
            R[i, j] = mean_ret
            kind = "forward transfer" if j > i else "forgetting"
            print(f"[CRL] R[{i},{j}] ({kind}: train through task {i}={labels[i]!r}, eval on task {j}={labels[j]!r}) = {R[i, j]:.3f}")

    agent.save_artifacts(state, run_dir)

    # Retention (clamped to [0,1]) and its complement Drop, from which the
    # recency-weighted Forgetting[j] is derived - all from the R/R_rand data above.
    diag = np.diag(R)  # R[j, j]: performance right after finishing task j
    Retention = np.full((num_tasks, num_tasks), np.nan)
    Drop = np.full((num_tasks, num_tasks), np.nan)
    for i in range(num_tasks):
        for j in (range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)):
            denom = diag[j] - R_rand[j]
            if denom <= 0:
                print(
                    f"[CRL] WARNING: R[{j},{j}]={diag[j]:.3f} <= R_rand[{j}]={R_rand[j]:.3f}; "
                    f"Retention[{i},{j}] is undefined, leaving as NaN."
                )
                continue
            Retention[i, j] = float(np.clip((R[i, j] - R_rand[j]) / denom, 0.0, 1.0))
            if j < i:  # Drop/forgetting only defined for already-trained tasks
                Drop[i, j] = 1.0 - Retention[i, j]

    # Recency weighting: later checkpoints (further past task j's own training)
    # count less than checkpoints reached soon after. FORGETTING_LAMBDA=0 recovers the
    # plain unweighted mean (all weights equal to 1).
    forgetting_lambda = float(config.get("FORGETTING_LAMBDA", 1.0))
    last_task = num_tasks - 1
    Forgetting = np.full(num_tasks, np.nan)
    for j in range(num_tasks - 1):
        later_i = np.arange(j + 1, num_tasks)
        later_drops = Drop[later_i, j]
        valid = ~np.isnan(later_drops)
        if not valid.any():
            continue
        later_i, later_drops = later_i[valid], later_drops[valid]
        weights = np.exp(-forgetting_lambda * (later_i - j) / (last_task - j))
        Forgetting[j] = float(np.sum(weights * later_drops) / np.sum(weights))
    mean_forgetting = float(np.nanmean(Forgetting)) if num_tasks > 1 else float("nan")
    # Same per-task-equal-weight averaging as mean_forgetting, so this is exactly its
    # complement - not the same as a flat per-cell average over Retention (which would
    # implicitly weight tasks with more later checkpoints more heavily).
    mean_retention = 1.0 - mean_forgetting

    _print_matrix("R (mean return)", R, labels)
    _print_vector("R_rand (random-agent floor)", R_rand, labels)
    _print_matrix("Retention (clamped to [0,1])", Retention, labels)
    _print_vector(f"Forgetting (per task, recency-weighted avg drop, lambda={forgetting_lambda})", Forgetting, labels)
    print(f"\n[CRL] mean forgetting across tasks: {mean_forgetting:.4f}  (mean retention: {mean_retention:.4f})")

    # Wall-clock, GPU-inclusive: train_task/evaluate_policy are synchronous Python
    # calls followed by a device sync (jax.block_until_ready/device_get) before the
    # timer stops, so timing the call site captures real compute time.
    total_train_time = float(np.nansum(train_time_per_task))
    total_eval_time = float(eval_time_rand + eval_time_matrix)
    total_compute_time = total_train_time + total_eval_time
    print(
        f"\n[CRL] compute time: train={total_train_time:.1f}s, "
        f"eval={total_eval_time:.1f}s (rand={eval_time_rand:.1f}s, matrix={eval_time_matrix:.1f}s), "
        f"total={total_compute_time:.1f}s ({total_compute_time / 3600:.2f} h)"
    )

    # Columnar crl_curve dump (empty when the agent produced none): one continuous
    # series over all tasks; see config.yaml CRL_CURVE.
    crl_curve_points = agent.collect_curve_points()
    crl_curve_cols = {}
    if crl_curve_points:
        crl_curve_cols = {
            key: [p[key] for p in crl_curve_points]
            for key in ("wandb_step", "env_step", "task_idx", "task_label", "source", "base_return", "completed_frac")
        }
        print(f"[CRL] crl_curve: {len(crl_curve_points)} points collected across {num_tasks} tasks.")

    np.savez(
        f"{run_dir}/matrix.npz",
        **{f"crl_curve_{key}": np.array(col) for key, col in crl_curve_cols.items()},
        R=R,
        R_rand=R_rand,
        Retention=Retention,
        Drop=Drop,
        Forgetting=Forgetting,
        mean_forgetting=np.array(mean_forgetting),
        mean_retention=np.array(mean_retention),
        task_mods=np.array([json.dumps(list(t.mods)) for t in tasks]),
        labels=np.array(labels),
        task_timesteps=np.array(budgets),
        env_id=np.array(config["ENV_ID"]),
        exp_name=np.array(config["EXP_NAME"]),
        agent=np.array(agent.name),
        train_time_per_task=train_time_per_task,
        eval_time_rand=np.array(eval_time_rand),
        eval_time_matrix=np.array(eval_time_matrix),
        total_compute_time=np.array(total_compute_time),
    )
    summary = {
        "env_id": config["ENV_ID"],
        "exp_name": config["EXP_NAME"],
        "agent": agent.name,
        **agent.describe(),
        "task_mods": [list(t.mods) for t in tasks],
        "labels": labels,
        "task_timesteps": budgets,
        "R": R.tolist(),
        "R_rand": R_rand.tolist(),
        "Retention": Retention.tolist(),
        "Drop": Drop.tolist(),
        "Forgetting": Forgetting.tolist(),
        "mean_forgetting": mean_forgetting,
        "mean_retention": mean_retention,
        "checkpoints": ckpt_paths,
        "random_agent_checkpoint": rand_ckpt_path,
        "compute_time_sec": {
            "train_per_task": train_time_per_task.tolist(),
            "eval_rand": eval_time_rand,
            "eval_matrix": eval_time_matrix,
            "train_total": total_train_time,
            "eval_total": total_eval_time,
            "total": total_compute_time,
        },
    }
    if crl_curve_cols:
        summary["crl_curve"] = crl_curve_cols
    with open(f"{run_dir}/matrix.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[CRL] matrix saved to {run_dir}/matrix.npz and {run_dir}/matrix.json")

    if config["TRACK"]:
        for j in range(num_tasks):
            wandb.log({f"crl/R_rand/{j}": R_rand[j]})
        for i in range(num_tasks):
            for j in (range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)):
                wandb.log({f"crl/R/{i}_{j}": R[i, j], f"crl/retention/{i}_{j}": Retention[i, j]})
            wandb.log({f"crl/diag/{i}": R[i, i]})
        for j in range(num_tasks):
            if not np.isnan(Forgetting[j]):
                wandb.log({f"crl/forgetting/{j}": Forgetting[j]})
        wandb.log({
            "crl/mean_forgetting": mean_forgetting,
            "crl/mean_retention": mean_retention,
            "crl/compute_time/train_total_sec": total_train_time,
            "crl/compute_time/eval_total_sec": total_eval_time,
            "crl/compute_time/total_sec": total_compute_time,
        })
        wandb.finish()

    summary["run_dir"] = run_dir
    return summary


def run_benchmark(
    agent=None,
    *,
    sequence: "str | None" = None,
    method: "str | None" = None,
    modality: "str | None" = None,
    overrides: "list[str] | None" = None,
    config: "dict | None" = None,
) -> dict:
    """THE single entry point: compose the benchmark config and run it.

        from framework import run_benchmark
        result = run_benchmark(agent=MyAgent, sequence="pong_dyn4", modality="oc",
                               overrides=["SEED=1", "TRACK=False"])

    `agent`: a ContinualAgent subclass/instance, a registry name ("ppo",
    "random", ...), or None to use the config's AGENT key. `sequence` accepts
    the flat spelling ("pong_dyn4"). `overrides` are hydra-style "KEY=value"
    strings applied on top. Pass an already-composed flat `config` dict to skip
    hydra entirely (the group args are then ignored).

    Returns the matrix.json summary dict, plus "run_dir".
    """
    if config is None:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from tools.config_groups import resolve_sequence

        ov = list(overrides or [])
        if sequence is not None:
            ov.insert(0, f"sequence={resolve_sequence(sequence)}")
        if method is not None:
            ov.insert(0, f"method={method}")
        if modality is not None:
            ov.insert(0, f"modality={modality}")
        with initialize_config_dir(config_dir=str(CRL_ROOT / "config"), version_base=None):
            cfg = compose(config_name="config", overrides=ov)
        config = OmegaConf.to_container(cfg, resolve=True)
        print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    return run_from_config(config, agent=agent)
