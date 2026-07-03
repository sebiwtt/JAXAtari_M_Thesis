# =============================================================================
# Algorithm-agnostic continual-RL benchmark harness.
#
# `run_continual(config, algo)` trains ONE agent sequentially over ordered
# single-mod tasks, carrying params forward (naive finetuning), and fills:
#
#   R[i, j]         = return of the agent trained through task i, evaluated on task j  (j <= i)
#   R_rand[j]       = return of a fresh/untrained agent on task j - the "knows nothing" floor
#   Retention[i, j] = (R[i, j] - R_rand[j]) / (R[j, j] - R_rand[j])                     (j <= i)
#                     1.0 = matches post-task-j performance, 0.0 = performs like random.
#
# EVAL_FULL_MATRIX also fills j > i: forward transfer to not-yet-trained tasks.
#
# The algorithm (PPO today, e.g. PQN later) plugs in as a `CRLAlgorithm` bundle;
# this file is host-side Python orchestration only - each task's training runs
# whatever jit/scan/vmap machinery the algorithm implements internally.
# =============================================================================

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import wandb

from crl_eval import evaluate_policy


@dataclass(frozen=True)
class CRLAlgorithm:
    """Algorithm bundle plugged into `run_continual`. `params` is an opaque pytree the
    harness only carries between calls, never inspects.

    init_params(config, seed) -> params
        Fresh untrained params (used as the random-agent floor R_rand).
    train(config, init_params=None, run_name=..., manage_wandb=False,
          wandb_step_offset=..., wandb_group=...) -> params
        One single-task training run. Trains on config["TRAIN_MODS"] (set by the
        harness per task), resumes from init_params when given, and must not
        init/finish wandb itself when manage_wandb=False.
    save_checkpoint(path, config, params) -> None
        Serialize params so this bundle's own `load_policy` can load them.
    load_policy(model_path, config) -> act_fn
        Rebuild the policy from a checkpoint; act_fn(obs, key) -> (action, key)
        picks one action for a single batch-1 observation. Everything else about
        evaluation - env construction, eval seeding, rollout, scoring - is
        harness-owned (crl_eval.evaluate_policy), so algorithms cannot diverge
        on the measurement protocol.
    """
    name: str
    init_params: Callable[[dict, int], Any]
    train: Callable[..., Any]
    save_checkpoint: Callable[[str, dict, Any], None]
    load_policy: Callable[[str, dict], Callable]


def _task_label(mods) -> str:
    return "base" if len(mods) == 0 else str(mods[0])


def _print_matrix(name: str, M: np.ndarray, labels: list[str]) -> None:
    print(f"\n{name}:")
    col_w = max(10, max(len(l) for l in labels) + 2)
    print(" " * col_w + "".join(f"{l:>{col_w}}" for l in labels))
    for i, row_label in enumerate(labels):
        row = "".join(
            f"{M[i, j]:>{col_w}.3f}" if not np.isnan(M[i, j]) else f"{'--':>{col_w}}"
            for j in range(M.shape[1])
        )
        print(f"{row_label:>{col_w}}" + row)


def _print_vector(name: str, v: np.ndarray, labels: list[str]) -> None:
    print(f"\n{name}:")
    for label, value in zip(labels, v):
        print(f"  {label:>12}: {value:.3f}")


def _eval_cell(act_fn, config: dict, mods: list, cell_name: str) -> float:
    """One matrix cell: evaluate a loaded policy on one task's mods, with completion check."""
    episodic_returns, completed = evaluate_policy(act_fn, config, mods)
    n_completed = int(np.sum(completed))
    if n_completed < completed.shape[0]:
        print(
            f"[CRL] WARNING: {cell_name} only {n_completed}/{completed.shape[0]} eval episodes "
            f"completed within the eval scan window; this value may be inflated."
        )
    return float(np.mean(episodic_returns))


def run_continual(config: dict, algo: CRLAlgorithm) -> None:
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    task_mods_list = [list(m) for m in config["TASK_MODS"]]
    assert len(task_mods_list) > 0, "TASK_MODS must contain at least one task"
    assert len(task_mods_list[0]) == 0, "TASK_MODS[0] must be the base task (no mods)"
    for i, mods in enumerate(task_mods_list):
        assert len(mods) <= 1, f"CRL tasks must use at most one mod each; TASK_MODS[{i}]={mods} has {len(mods)}"
    num_tasks = len(task_mods_list)
    labels = [_task_label(m) for m in task_mods_list]

    # Mirrors the trainer's own derivation, needed here for the wandb step offset.
    batch_size = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    num_iterations = int(config["TOTAL_TIMESTEPS"] // batch_size)

    group_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}'
    base_run_name = f'{group_name}_{config["SEED"]}'
    run_dir = f"runs/{base_run_name}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"[CRL] algorithm: {algo.name}")

    if config["TRACK"]:
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=base_run_name,
            group=group_name,  # groups seed replicates of the same sweep in the wandb UI
            save_code=True,
        )

    # Random-agent floor R_rand[j]: one eval pass per task, no training. Keyed off
    # EVAL_SEED (not SEED) so it's independent of the training seed.
    rand_params = algo.init_params(config, config["EVAL_SEED"])
    rand_ckpt_path = f"{run_dir}/random_agent.cleanrl_model"
    algo.save_checkpoint(rand_ckpt_path, config, rand_params)
    print(f"[CRL] random-agent baseline checkpoint saved to {rand_ckpt_path}")

    R_rand = np.full(num_tasks, np.nan)
    rand_act = algo.load_policy(rand_ckpt_path, config)
    for j in range(num_tasks):
        R_rand[j] = _eval_cell(rand_act, config, task_mods_list[j], f"R_rand[{j}]")
        print(f"[CRL] R_rand[{j}] (random agent on task {j}={labels[j]!r}) = {R_rand[j]:.3f}")

    R = np.full((num_tasks, num_tasks), np.nan)
    ckpt_paths: list[str] = []
    carried_params = None

    for i in range(num_tasks):
        task_mods = task_mods_list[i]
        task_config = dict(config)
        task_config["TRAIN_MODS"] = tuple(task_mods)

        task_run_name = f"{base_run_name}_task{i}"
        print(f"\n=== CRL task {i}/{num_tasks - 1}: mods={task_mods} (label={labels[i]!r}) ===")
        carried_params = algo.train(
            task_config,
            init_params=carried_params,
            run_name=task_run_name,
            manage_wandb=False,
            wandb_step_offset=i * num_iterations,
            wandb_group=labels[i],
        )

        ckpt_path = f"{run_dir}/task_{i}.cleanrl_model"
        algo.save_checkpoint(ckpt_path, task_config, carried_params)
        print(f"[CRL] task {i} checkpoint saved to {ckpt_path}")
        ckpt_paths.append(ckpt_path)

        # j <= i: retention (tasks already trained on). EVAL_FULL_MATRIX also fills
        # j > i: forward transfer to tasks not yet trained on.
        task_act = algo.load_policy(ckpt_path, config)
        eval_js = range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)
        for j in eval_js:
            R[i, j] = _eval_cell(task_act, config, task_mods_list[j], f"R[{i},{j}]")
            kind = "forward transfer" if j > i else "retention"
            print(f"[CRL] R[{i},{j}] ({kind}: train through task {i}={labels[i]!r}, eval on task {j}={labels[j]!r}) = {R[i, j]:.3f}")

    diag = np.diag(R)  # R[j, j], populated before it's needed as a denominator
    Retention = np.full((num_tasks, num_tasks), np.nan)
    for i in range(num_tasks):
        for j in (range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)):
            denom = diag[j] - R_rand[j]
            if denom == 0:
                print(
                    f"[CRL] WARNING: R[{j},{j}]={diag[j]:.3f} equals R_rand[{j}]={R_rand[j]:.3f}; "
                    f"Retention[{i},{j}] is undefined (0/0), leaving as NaN."
                )
                continue
            Retention[i, j] = (R[i, j] - R_rand[j]) / denom

    _print_matrix("R (mean return)", R, labels)
    _print_vector("R_rand (random-agent floor)", R_rand, labels)
    _print_matrix("Retention ((R[i,j] - R_rand[j]) / (R[j,j] - R_rand[j]))", Retention, labels)

    np.savez(
        f"{run_dir}/matrix.npz",
        R=R,
        R_rand=R_rand,
        Retention=Retention,
        task_mods=np.array([json.dumps(m) for m in task_mods_list]),
        labels=np.array(labels),
    )
    with open(f"{run_dir}/matrix.json", "w") as f:
        json.dump(
            {
                "algorithm": algo.name,
                "task_mods": task_mods_list,
                "labels": labels,
                "R": R.tolist(),
                "R_rand": R_rand.tolist(),
                "Retention": Retention.tolist(),
                "checkpoints": ckpt_paths,
                "random_agent_checkpoint": rand_ckpt_path,
            },
            f,
            indent=2,
        )
    print(f"\n[CRL] matrix saved to {run_dir}/matrix.npz and {run_dir}/matrix.json")

    if config["TRACK"]:
        for j in range(num_tasks):
            wandb.log({f"crl/R_rand/{j}": R_rand[j]})
        for i in range(num_tasks):
            for j in (range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)):
                wandb.log({f"crl/R/{i}_{j}": R[i, j], f"crl/retention/{i}_{j}": Retention[i, j]})
            wandb.log({f"crl/diag/{i}": R[i, i]})
        wandb.finish()
