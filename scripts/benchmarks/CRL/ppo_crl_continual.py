# =============================================================================
# Naive-finetuning continual-RL evaluation harness for the JAXtari PPO trainer
# =============================================================================
# Trains ONE agent sequentially over an ordered list of single-mod Pong tasks,
# carrying parameters forward across tasks (naive finetuning: no replay, no
# regularization, no architecture masking - the simplest possible CL baseline).
# After each task, evaluates the CURRENT agent on every task seen so far to fill
# in one row of a lower-triangular retention matrix:
#
#   R[i, j]         = mean return of the agent trained through task i, evaluated on task j  (j <= i)
#   Retention[i, j] = R[i, j] / R[j, j]                                                       (j <= i)
#
# This file only orchestrates; the actual PPO loop lives in `ppo_crl_finetune.train`
# and evaluation lives in `ppo_crl_eval.evaluate` - both are reused unmodified in
# spirit (train() gained an optional `init_params` resume path, evaluate() gained an
# episode-completion signal, see their docstrings/comments).
# =============================================================================

import json
import os
from functools import partial

import flax
import hydra
import jax
import numpy as np
import wandb
from omegaconf import OmegaConf

from ppo_eval import evaluate
from ppo_trainer import Actor, Critic, MLP_Network, Network, make_env, train


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


def run_continual(config: dict) -> None:
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    task_mods_list = [list(m) for m in config["TASK_MODS"]]
    assert len(task_mods_list) > 0, "TASK_MODS must contain at least one task"
    assert len(task_mods_list[0]) == 0, "TASK_MODS[0] must be the base task (no mods)"
    for i, mods in enumerate(task_mods_list):
        assert len(mods) <= 1, f"CRL tasks must use at most one mod each; TASK_MODS[{i}]={mods} has {len(mods)}"
    num_tasks = len(task_mods_list)
    labels = [_task_label(m) for m in task_mods_list]

    # Mirror `train()`'s own derivation so the wandb step offset per task is known
    # up front; `train()` recomputes the same thing internally from the same config.
    batch_size = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    num_iterations = int(config["TOTAL_TIMESTEPS"] // batch_size)

    base_run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'
    run_dir = f"runs/{base_run_name}"
    os.makedirs(run_dir, exist_ok=True)

    if config["TRACK"]:
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=base_run_name,
            save_code=True,
        )

    Model = (Network, Actor, Critic) if config["PIXEL_BASED"] else (MLP_Network, Actor, Critic)

    R = np.full((num_tasks, num_tasks), np.nan)
    ckpt_paths: list[str] = []
    carried_params = None

    for i in range(num_tasks):
        task_mods = task_mods_list[i]
        task_config = dict(config)
        # Both TRAIN_MODS and EVAL_MODS drive `train()`'s own (task-namespaced)
        # periodic-eval/video/SAVE_MODEL side effects; pin them to this task's single
        # mod so that mid-training housekeeping matches what's actually being trained.
        task_config["TRAIN_MODS"] = tuple(task_mods)
        task_config["EVAL_MODS"] = tuple(task_mods)

        task_run_name = f"{base_run_name}_task{i}"
        print(f"\n=== CRL task {i}/{num_tasks - 1}: mods={task_mods} (label={labels[i]!r}) ===")
        carried_params = train(
            task_config,
            init_params=carried_params,
            run_name=task_run_name,
            manage_wandb=False,
            wandb_step_offset=i * num_iterations,
        )

        # Orchestrator's own checkpoint of the agent after task i, used to fill row i
        # of the retention matrix. Same serialization format as `train()`'s internal
        # `eval_and_vid` so the existing `evaluate()` can load it unmodified.
        ckpt_path = f"{run_dir}/task_{i}.cleanrl_model"
        with open(ckpt_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        task_config,
                        [carried_params.network_params, carried_params.actor_params, carried_params.critic_params],
                    ]
                )
            )
        print(f"[CRL] task {i} checkpoint saved to {ckpt_path}")
        ckpt_paths.append(ckpt_path)

        for j in range(i + 1):
            eval_mods = task_mods_list[j]
            episodic_returns, _, completed = evaluate(
                model_path=ckpt_path,
                make_env=partial(
                    make_env,
                    mods=eval_mods,
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    smooth_image=config["SMOOTH_IMAGE"],
                    eval=True,
                ),
                env_id=config["ENV_ID"],
                eval_episodes=config["EVAL_EPISODES"],
                run_name=f"{base_run_name}-eval-{i}-{j}",
                Model=Model,
                seed=config["EVAL_SEED"],
            )
            episodic_returns = np.asarray(jax.device_get(episodic_returns))
            completed = np.asarray(jax.device_get(completed))
            n_completed = int(completed.sum())
            if n_completed < completed.shape[0]:
                print(
                    f"[CRL] WARNING: R[{i},{j}] only {n_completed}/{completed.shape[0]} eval episodes "
                    f"completed within the eval scan window; this cell's mean return may be inflated."
                )
            R[i, j] = float(episodic_returns.mean())
            print(f"[CRL] R[{i},{j}] (train through task {i}={labels[i]!r}, eval on task {j}={labels[j]!r}) = {R[i, j]:.3f}")

    diag = np.diag(R)  # R[j, j], fully populated by the time any row needs it as a denominator
    Retention = np.full((num_tasks, num_tasks), np.nan)
    for i in range(num_tasks):
        for j in range(i + 1):
            Retention[i, j] = R[i, j] / diag[j]

    _print_matrix("R (mean return)", R, labels)
    _print_matrix("Retention (R[i,j] / R[j,j])", Retention, labels)

    np.savez(
        f"{run_dir}/matrix.npz",
        R=R,
        Retention=Retention,
        task_mods=np.array([json.dumps(m) for m in task_mods_list]),
        labels=np.array(labels),
    )
    with open(f"{run_dir}/matrix.json", "w") as f:
        json.dump(
            {
                "task_mods": task_mods_list,
                "labels": labels,
                "R": R.tolist(),
                "Retention": Retention.tolist(),
                "checkpoints": ckpt_paths,
            },
            f,
            indent=2,
        )
    print(f"\n[CRL] matrix saved to {run_dir}/matrix.npz and {run_dir}/matrix.json")

    if config["TRACK"]:
        for i in range(num_tasks):
            for j in range(i + 1):
                wandb.log({f"crl/R/{i}_{j}": R[i, j], f"crl/retention/{i}_{j}": Retention[i, j]})
            wandb.log({f"crl/diag/{i}": R[i, i]})
        wandb.finish()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    run_continual(merged_config)


if __name__ == "__main__":
    main()
