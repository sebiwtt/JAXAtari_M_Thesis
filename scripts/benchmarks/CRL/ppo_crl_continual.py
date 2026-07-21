# =============================================================================
# Continual-RL evaluation harness for the JAXtari PPO trainer
# =============================================================================
# Trains ONE agent sequentially over ordered single-mod Pong tasks, carrying
# params forward. CL_METHOD selects the continual-learning method; methods live
# in continual/ (one file each) behind the continual.base.CLMethod interface,
# so this loop is method-agnostic:
#
#   ft      - naive finetuning (no mitigation)
#   ewc     - Elastic Weight Consolidation      (continual/ewc.py)
#   agem    - Averaged Gradient Episodic Memory (continual/agem.py)
#   packnet - PackNet iterative pruning         (continual/packnet.py)
#
# After each task, evaluates on every task seen so far:
#
#   R[i, j]         = return of the agent trained through task i, evaluated on task j  (j <= i)
#   R_rand[j]       = return of a fresh/untrained agent on task j - the "knows nothing" floor,
#                     not 0 (Pong's random-policy floor is close to -21)
#   Retention[i, j] = (R[i, j] - R_rand[j]) / (R[j, j] - R_rand[j])                     (j <= i)
#                     1.0 = matches post-task-j performance, 0.0 = performs like random.
#
# EVAL_FULL_MATRIX also fills j > i: forward transfer to not-yet-trained tasks.
#
# Orchestration only; PPO lives in `ppo_trainer.train`, evaluation in
# `ppo_eval.evaluate`, models in `networks.py`, envs in `envs.py`.
# =============================================================================

import json
import os
import time
from functools import partial

import flax
import hydra
import jax
import numpy as np
import wandb
from omegaconf import OmegaConf

from continual import make_cl_method
from envs import make_env
from networks import Actor, AgentParams, Critic, MLP_Network, Network
from ppo_eval import evaluate
from ppo_trainer import train


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


def _init_random_agent_params(config: dict, key: jax.random.PRNGKey) -> AgentParams:
    """Freshly-initialized, untrained params - the R_rand floor for retention.

    Mirrors `train()`'s fresh-init branch, standalone, since `train()` always runs at
    least one PPO iteration (RTPT requires max_iterations > 0).
    """
    env = make_env(
        config["ENV_ID"], config["SEED"], 1, [], config["PIXEL_BASED"], config["NATIVE_DOWNSCALING"], config["SMOOTH_IMAGE"], config["GRAYSCALE"]
    )()
    network = Network() if config["PIXEL_BASED"] else MLP_Network()
    actor = Actor(action_dim=env.action_space().n)
    critic = Critic()

    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, obs_key1, obs_key2, obs_key3 = jax.random.split(key, 4)
    network_params = network.init(network_key, env.observation_space().sample(obs_key1).squeeze()[None, ...])
    return AgentParams(
        network_params=network_params,
        actor_params=actor.init(actor_key, network.apply(network_params, np.array([env.observation_space().sample(obs_key2).squeeze()]))),
        critic_params=critic.init(critic_key, network.apply(network_params, np.array([env.observation_space().sample(obs_key3).squeeze()]))),
    )


def _save_checkpoint(path: str, config: dict, params: AgentParams) -> None:
    """Serialization format `ppo_eval.evaluate` expects."""
    with open(path, "wb") as f:
        f.write(
            flax.serialization.to_bytes(
                [config, [params.network_params, params.actor_params, params.critic_params]]
            )
        )


def run_continual(config: dict) -> None:
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    task_mods_list = [list(m) for m in config["TASK_MODS"]]
    assert len(task_mods_list) > 0, "TASK_MODS must contain at least one task"
    assert len(task_mods_list[0]) == 0, "TASK_MODS[0] must be the base task (no mods)"
    for i, mods in enumerate(task_mods_list):
        assert len(mods) <= 1, f"CRL tasks must use at most one mod each; TASK_MODS[{i}]={mods} has {len(mods)}"
    num_tasks = len(task_mods_list)
    labels = [_task_label(m) for m in task_mods_list]

    # Validates CL_METHOD and its config keys before any training starts.
    method = make_cl_method(config, num_tasks)

    # Mirrors train()'s own derivation, needed here for the wandb step offset.
    batch_size = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    num_iterations = int(config["TOTAL_TIMESTEPS"] // batch_size)

    group_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}'
    base_run_name = f'{group_name}_{config["SEED"]}'
    run_dir = f"runs/{base_run_name}"
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

    Model = (Network, Actor, Critic) if config["PIXEL_BASED"] else (MLP_Network, Actor, Critic)

    # Random-agent floor R_rand[j]: one eval pass per task, no training. Keyed off
    # EVAL_SEED (not SEED) so it's independent of the training seed.
    rand_params = _init_random_agent_params(config, jax.random.PRNGKey(config["EVAL_SEED"]))
    rand_ckpt_path = f"{run_dir}/random_agent.cleanrl_model"
    _save_checkpoint(rand_ckpt_path, config, rand_params)
    print(f"[CRL] random-agent baseline checkpoint saved to {rand_ckpt_path}")

    R_rand = np.full(num_tasks, np.nan)
    eval_time_rand = 0.0
    for j in range(num_tasks):
        eval_mods = task_mods_list[j]
        eval_t0 = time.perf_counter()
        episodic_returns, _, completed = evaluate(
            model_path=rand_ckpt_path,
            make_env=partial(
                make_env,
                mods=eval_mods,
                pixel_based=config["PIXEL_BASED"],
                native_downscaling=config["NATIVE_DOWNSCALING"],
                smooth_image=config["SMOOTH_IMAGE"],
                grayscale=config["GRAYSCALE"],
                eval=True,
            ),
            env_id=config["ENV_ID"],
            eval_episodes=config["EVAL_EPISODES"],
            Model=Model,
            seed=config["EVAL_SEED"],
        )
        episodic_returns = np.asarray(jax.device_get(episodic_returns))
        completed = np.asarray(jax.device_get(completed))
        eval_time_rand += time.perf_counter() - eval_t0
        n_completed = int(completed.sum())
        if n_completed < completed.shape[0]:
            print(
                f"[CRL] WARNING: R_rand[{j}] only {n_completed}/{completed.shape[0]} eval episodes "
                f"completed within the eval scan window; this floor value may be inflated."
            )
        R_rand[j] = float(episodic_returns.mean())
        print(f"[CRL] R_rand[{j}] (random agent on task {j}={labels[j]!r}) = {R_rand[j]:.3f}")

    # CL method state: built once from param shapes, carried through the task loop.
    cl_state = method.init_state(rand_params)
    eval_tmp_path = f"{run_dir}/cl_eval_tmp.cleanrl_model"

    R = np.full((num_tasks, num_tasks), np.nan)
    ckpt_paths: list[str] = []
    carried_params = None
    train_time_per_task = np.full(num_tasks, np.nan)
    eval_time_matrix = 0.0

    for i in range(num_tasks):
        task_mods = task_mods_list[i]
        task_config = dict(config)
        task_config["TRAIN_MODS"] = tuple(task_mods)

        task_run_name = f"{base_run_name}_task{i}"
        print(f"\n=== CRL task {i}/{num_tasks - 1}: mods={task_mods} (label={labels[i]!r}, cl_method={method.name}) ===")
        train_t0 = time.perf_counter()
        carried_params, cl_state = method.train_task(
            train,
            task_config,
            carried_params,
            cl_state,
            i,
            run_name=task_run_name,
            wandb_step_offset=i * num_iterations,
            manage_wandb=False,
            wandb_group=labels[i],
        )
        jax.block_until_ready(carried_params)
        train_time_per_task[i] = time.perf_counter() - train_t0

        ckpt_path = f"{run_dir}/task_{i}.cleanrl_model"
        _save_checkpoint(ckpt_path, task_config, carried_params)
        print(f"[CRL] task {i} checkpoint saved to {ckpt_path}")
        ckpt_paths.append(ckpt_path)

        # j <= i: retention (tasks already trained on). EVAL_FULL_MATRIX also fills
        # j > i: forward transfer to tasks not yet trained on. Methods may substitute
        # task-specific eval params (e.g. PackNet subnetworks) via eval_params.
        eval_js = range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)
        for j in eval_js:
            eval_mods = task_mods_list[j]
            eval_params = method.eval_params(carried_params, cl_state, j, i)
            if eval_params is carried_params:
                eval_ckpt_path = ckpt_path
            else:
                eval_ckpt_path = eval_tmp_path
                _save_checkpoint(eval_ckpt_path, task_config, eval_params)
            eval_t0 = time.perf_counter()
            episodic_returns, _, completed = evaluate(
                model_path=eval_ckpt_path,
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
                Model=Model,
                seed=config["EVAL_SEED"],
            )
            episodic_returns = np.asarray(jax.device_get(episodic_returns))
            completed = np.asarray(jax.device_get(completed))
            eval_time_matrix += time.perf_counter() - eval_t0
            n_completed = int(completed.sum())
            if n_completed < completed.shape[0]:
                print(
                    f"[CRL] WARNING: R[{i},{j}] only {n_completed}/{completed.shape[0]} eval episodes "
                    f"completed within the eval scan window; this cell's mean return may be inflated."
                )
            R[i, j] = float(episodic_returns.mean())
            kind = "forward transfer" if j > i else "retention"
            print(f"[CRL] R[{i},{j}] ({kind}: train through task {i}={labels[i]!r}, eval on task {j}={labels[j]!r}) = {R[i, j]:.3f}")

    method.save_artifacts(cl_state, run_dir)
    if os.path.exists(eval_tmp_path):
        os.remove(eval_tmp_path)

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

    # Wall-clock, GPU-inclusive: train_task/evaluate are synchronous Python calls
    # that already force a device sync (train()'s per-iteration .item() calls;
    # jax.block_until_ready/device_get here) before returning, so timing the call
    # site captures real compute time without needing to thread timing through
    # train()/continual method internals.
    total_train_time = float(np.nansum(train_time_per_task))
    total_eval_time = float(eval_time_rand + eval_time_matrix)
    total_compute_time = total_train_time + total_eval_time
    print(
        f"\n[CRL] compute time: train={total_train_time:.1f}s, "
        f"eval={total_eval_time:.1f}s (rand={eval_time_rand:.1f}s, matrix={eval_time_matrix:.1f}s), "
        f"total={total_compute_time:.1f}s ({total_compute_time / 3600:.2f} h)"
    )

    np.savez(
        f"{run_dir}/matrix.npz",
        R=R,
        R_rand=R_rand,
        Retention=Retention,
        task_mods=np.array([json.dumps(m) for m in task_mods_list]),
        labels=np.array(labels),
        env_id=np.array(config["ENV_ID"]),
        exp_name=np.array(config["EXP_NAME"]),
        train_time_per_task=train_time_per_task,
        eval_time_rand=np.array(eval_time_rand),
        eval_time_matrix=np.array(eval_time_matrix),
        total_compute_time=np.array(total_compute_time),
    )
    with open(f"{run_dir}/matrix.json", "w") as f:
        json.dump(
            {
                "env_id": config["ENV_ID"],
                "exp_name": config["EXP_NAME"],
                "cl_method": method.name,
                "task_mods": task_mods_list,
                "labels": labels,
                "R": R.tolist(),
                "R_rand": R_rand.tolist(),
                "Retention": Retention.tolist(),
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
        wandb.log({
            "crl/compute_time/train_total_sec": total_train_time,
            "crl/compute_time/eval_total_sec": total_eval_time,
            "crl/compute_time/total_sec": total_compute_time,
        })
        wandb.finish()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    run_continual(config)


if __name__ == "__main__":
    main()
