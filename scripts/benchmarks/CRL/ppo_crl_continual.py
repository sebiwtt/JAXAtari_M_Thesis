# =============================================================================
# Continual-RL evaluation harness for the JAXtari PPO trainer
# =============================================================================
# Trains ONE agent sequentially over ordered single-mod Pong tasks, carrying params
# forward. CL_METHOD selects the continual-learning method (see crl_methods.py):
#   "ft"  (default) - naive finetuning, no forgetting mitigation
#   "ewc"           - Elastic Weight Consolidation (Kirkpatrick et al. 2017): after each
#                     task a diagonal Fisher is estimated and folded into an EWCState
#                     (EWC_MODE last/multi/online); subsequent tasks add the quadratic
#                     penalty 0.5*EWC_COEF*mean_i F_i (theta_i - theta*_i)^2 to the PPO loss.
#   "agem"          - Averaged Gradient Episodic Memory (Chaudhry et al. 2019): after each
#                     task, AGEM_MEMORY_PER_TASK transitions from a final-policy rollout
#                     are appended to an episodic memory; subsequent tasks project every
#                     minibatch gradient that conflicts with the memory's BC gradient.
#   "packnet"       - PackNet (Mallya & Lazebnik 2018): each task's budget is split into
#                     train -> prune -> finetune (PACKNET_FINETUNE_FRAC). Pruning gives
#                     each task an equal share of the network's free weights; earlier
#                     tasks' weights are frozen via gradient masks, and R[i,j] (j <= i)
#                     is evaluated on task j's recovered subnetwork.
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
# Orchestration only; PPO lives in `ppo_trainer.train`, evaluation in `ppo_eval.evaluate`.
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

from crl_methods import (
    agem_extend_memory,
    ewc_update_state,
    packnet_eval_params,
    packnet_finetune_mask,
    packnet_init_owner,
    packnet_ownership_summary,
    packnet_prune,
    packnet_train_mask,
)
from ppo_eval import evaluate
from ppo_trainer import AgentParams, Actor, Critic, MLP_Network, Network, make_env, train


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


def run_continual(config: dict) -> None:
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    cl_method = str(config.get("CL_METHOD", "ft")).lower()
    assert cl_method in ("ft", "ewc", "agem", "packnet"), (
        f"unknown CL_METHOD {cl_method!r} (supported: 'ft', 'ewc', 'agem', 'packnet')"
    )
    if cl_method == "ewc":
        assert float(config.get("EWC_COEF", 0.0)) > 0.0, "CL_METHOD=ewc requires EWC_COEF > 0"

    task_mods_list = [list(m) for m in config["TASK_MODS"]]
    assert len(task_mods_list) > 0, "TASK_MODS must contain at least one task"
    assert len(task_mods_list[0]) == 0, "TASK_MODS[0] must be the base task (no mods)"
    for i, mods in enumerate(task_mods_list):
        assert len(mods) <= 1, f"CRL tasks must use at most one mod each; TASK_MODS[{i}]={mods} has {len(mods)}"
    num_tasks = len(task_mods_list)
    labels = [_task_label(m) for m in task_mods_list]

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
    with open(rand_ckpt_path, "wb") as f:
        f.write(
            flax.serialization.to_bytes(
                [config, [rand_params.network_params, rand_params.actor_params, rand_params.critic_params]]
            )
        )
    print(f"[CRL] random-agent baseline checkpoint saved to {rand_ckpt_path}")

    R_rand = np.full(num_tasks, np.nan)
    for j in range(num_tasks):
        eval_mods = task_mods_list[j]
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
        n_completed = int(completed.sum())
        if n_completed < completed.shape[0]:
            print(
                f"[CRL] WARNING: R_rand[{j}] only {n_completed}/{completed.shape[0]} eval episodes "
                f"completed within the eval scan window; this floor value may be inflated."
            )
        R_rand[j] = float(episodic_returns.mean())
        print(f"[CRL] R_rand[{j}] (random agent on task {j}={labels[j]!r}) = {R_rand[j]:.3f}")

    R = np.full((num_tasks, num_tasks), np.nan)
    ckpt_paths: list[str] = []
    carried_params = None
    ewc_state = None
    agem_memory = None

    pn_owner = None
    if cl_method == "packnet":
        # Owner tree needs param shapes only; rand_params has the same architecture.
        pn_owner = packnet_init_owner(rand_params)
        pn_ft_frac = float(config.get("PACKNET_FINETUNE_FRAC", 0.2))
        assert 0.0 < pn_ft_frac < 1.0, "PACKNET_FINETUNE_FRAC must be in (0, 1)"
        pn_ft_iters = max(1, round(num_iterations * pn_ft_frac))
        pn_train_iters = num_iterations - pn_ft_iters
        assert pn_train_iters >= 1, "PACKNET_FINETUNE_FRAC leaves no iterations for the train phase"
        print(f"[CRL] PackNet per-task split: {pn_train_iters} train + {pn_ft_iters} finetune iterations")

    for i in range(num_tasks):
        task_mods = task_mods_list[i]
        task_config = dict(config)
        task_config["TRAIN_MODS"] = tuple(task_mods)

        task_run_name = f"{base_run_name}_task{i}"
        print(f"\n=== CRL task {i}/{num_tasks - 1}: mods={task_mods} (label={labels[i]!r}, cl_method={cl_method}) ===")
        # Post-task CL state (Fisher / memory block) of the last task would never be
        # consumed by a later task, so skip collecting it.
        want_fisher = cl_method == "ewc" and i < num_tasks - 1
        want_agem_samples = cl_method == "agem" and i < num_tasks - 1
        if cl_method == "packnet":
            # Phase 1/2: train on the free capacity (+ this task's weights + shared critic).
            train_cfg = dict(task_config)
            train_cfg["TOTAL_TIMESTEPS"] = pn_train_iters * batch_size
            carried_params = train(
                train_cfg,
                init_params=carried_params,
                run_name=task_run_name,
                manage_wandb=False,
                wandb_step_offset=i * num_iterations,
                wandb_group=labels[i],
                grad_mask=packnet_train_mask(pn_owner, i),
            )
            # Prune: task i claims its equal share of free weights, the rest are zeroed.
            carried_params, pn_owner = packnet_prune(carried_params, pn_owner, i, num_tasks)
            print(f"[CRL] PackNet pruned after task {i}; ownership fractions: {packnet_ownership_summary(pn_owner)}")
            # Phase 2/2: finetune only task i's weights to recover from the pruning.
            ft_cfg = dict(task_config)
            ft_cfg["TOTAL_TIMESTEPS"] = pn_ft_iters * batch_size
            ft_cfg["LEARNING_RATE"] = float(config.get("PACKNET_FINETUNE_LR", config["LEARNING_RATE"]))
            carried_params = train(
                ft_cfg,
                init_params=carried_params,
                run_name=f"{task_run_name}_ft",
                manage_wandb=False,
                wandb_step_offset=i * num_iterations + pn_train_iters,
                wandb_group=labels[i],
                grad_mask=packnet_finetune_mask(pn_owner, i),
            )
            result = carried_params
        else:
            result = train(
                task_config,
                init_params=carried_params,
                run_name=task_run_name,
                manage_wandb=False,
                wandb_step_offset=i * num_iterations,
                wandb_group=labels[i],
                ewc_state=ewc_state,
                return_fisher=want_fisher,
                agem_memory=agem_memory,
                return_agem_samples=want_agem_samples,
            )
        if want_fisher:
            carried_params, new_fisher = result
            ewc_state = ewc_update_state(
                ewc_state,
                carried_params,
                new_fisher,
                mode=config.get("EWC_MODE", "online"),
                decay=config.get("EWC_DECAY", 0.9),
            )
            print(f'[CRL] EWC state updated after task {i} (mode={config.get("EWC_MODE", "online")})')
        elif want_agem_samples:
            carried_params, new_block = result
            agem_memory = agem_extend_memory(agem_memory, new_block)
            print(f"[CRL] A-GEM memory extended after task {i} (total {agem_memory.actions.shape[0]} transitions)")
        else:
            carried_params = result

        # Same serialization format as evaluate() expects, so it can load this unmodified.
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

        # j <= i: retention (tasks already trained on). EVAL_FULL_MATRIX also fills
        # j > i: forward transfer to tasks not yet trained on.
        eval_js = range(num_tasks) if config.get("EVAL_FULL_MATRIX", False) else range(i + 1)
        for j in eval_js:
            eval_mods = task_mods_list[j]
            eval_ckpt_path = ckpt_path
            if cl_method == "packnet" and j <= i:
                # Original-PackNet eval: task j runs on its recovered subnetwork (weights
                # owned by tasks <= j; later tasks' weights and free capacity zeroed).
                # j > i (forward transfer) keeps the full current params instead.
                masked = packnet_eval_params(carried_params, pn_owner, j)
                eval_ckpt_path = f"{run_dir}/packnet_eval_tmp.cleanrl_model"
                with open(eval_ckpt_path, "wb") as f:
                    f.write(
                        flax.serialization.to_bytes(
                            [task_config, [masked.network_params, masked.actor_params, masked.critic_params]]
                        )
                    )
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
            n_completed = int(completed.sum())
            if n_completed < completed.shape[0]:
                print(
                    f"[CRL] WARNING: R[{i},{j}] only {n_completed}/{completed.shape[0]} eval episodes "
                    f"completed within the eval scan window; this cell's mean return may be inflated."
                )
            R[i, j] = float(episodic_returns.mean())
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

    if cl_method == "packnet":
        # Owner tree is needed to recover per-task subnetworks from the (full) task_i
        # checkpoints later; without it the masked eval params are not reproducible.
        owner_path = f"{run_dir}/packnet_owner.msgpack"
        with open(owner_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [pn_owner.network_params, pn_owner.actor_params, pn_owner.critic_params]
                )
            )
        print(f"[CRL] PackNet owner tree saved to {owner_path}")
        tmp_path = f"{run_dir}/packnet_eval_tmp.cleanrl_model"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

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
        env_id=np.array(config["ENV_ID"]),
        exp_name=np.array(config["EXP_NAME"]),
    )
    with open(f"{run_dir}/matrix.json", "w") as f:
        json.dump(
            {
                "env_id": config["ENV_ID"],
                "exp_name": config["EXP_NAME"],
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


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    run_continual(merged_config)


if __name__ == "__main__":
    main()
