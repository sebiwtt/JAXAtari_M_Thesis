# =============================================================================
# Task-difficulty ranking harness for the JAXtari PPO trainer
# =============================================================================
# Answers: "starting from an agent trained on the base task, how much extra
# training does each single-mod task need to recover base-level performance?"
#
#   1. Train ONE agent on the base task (TASK_MODS[0] == []) for TOTAL_TIMESTEPS.
#   2. Evaluate it on the base task -> target return R_base (true reward/episode
#      boundaries via eval=True; NOT the clipped, episodic-life training return).
#   3. For each other task j, resume from the *same* base checkpoint and finetune,
#      evaluating on task j every EVAL_EVERY_ITERS iterations. Record the global_step
#      at which the eval return first reaches the target; early-stop there (unless
#      TRAIN_FULL_BUDGET). Each task branches independently from base (not chained).
#   4. Rank tasks by steps-to-target: fewer steps = easier to adapt to.
#
# The target is a fixed return level (R_base), so "steps to reach it" is directly
# comparable across tasks. A task whose ceiling sits below the target simply never
# crosses and is reported as "not reached" (inf), which is itself a difficulty signal.
#
# Orchestration only; PPO lives in `ppo_trainer.train` (via its iteration_callback
# hook), evaluation in `ppo_eval.evaluate`.
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
from ppo_trainer import AgentParams, Actor, Critic, MLP_Network, Network, make_env, train


def _task_label(mods) -> str:
    return "base" if len(mods) == 0 else str(mods[0])


def _save_ckpt(path: str, config: dict, params: AgentParams) -> None:
    """Serialize params in the exact [config, [net, actor, critic]] layout evaluate() loads."""
    with open(path, "wb") as f:
        f.write(
            flax.serialization.to_bytes(
                [config, [params.network_params, params.actor_params, params.critic_params]]
            )
        )


def _eval_return(config: dict, Model, ckpt_path: str, eval_mods: list) -> tuple[float, int, int]:
    """Mean eval return of the checkpoint on `eval_mods`, plus (completed, total) episode counts."""
    episodic_returns, _, completed = evaluate(
        model_path=ckpt_path,
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
    return float(episodic_returns.mean()), int(completed.sum()), int(completed.shape[0])


def run_difficulty(config: dict) -> None:
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    task_mods_list = [list(m) for m in config["TASK_MODS"]]
    assert len(task_mods_list) > 1, "TASK_MODS must contain the base task plus >=1 adaptation task"
    assert len(task_mods_list[0]) == 0, "TASK_MODS[0] must be the base task (no mods)"
    for i, mods in enumerate(task_mods_list):
        assert len(mods) <= 1, f"CRL tasks must use at most one mod each; TASK_MODS[{i}]={mods} has {len(mods)}"
    labels = [_task_label(m) for m in task_mods_list]

    # Difficulty-specific knobs, all optional with sensible defaults.
    adapt_timesteps = int(config.get("ADAPT_TIMESTEPS", config["TOTAL_TIMESTEPS"]))
    eval_every_iters = int(config.get("EVAL_EVERY_ITERS", 1))
    # Absolute return the adapting agent must reach; margin lets you accept "close enough"
    # (e.g. base return minus 0.5) so a slightly noisier ceiling still counts as recovered.
    threshold_margin = float(config.get("THRESHOLD_MARGIN", 0.0))
    train_full_budget = bool(config.get("TRAIN_FULL_BUDGET", False))

    batch_size = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    base_iterations = int(config["TOTAL_TIMESTEPS"] // batch_size)

    group_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_difficulty'
    base_run_name = f'{group_name}_{config["SEED"]}'
    run_dir = f"runs/{base_run_name}"
    os.makedirs(run_dir, exist_ok=True)

    if config["TRACK"]:
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=base_run_name,
            group=group_name,
            save_code=True,
        )

    Model = (Network, Actor, Critic) if config["PIXEL_BASED"] else (MLP_Network, Actor, Critic)

    # ------------------------------------------------------------------ #
    # 1. Train the base agent (full TOTAL_TIMESTEPS budget) on the base task.
    # ------------------------------------------------------------------ #
    base_config = dict(config)
    base_config["TRAIN_MODS"] = tuple(task_mods_list[0])
    print(f"\n=== base training: mods={task_mods_list[0]} (label={labels[0]!r}), {config['TOTAL_TIMESTEPS']:,} steps ===")
    base_params = train(
        base_config,
        init_params=None,
        run_name=f"{base_run_name}_base",
        manage_wandb=False,
        wandb_step_offset=0,
        wandb_group="base",
    )
    base_ckpt = f"{run_dir}/base.cleanrl_model"
    _save_ckpt(base_ckpt, base_config, base_params)
    print(f"[DIFF] base checkpoint saved to {base_ckpt}")

    # ------------------------------------------------------------------ #
    # 2. Target return: base agent evaluated on the base task.
    # ------------------------------------------------------------------ #
    r_base, n_done, n_total = _eval_return(config, Model, base_ckpt, task_mods_list[0])
    if n_done < n_total:
        print(f"[DIFF] WARNING: base eval only {n_done}/{n_total} episodes completed; R_base may be inflated.")
    target_return = r_base - threshold_margin
    print(f"[DIFF] R_base (base agent on base task) = {r_base:.3f}  ->  target_return = {target_return:.3f}")

    # ------------------------------------------------------------------ #
    # 3. Adapt from the base checkpoint to every other task, one at a time,
    #    each branching independently from `base_params`.
    # ------------------------------------------------------------------ #
    adapt_iterations = int(adapt_timesteps // batch_size)
    results: list[dict] = []

    for i in range(1, len(task_mods_list)):
        task_mods = task_mods_list[i]
        label = labels[i]
        task_config = dict(config)
        task_config["TRAIN_MODS"] = tuple(task_mods)
        task_config["TOTAL_TIMESTEPS"] = adapt_timesteps  # drives NUM_ITERATIONS + LR anneal inside train()

        probe_ckpt = f"{run_dir}/adapt_{label}.cleanrl_model"
        # Mutable state closed over by the callback (avoids nonlocal gymnastics).
        state = {"crossing_step": None, "best_return": -np.inf, "best_step": 0, "curve": []}

        def probe(iteration, global_step, params, _cfg=task_config, _mods=task_mods,
                  _ckpt=probe_ckpt, _label=label, _st=state, _last=adapt_iterations):
            is_probe = (iteration % eval_every_iters == 0) or (iteration == _last)
            if not is_probe:
                return False
            _save_ckpt(_ckpt, _cfg, params)
            mean_ret, done, total = _eval_return(config, Model, _ckpt, _mods)
            _st["curve"].append((int(global_step), mean_ret))
            if mean_ret > _st["best_return"]:
                _st["best_return"], _st["best_step"] = mean_ret, int(global_step)
            reached = mean_ret >= target_return
            tag = " >= target" if reached else ""
            print(f"[DIFF] adapt {_label}: step={global_step:,} eval_return={mean_ret:.3f} "
                  f"(target={target_return:.3f}, {done}/{total} done){tag}")
            if config["TRACK"]:
                wandb.log({f"difficulty/adapt_return/{_label}": mean_ret, f"difficulty/adapt_step/{_label}": global_step})
            if reached and _st["crossing_step"] is None:
                _st["crossing_step"] = int(global_step)
                if not train_full_budget:
                    return True  # early-stop: we have our steps-to-target
            return False

        print(f"\n=== adapt to task {i}/{len(task_mods_list) - 1}: mods={task_mods} (label={label!r}), "
              f"up to {adapt_timesteps:,} steps ===")
        train(
            task_config,
            init_params=base_params,  # always branch from the base checkpoint, not the previous task
            run_name=f"{base_run_name}_adapt_{label}",
            manage_wandb=False,
            wandb_step_offset=base_iterations + (i - 1) * adapt_iterations,
            wandb_group=f"adapt_{label}",
            iteration_callback=probe,
        )

        reached = state["crossing_step"] is not None
        steps = state["crossing_step"] if reached else float("inf")
        results.append({
            "label": label,
            "mods": task_mods,
            "reached_target": reached,
            "steps_to_target": steps,
            "best_return": state["best_return"],
            "best_step": state["best_step"],
            "curve": state["curve"],
        })
        status = f"{steps:,} steps" if reached else f"NOT reached (best {state['best_return']:.3f} @ {state['best_step']:,})"
        print(f"[DIFF] task {label!r}: steps-to-target = {status}")

    # ------------------------------------------------------------------ #
    # 4. Rank by steps-to-target (unreached tasks sort last, hardest).
    # ------------------------------------------------------------------ #
    ranked = sorted(results, key=lambda r: (not r["reached_target"], r["steps_to_target"], -r["best_return"]))

    print("\n" + "=" * 64)
    print(f"TASK DIFFICULTY RANKING  (target return = {target_return:.3f}, R_base = {r_base:.3f})")
    print("  easiest = fewest steps to recover base-level performance after adapting")
    print("=" * 64)
    print(f"{'rank':>4}  {'task':<16}{'steps_to_target':>18}{'best_return':>14}")
    for rank, r in enumerate(ranked, 1):
        steps_str = f"{r['steps_to_target']:,}" if r["reached_target"] else "not reached"
        print(f"{rank:>4}  {r['label']:<16}{steps_str:>18}{r['best_return']:>14.3f}")

    out = {
        "env_id": config["ENV_ID"],
        "exp_name": config["EXP_NAME"],
        "base_task": task_mods_list[0],
        "r_base": r_base,
        "target_return": target_return,
        "threshold_margin": threshold_margin,
        "adapt_timesteps": adapt_timesteps,
        "eval_every_iters": eval_every_iters,
        "train_full_budget": train_full_budget,
        "base_checkpoint": base_ckpt,
        "ranking": [
            {
                "rank": rank,
                "label": r["label"],
                "mods": r["mods"],
                "reached_target": r["reached_target"],
                "steps_to_target": (r["steps_to_target"] if r["reached_target"] else None),
                "best_return": r["best_return"],
                "best_step": r["best_step"],
                "curve": r["curve"],
            }
            for rank, r in enumerate(ranked, 1)
        ],
    }
    out_path = f"{run_dir}/difficulty.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[DIFF] difficulty ranking saved to {out_path}")

    if config["TRACK"]:
        wandb.log({"difficulty/r_base": r_base, "difficulty/target_return": target_return})
        for rank, r in enumerate(ranked, 1):
            steps = r["steps_to_target"] if r["reached_target"] else float("nan")
            wandb.log({f"difficulty/rank/{r['label']}": rank, f"difficulty/steps_to_target/{r['label']}": steps})
        try:
            table = wandb.Table(columns=["rank", "task", "steps_to_target", "reached", "best_return"])
            for rank, r in enumerate(ranked, 1):
                table.add_data(rank, r["label"],
                               r["steps_to_target"] if r["reached_target"] else None,
                               r["reached_target"], r["best_return"])
            wandb.log({"difficulty/ranking_table": table})
        except Exception as e:  # table logging is best-effort; don't fail the run over it
            print(f"[DIFF] WARNING: could not log wandb ranking table: {e}")
        wandb.finish()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    run_difficulty(merged_config)


if __name__ == "__main__":
    main()
