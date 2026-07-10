# =============================================================================
# Common interface for continual-RL methods (one file per method, MEAL-style)
# =============================================================================
# The orchestrator (ppo_crl_continual.py) drives every method identically:
#
#     method   = make_cl_method(config, num_tasks)          # continual/__init__.py
#     cl_state = method.init_state(init_params)
#     for i, task in enumerate(tasks):
#         params, cl_state = method.train_task(train, task_config, params, cl_state, i, ...)
#         R[i, j] <- evaluate(method.eval_params(params, cl_state, j, i))   for each j
#     method.save_artifacts(cl_state, run_dir)
#
# `ppo_trainer.train` stays CL-agnostic: it calls exactly two jit-safe hooks,
# `loss_penalty` (added to the PPO loss) and `transform_grads` (on the raw
# grads before Adam). The method object itself is a trace-time constant;
# `cl_state` - the method's device data (Fisher, memory, masks, ...) - is
# threaded through as a jit *argument*, so large buffers are never baked into
# the compiled executable as constants.
#
# Method-to-hook map:
#   ft      - no hooks (train() compiles identically to plain PPO)
#   ewc     - loss_penalty;  update_state estimates a Fisher from the rollout
#   agem    - transform_grads (projection);  update_state extends the memory
#   packnet - transform_grads (phase mask);  overrides train_task entirely
#             (train -> prune -> finetune) and eval_params (subnetworks)
# =============================================================================

import jax.numpy as jnp


class CLMethod:
    """Base class = the naive-finetuning behavior; methods override what they need."""

    name = "base"
    #: Call the loss_penalty/transform_grads hooks inside train(). False keeps the
    #: compiled update byte-identical to plain PPO (used by ft).
    uses_train_hooks = True
    #: Ask train() for a GAE-completed rollout of the final task policy, fed to
    #: `update_state` to build the next task's cl_state (EWC's Fisher, A-GEM's memory).
    needs_final_rollout = False

    def __init__(self, config: dict, num_tasks: int):
        self.config = config
        self.num_tasks = num_tasks

    # ---- lifecycle (plain Python, runs between train() calls) ---------------

    def init_state(self, params):
        """Build the initial cl_state from (freshly initialized) params; also the
        place to construct model-dependent helpers (params carry the action dim)."""
        return None

    def update_state(self, cl_state, params, storage, key):
        """Fold one finished task into cl_state, given the task's final params and
        a final-policy rollout (only called when `needs_final_rollout`)."""
        return cl_state

    def train_task(
        self,
        train_fn,
        task_config: dict,
        init_params,
        cl_state,
        task_idx: int,
        run_name: str,
        wandb_step_offset: int,
        **train_kwargs,
    ):
        """Run one task; returns (params, new cl_state). Default: a single train()
        call, collecting the final rollout for update_state when the method needs
        it - skipped on the last task, where nothing would consume it."""
        collect = self.needs_final_rollout and task_idx < self.num_tasks - 1
        result = train_fn(
            task_config,
            init_params=init_params,
            run_name=run_name,
            wandb_step_offset=wandb_step_offset,
            cl_method=self if self.uses_train_hooks else None,
            cl_state=cl_state,
            return_final_rollout=collect,
            **train_kwargs,
        )
        if collect:
            params, (storage, key) = result
            cl_state = self.update_state(cl_state, params, storage, key)
        else:
            params = result
        return params, cl_state

    # ---- jit-safe hooks, called inside ppo_trainer.train ---------------------

    def loss_penalty(self, params, cl_state) -> jnp.ndarray:
        """Scalar added to the PPO loss (traced; `cl_state is None` may be branched
        on at trace time)."""
        return jnp.array(0.0)

    def transform_grads(self, grads, params, cl_state, key):
        """Rewrite the minibatch grads before Adam; returns (grads, metrics) where
        metrics is a {name: scalar} dict logged per iteration under losses/."""
        return grads, {}

    # ---- evaluation / persistence --------------------------------------------

    def eval_params(self, params, cl_state, eval_task: int, trained_task: int):
        """Params used to fill R[trained_task, eval_task]; default: unchanged."""
        return params

    def save_artifacts(self, cl_state, run_dir: str) -> None:
        """Persist whatever is needed to reproduce per-task eval params later."""
