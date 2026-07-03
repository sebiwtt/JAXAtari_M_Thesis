# =============================================================================
# PPO (naive finetuning) plugged into the continual-RL benchmark harness.
#
# The generic protocol lives in `crl_benchmark.run_continual` and the shared,
# harness-owned evaluation in `crl_eval.evaluate_policy`; the PPO loop lives in
# `ppo_trainer.train`. This file adapts PPO to the `CRLAlgorithm` interface and
# provides the hydra entry point. To benchmark another algorithm (e.g. PQN),
# write these same four adapter functions for it and call `run_continual` with
# its bundle.
# =============================================================================

import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from crl_benchmark import CRLAlgorithm, run_continual
from crl_env import make_env
from ppo_trainer import AgentParams, Actor, Critic, MLP_Network, Network, train


def _build_networks(config: dict):
    """Instantiate the torso/heads and a template param pytree (shapes only)."""
    env = make_env(
        config["ENV_ID"], config["SEED"], 1, [], config["PIXEL_BASED"], config["NATIVE_DOWNSCALING"], config["SMOOTH_IMAGE"]
    )()
    network = Network() if config["PIXEL_BASED"] else MLP_Network()
    actor = Actor(action_dim=env.action_space().n)
    critic = Critic()
    return env, network, actor, critic


def _init_params(config: dict, seed: int) -> AgentParams:
    """Freshly-initialized, untrained params (the R_rand floor). Mirrors `train()`'s
    fresh-init branch, standalone, since `train()` always runs at least one iteration."""
    env, network, actor, critic = _build_networks(config)
    key = jax.random.PRNGKey(seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, obs_key1, obs_key2, obs_key3 = jax.random.split(key, 4)
    network_params = network.init(network_key, env.observation_space().sample(obs_key1).squeeze()[None, ...])
    return AgentParams(
        network_params=network_params,
        actor_params=actor.init(actor_key, network.apply(network_params, np.array([env.observation_space().sample(obs_key2).squeeze()]))),
        critic_params=critic.init(critic_key, network.apply(network_params, np.array([env.observation_space().sample(obs_key3).squeeze()]))),
    )


def _save_checkpoint(path: str, config: dict, params: AgentParams) -> None:
    with open(path, "wb") as f:
        f.write(
            flax.serialization.to_bytes(
                [config, [params.network_params, params.actor_params, params.critic_params]]
            )
        )


def _load_policy(model_path: str, config: dict):
    """Rebuild the PPO networks, load checkpoint params, and return
    act(obs, key) -> (action, key) for the harness-owned evaluation."""
    # Template params define the pytree structure/shapes for from_bytes; the values
    # are irrelevant and fully overwritten by the checkpoint.
    template = _init_params(config, seed=0)
    _, network, actor, _ = _build_networks(config)
    with open(model_path, "rb") as f:
        (_, (network_params, actor_params, _)) = flax.serialization.from_bytes(
            (None, (template.network_params, template.actor_params, template.critic_params)), f.read()
        )

    def act(obs, key):
        hidden = network.apply(network_params, obs)
        logits = actor.apply(actor_params, hidden)
        # Gumbel-max trick for categorical sampling.
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    return act


PPO_FINETUNE = CRLAlgorithm(
    name="ppo_naive_finetune",
    init_params=_init_params,
    train=train,
    save_checkpoint=_save_checkpoint,
    load_policy=_load_policy,
)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    run_continual(merged_config, PPO_FINETUNE)


if __name__ == "__main__":
    main()
