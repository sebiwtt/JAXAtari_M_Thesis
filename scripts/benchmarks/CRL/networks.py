# =============================================================================
# Networks + parameter container for the JAXtari PPO trainer
# =============================================================================
# Shared TORSO (Network or MLP_Network, output dim 512) + two linear HEADS
# (Actor, Critic); heads are byte-for-byte interchangeable between modalities.
# `build_models` reconstructs the (stateless) modules from a config + action
# dim, so any consumer of saved AgentParams (trainer, eval, continual methods)
# can apply them without importing trainer internals.
# =============================================================================

from typing import NamedTuple, Sequence

import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class Network(nn.Module):
    """Pixel torso: Nature-CNN feature extractor."""
    @nn.compact
    def __call__(self, x):
        if x.ndim == 5:
            # (B, F, H, W, C) -> (B, H, W, F*C): each stacked frame's channels become conv input channels.
            b, f, h, w, c = x.shape
            x = jnp.transpose(x, (0, 2, 3, 1, 4)).reshape(b, h, w, f * c)
        else:
            x = jnp.transpose(x, (0, 2, 3, 1))  # (B, F, H, W) -> (B, H, W, F) for conv
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class MLP_Network(nn.Module):
    """Object-centric torso: 2-layer MLP producing the same 512-d output as Network."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            461,  # roughly matches the CNN's parameter count
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    """Torso features -> scalar state value V(s)."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Torso features -> action logits."""
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class AgentParams(NamedTuple):
    """Bundles the three param sets so a single `TrainState` can hold/update all of them."""
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


def action_dim_from_params(params: AgentParams) -> int:
    """Read the actor head's output dim back out of saved params."""
    return params.actor_params["params"]["Dense_0"]["bias"].shape[0]


def build_models(config: dict, action_dim: int) -> "tuple[nn.Module, Actor, Critic]":
    """(network, actor, critic) modules matching `config`'s modality - stateless,
    so they can `apply` any structurally compatible AgentParams."""
    network = Network() if config["PIXEL_BASED"] else MLP_Network()
    return network, Actor(action_dim=action_dim), Critic()
