from typing import Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.networks.atari import DeepAtariTorso

from rl.types import ActivationFn


class RewardPredictor(hk.Module):
    """Single reward predictor, for use with non-pixel-based data."""

    def __init__(
        self, hidden_sizes: Sequence[int], activation: ActivationFn = jax.nn.relu
    ):

        super().__init__()
        self._hidden_sizes = hidden_sizes
        self._activation = activation

    def __call__(self, observations: chex.Array, actions: chex.Array) -> chex.Array:
        inputs = jnp.concatenate([observations, actions], axis=-1)

        outputs = hk.nets.MLP(
            output_sizes=self._hidden_sizes + (1,),
            activation=self._activation,
            activate_final=False,
        )(inputs)

        return jnp.squeeze(outputs, -1)


class AtariRewardPredictor(hk.Module):
    """Atari reward predictor."""

    def __init__(self, use_layer_norm: bool = True):
        super().__init__()

        self.use_layer_norm = use_layer_norm

    def __call__(self, observations: chex.Array, actions: chex.Array) -> chex.Array:
        obs_embs = DeepAtariTorso(use_layer_norm=self.use_layer_norm)(observations)

        obs_actions = jnp.concatenate([obs_embs, actions], axis=-1)
        outputs = hk.Linear(1)(obs_actions)

        return jnp.squeeze(outputs, -1)


def make_ensemble_network(
    env_spec: specs.EnvironmentSpec, model: hk.Transformed, num_ensemble: int = 10
) -> networks_lib.FeedForwardNetwork:
    """Makes an ensemble of networks."""

    dummy_observations = utils.zeros_like(env_spec.observations)
    dummy_actions = utils.zeros_like(env_spec.actions)

    def init(key: chex.PRNGKey) -> hk.Params:
        init_keys = jax.random.split(key, num_ensemble)
        batched_init = jax.vmap(model.init, in_axes=(0, None, None))

        params = batched_init(init_keys, dummy_observations, dummy_actions)
        return params

    def apply(
        params: hk.Params, observations: chex.Array, actions: chex.Array
    ) -> chex.Array:
        batched_apply = jax.vmap(model.apply, in_axes=(0, None, None))

        # size of outputs: [num_ensemble, B]
        outputs = batched_apply(params, observations, actions)
        return outputs

    return networks_lib.FeedForwardNetwork(init=init, apply=apply)
