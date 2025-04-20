"""IMPALA networks definition."""

import dataclasses
import functools
from typing import Callable

import chex
import haiku as hk
import jax.numpy as jnp
from acme import specs, types
from acme.jax import networks as networks_lib

from rl.networks import AtariRewardPredictor, make_ensemble_network

IMPALANetworks = networks_lib.UnrollableNetwork


def bonus_fn(
    reward_predictions: networks_lib.NetworkOutput,
    sigma_scale: float = 0.01,
) -> chex.Array:
    """Bonus function."""

    # predictor output is of shape [num_ensemble, B]
    reward_std = jnp.std(reward_predictions, axis=0)

    bonuses = jnp.minimum(sigma_scale * reward_std, 1.0)
    return bonuses


@dataclasses.dataclass
class AtariVAPORLiteNetworks:
    """Networks for VAPOR-Lite for Atari (general IMPALA network along with the reward predictor ensemble)."""

    policy_network: IMPALANetworks
    reward_ensemble: networks_lib.FeedForwardNetwork
    prior_ensemble: networks_lib.FeedForwardNetwork
    uncertainty_bonus: Callable[[networks_lib.NetworkOutput], chex.Array]


def make_networks(
    env_spec: specs.EnvironmentSpec, num_ensemble: int = 10, sigma_scale: float = 0.01
) -> AtariVAPORLiteNetworks:
    """Builds networks for Atari VAPOR-Lite."""

    # First make IMPALA module.
    def make_impala_module() -> networks_lib.DeepIMPALAAtariNetwork:
        return networks_lib.DeepIMPALAAtariNetwork(env_spec.actions.num_values)

    policy_network = networks_lib.make_unrollable_network(env_spec, make_impala_module)

    # Now make reward predictor function and ensemble prior. These should both be the same architecture
    def reward_predictor_fn(
        observations: networks_lib.Observation, actions: networks_lib.Action
    ) -> networks_lib.NetworkOutput:
        return AtariRewardPredictor(use_layer_norm=True)(observations, actions)

    model = hk.without_apply_rng(hk.transform(reward_predictor_fn))
    reward_ensemble = make_ensemble_network(env_spec, model, num_ensemble=num_ensemble)
    prior_ensemble = make_ensemble_network(env_spec, model, num_ensemble=num_ensemble)

    # Now package them both together.
    return AtariVAPORLiteNetworks(
        policy_network=policy_network,
        reward_ensemble=reward_ensemble,
        prior_ensemble=prior_ensemble,
        uncertainty_bonus=functools.partial(bonus_fn, sigma_scale=sigma_scale),
    )


def compute_uncertainty_bonus(
    reward_ensemble_params: networks_lib.Params,
    transitions: types.Transition,
    networks: AtariVAPORLiteNetworks,
) -> chex.Array:
    """Computes uncertainty bonus from reward module."""

    reward_predictions = networks.reward_ensemble.apply(
        reward_ensemble_params, transitions.observation, transitions.action
    )
    return networks.uncertainty_bonus(reward_predictions)
