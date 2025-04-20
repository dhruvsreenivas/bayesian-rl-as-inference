"""Actor-critic networks definition."""

import dataclasses
import functools
from typing import Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs, types
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils

from rl.networks import RewardPredictor, make_ensemble_network
from rl.types import ActorCriticOutput


def bonus_fn(
    reward_predictions: networks_lib.NetworkOutput,
    sigma_scale: float = 3.0,
) -> chex.Array:
    """Bonus function."""

    # predictor output is of shape [num_ensemble, B]
    reward_std = jnp.std(reward_predictions, axis=0)

    bonuses = jnp.minimum(sigma_scale * reward_std, 1.0)
    return bonuses


@dataclasses.dataclass
class BsuiteVAPORLiteNetworks:
    """Networks for VAPOR-Lite for Bsuite (generic feedforward actor-critic along with the reward predictor ensemble)."""

    actor_critic: networks_lib.FeedForwardNetwork
    reward_ensemble: networks_lib.FeedForwardNetwork
    prior_ensemble: networks_lib.FeedForwardNetwork
    uncertainty_bonus: Callable[[networks_lib.NetworkOutput], chex.Array]


def apply_policy_and_sample(
    networks: BsuiteVAPORLiteNetworks, eval_mode: bool = False
) -> actor_core_lib.FeedForwardPolicy:
    """Returns a function that computes actions."""

    def apply_and_sample(params, key, obs):
        logits = networks.actor_critic.apply(params, obs).logits

        if eval_mode:
            action = jnp.argmax(logits)
        else:
            action = jax.random.categorical(key, logits).squeeze()

        return int(action)

    return apply_and_sample


def make_networks(
    env_spec: specs.EnvironmentSpec, num_ensemble: int = 10, sigma_scale: float = 3.0
) -> BsuiteVAPORLiteNetworks:
    """Builds networks for Bsuite VAPOR-Lite."""

    # First make feedforward module.
    def actor_critic_fn(inputs: chex.Array) -> ActorCriticOutput:
        """Standard Bsuite actor-critic function."""

        flat_inputs = hk.Flatten()(inputs)

        torso = hk.nets.MLP([64, 64])
        policy_head = hk.Linear(env_spec.actions.num_values)
        value_head = hk.Linear(1)
        embedding = torso(flat_inputs)
        logits = policy_head(embedding)
        value = value_head(embedding)

        return ActorCriticOutput(logits=logits, value=jnp.squeeze(value, axis=-1))

    actor_critic = hk.without_apply_rng(hk.transform(actor_critic_fn))

    # create dummy observations to use here.
    dummy_observations = utils.zeros_like(env_spec.observations)
    dummy_observations = utils.add_batch_dim(dummy_observations)

    actor_critic = networks_lib.FeedForwardNetwork(
        init=lambda key: actor_critic.init(key, dummy_observations),
        apply=actor_critic.apply,
    )

    # Now make reward ensemble function and ensemble prior. These should both be the same architecture.
    def reward_predictor_fn(
        observations: chex.Array, actions: chex.Array
    ) -> chex.Array:
        return RewardPredictor(hidden_sizes=[64, 64])(observations, actions)

    model = hk.without_apply_rng(hk.transform(reward_predictor_fn))
    reward_ensemble = make_ensemble_network(env_spec, model, num_ensemble=num_ensemble)
    prior_ensemble = make_ensemble_network(env_spec, model, num_ensemble=num_ensemble)

    return BsuiteVAPORLiteNetworks(
        actor_critic=actor_critic,
        reward_ensemble=reward_ensemble,
        prior_ensemble=prior_ensemble,
        uncertainty_bonus=functools.partial(bonus_fn, sigma_scale=sigma_scale),
    )


def compute_uncertainty_bonus(
    reward_ensemble_params: networks_lib.Params,
    transitions: types.Transition,
    networks: BsuiteVAPORLiteNetworks,
) -> chex.Array:
    """Computes uncertainty bonus from reward module."""

    reward_predictions = networks.reward_ensemble.apply(
        reward_ensemble_params, transitions.observation, transitions.action
    )
    return networks.uncertainty_bonus(reward_predictions)
