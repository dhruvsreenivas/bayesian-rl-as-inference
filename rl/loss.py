"""VAPOR-lite loss functions."""

from typing import Callable, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import reverb
import rlax
import tree
from acme.agents.jax.actor_core import RecurrentState
from acme.jax import networks as networks_lib
from acme.jax import utils

from rl.types import ActorCriticOutput, Metrics

Observation = chex.Array
Action = int
IMPALAOutputs = Tuple[Tuple[networks_lib.Logits, networks_lib.Value], RecurrentState]

IMPALAPolicyValueFn = Callable[
    [networks_lib.Params, Observation, RecurrentState], IMPALAOutputs
]
FeedForwardPolicyValueFn = Callable[
    [networks_lib.Params, Observation], ActorCriticOutput
]
RewardPredictorFn = Callable[[networks_lib.Params, Observation, Action], chex.Array]
PriorPredictorFn = Callable[[networks_lib.Params, Observation, Action], chex.Array]


def reward_predictor_loss(
    forward_fn: RewardPredictorFn,
    prior_forward_fn: PriorPredictorFn,
    reward_std: float = 0.1,
) -> Callable[
    [hk.Params, hk.Params, reverb.ReplaySample, chex.PRNGKey],
    Tuple[chex.Array, Metrics],
]:
    """Reward predictor loss function."""

    def loss_fn(
        params: hk.Params,
        prior_params: hk.Params,
        sample: reverb.ReplaySample,
        key: chex.PRNGKey,
    ) -> chex.Array:
        """Batched reward prediction loss. Adds Gaussian noise to the reward targets."""

        data = sample.data
        observations, actions, rewards = data.observation, data.action, data.reward

        # now add Gaussian noise to the reward targets -> of shape []
        reward_noise = jax.random.normal(key, shape=rewards.shape)
        reward_noise *= reward_std
        rewards += reward_noise

        # now compute the reward predictions -> this is of shape [num_ensemble]
        predicted_rewards = forward_fn(params, observations, actions)

        # before we compute MSE, we add the prior function to each of the predicted rewards
        # shape [num_ensemble]
        prior_predicted_rewards = prior_forward_fn(prior_params, observations, actions)
        predicted_rewards += prior_predicted_rewards

        # now compute the MSE
        rewards = jnp.expand_dims(rewards, 0)  # [1]
        reward_errors = predicted_rewards - rewards  # [num_ensemble]
        loss = (reward_errors**2).mean()

        metrics = {"reward/loss": loss}
        return loss, metrics

    # TODO figure out how to handle mapreducing (should we vmap over keys?)
    return utils.mapreduce(loss_fn, in_axes=(None, None, 0, None))


def atari_vapor_lite_loss(
    unroll_fn: IMPALAPolicyValueFn,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
    td_lambda: float = 1.0,
) -> Callable[[hk.Params, reverb.ReplaySample], Tuple[chex.Array, Metrics]]:
    """Atari VAPOR-Lite loss."""

    def loss_fn(
        params: hk.Params, sample: reverb.ReplaySample
    ) -> Tuple[chex.Array, Metrics]:

        # Extract the data. Here, we assume that the rewards are not already augmented with the uncertainty bonus `sigma`.
        data = sample.data
        observations, actions, rewards, discounts, extra = (
            data.observation,
            data.action,
            data.reward,
            data.discount,
            data.extras,
        )
        initial_state = tree.map_structure(lambda s: s[0], extra["core_state"])
        behavior_logits = extra["logits"]
        sigmas = extra["sigmas"]

        # First add the uncertainty bonus to the rewards.
        chex.assert_equal_shape([rewards, sigmas])
        rewards += sigmas

        # Apply reward clipping.
        rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

        # Unroll current policy over sampled observations.
        (logits, values), _ = unroll_fn(params, observations, initial_state)

        # Compute importance sampling weights for V-trace (pi(a | s) / mu(a | s))
        rhos = rlax.categorical_importance_sampling_ratios(
            logits[:-1], behavior_logits[:-1], actions[:-1]
        )

        # Critic loss (V-Trace off-policy loss on the augmented rewards).
        vtrace_returns = rlax.vtrace_td_error_and_advantage(
            v_tm1=values[:-1],
            v_t=values[1:],
            r_t=rewards[:-1],
            discount_t=discounts[:-1] * discount,
            rho_tm1=rhos,
            lambda_=td_lambda,
        )
        critic_loss = jnp.square(vtrace_returns.errors)

        # Policy gradient loss (this is standard, as we're using augmented rewards already).
        policy_gradient_loss = rlax.policy_gradient_loss(
            logits_t=logits[:-1],
            a_t=actions[:-1],
            adv_t=vtrace_returns.pg_advantage,
            w_t=jnp.ones_like(rewards[:-1]),
        )

        # Uncertainty-weighted entropy regularization.
        entropy_loss = rlax.entropy_loss(logits[:-1], sigmas[:-1])

        # Combine weighted sum of actor & critic losses, averaged over the sequence.
        mean_loss = jnp.mean(
            policy_gradient_loss
            + baseline_cost * critic_loss
            + entropy_cost * entropy_loss
        )

        metrics = {
            "agent/policy_loss": jnp.mean(policy_gradient_loss),
            "agent/critic_loss": jnp.mean(baseline_cost * critic_loss),
            "agent/entropy_loss": jnp.mean(entropy_cost * entropy_loss),
            "agent/entropy": jnp.mean(entropy_loss),
            "agent/sigmas": jnp.mean(sigmas[:-1]),
        }

        return mean_loss, metrics

    return utils.mapreduce(loss_fn, in_axes=(None, 0))


def bsuite_vapor_lite_loss(
    network_fn: FeedForwardPolicyValueFn,
    *,
    discount: float,
    td_lambda: float,
    entropy_cost: float = 0.0,
) -> Callable[[hk.Params, reverb.ReplaySample], Tuple[chex.Array, Metrics]]:
    """Bsuite VAPOR-Lite loss function."""

    def loss_fn(
        params: hk.Params, sample: reverb.ReplaySample
    ) -> Tuple[chex.Array, Metrics]:

        # Extract the data.
        data = sample.data
        observations, actions, rewards, discounts, extras = (
            data.observation,
            data.action,
            data.reward,
            data.discount,
            data.extras,
        )
        sigmas = extras["sigmas"]

        # First add the uncertainty bonus to the rewards.
        chex.assert_equal_shape([rewards, sigmas])
        rewards += sigmas

        # Get actor-critic output.
        actor_critic_output = network_fn(params, observations)
        logits, values = actor_critic_output.logits, actor_critic_output.value

        # Critic loss.
        td_errors = rlax.td_lambda(
            v_tm1=values[:-1],
            r_t=rewards[:-1],
            discount_t=discounts[:-1] * discount,
            v_t=values[1:],
            lambda_=td_lambda,
        )
        critic_loss = jnp.square(td_errors)

        # Policy gradient loss.
        policy_gradient_loss = rlax.policy_gradient_loss(
            logits_t=logits[:-1],
            a_t=actions[:-1],
            adv_t=td_errors,
            w_t=jnp.ones_like(rewards[:-1]),
        )

        # Add weighted entropy term.
        entropy_loss = rlax.entropy_loss(logits[:-1], sigmas[:-1])

        # Combine weighted sum of actor & critic losses, averaged over the sequence.
        mean_loss = jnp.mean(
            policy_gradient_loss + critic_loss + entropy_cost * entropy_loss
        )

        metrics = {
            "agent/policy_loss": jnp.mean(policy_gradient_loss),
            "agent/critic_loss": jnp.mean(critic_loss),
            "agent/entropy_loss": jnp.mean(entropy_cost * entropy_loss),
            "agent/entropy": jnp.mean(entropy_loss),
            "agent/sigmas": jnp.mean(sigmas[:-1]),
        }

        return mean_loss, metrics

    return utils.mapreduce(loss_fn, in_axes=(None, 0))
