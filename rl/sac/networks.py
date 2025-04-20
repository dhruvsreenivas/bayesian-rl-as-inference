import dataclasses
from typing import Optional, Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from acme import core, specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import types, utils


@dataclasses.dataclass
class SACNetworks:
    """Networks for Soft Actor-Critic."""

    actor: networks_lib.FeedForwardNetwork
    critic: networks_lib.FeedForwardNetwork
    log_prob: networks_lib.LogProbFn
    sample: networks_lib.SampleFn
    sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(
    networks: SACNetworks, eval_mode: bool = False
) -> actor_core_lib.FeedForwardPolicy:
    """Returns a function that computes actions."""
    sample_fn = networks.sample if not eval_mode else networks.sample_eval
    if not sample_fn:
        raise ValueError("sample function is not provided")

    def apply_and_sample(params, key, obs):
        return sample_fn(networks.actor.apply(params, obs), key)

    return apply_and_sample


def default_models_to_snapshot(networks: SACNetworks, spec: specs.EnvironmentSpec):
    """Defines default models to be snapshotted."""

    dummy_obs = utils.zeros_like(spec.observations)
    dummy_action = utils.zeros_like(spec.actions)
    dummy_key = jax.random.PRNGKey(0)

    def critic(source: core.VariableSource) -> types.ModelToSnapshot:
        params = source.get_variables(["critic"])[0]
        return types.ModelToSnapshot(
            networks.critic.apply, params, {"obs": dummy_obs, "action": dummy_action}
        )

    def default_training_actor(source: core.VariableSource) -> types.ModelToSnapshot:
        params = source.get_variables(["policy"])[0]
        return types.ModelToSnapshot(
            apply_policy_and_sample(networks, False),
            params,
            {"key": dummy_key, "obs": dummy_obs},
        )

    def default_eval_actor(source: core.VariableSource) -> types.ModelToSnapshot:
        params = source.get_variables(["policy"])[0]
        return types.ModelToSnapshot(
            apply_policy_and_sample(networks, True),
            params,
            {"key": dummy_key, "obs": dummy_obs},
        )

    return {
        "critic": critic,
        "default_training_actor": default_training_actor,
        "default_eval_actor": default_eval_actor,
    }


def make_networks(
    spec: specs.EnvironmentSpec, hidden_layer_sizes: Sequence[int] = (256, 256)
) -> SACNetworks:
    """Creates the networks."""

    num_dims = np.prod(spec.actions.shape, dtype=int)

    def _actor_fn(observation: chex.Array) -> chex.Array:
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    output_sizes=hidden_layer_sizes,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
                networks_lib.NormalTanhDistribution(num_dims),
            ]
        )
        return network(observation)

    def _critic_fn(observation: chex.Array, action: chex.Array) -> chex.Array:
        network1 = hk.nets.MLP(
            output_sizes=hidden_layer_sizes + (1,),
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
        )
        network2 = hk.nets.MLP(
            output_sizes=hidden_layer_sizes + (1,),
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
        )

        inputs = jnp.concatenate([observation, action], axis=-1)
        q1 = network1(inputs)
        q2 = network2(inputs)

        return jnp.concatenate([q1, q2], axis=-1)

    actor = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))

    # Set up dummy variables for init (you don't actually need to add batch dim directly)
    dummy_action = utils.zeros_like(spec.actions)
    dummy_obs = utils.zeros_like(spec.observations)

    return SACNetworks(
        actor=networks_lib.FeedForwardNetwork(
            lambda key: actor.init(key, dummy_obs), actor.apply
        ),
        critic=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
        log_prob=lambda params, actions: params.log_prob(actions),
        sample=lambda params, key: params.sample(seed=key),
        sample_eval=lambda params: params.mode(),
    )
