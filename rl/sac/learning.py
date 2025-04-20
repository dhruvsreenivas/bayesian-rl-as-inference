import time
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple

import acme
import chex
import jax
import jax.numpy as jnp
import optax
import reverb
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting, loggers

from rl.sac import networks as sac_networks
from rl.types import Metrics


class TrainingState(NamedTuple):

    actor_params: networks_lib.Params
    critic_params: networks_lib.Params
    target_critic_params: networks_lib.Params

    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    # key for random sampling
    key: networks_lib.PRNGKey

    # tunable temperature
    alpha_params: Optional[networks_lib.Params] = None
    alpha_opt_state: Optional[optax.OptState] = None


class SACLearner(acme.Learner):
    """SAC learner."""

    _state: TrainingState

    def __init__(
        self,
        networks: sac_networks.SACNetworks,
        rng: networks_lib.PRNGKey,
        iterator: Iterator[reverb.ReplaySample],
        actor_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        discount: float = 0.99,
        entropy_coefficient: Optional[float] = None,
        target_entropy: float = 0.0,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        num_sgd_steps_per_step: int = 1,
    ):

        # set up whether we're tuning the temperature parameter in SAC or not
        adaptive_entropy_coefficient = entropy_coefficient is None
        if adaptive_entropy_coefficient:
            log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
            alpha_optimizer = optax.adam(learning_rate=3e-4)
            alpha_opt_state = alpha_optimizer.init(log_alpha)
        else:
            # make sure the target entropy is not defined, because we don't need it
            if target_entropy:
                raise ValueError(
                    "`target_entropy` should not be set when we're not tuning temperature parameter."
                )

        # Losses.
        def alpha_loss(
            log_alpha: chex.Array,
            actor_params: networks_lib.Params,
            transitions: types.Transition,
            key: networks_lib.PRNGKey,
        ) -> Tuple[chex.Array, Metrics]:
            dist_params = networks.actor.apply(actor_params, transitions.observation)
            action = networks.sample(dist_params, key)
            log_prob = networks.log_prob(dist_params, action)

            alpha = jnp.exp(log_alpha)
            alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)

            loss = jnp.mean(alpha_loss)
            return loss, {"alpha_loss": loss}

        def critic_loss(
            critic_params: networks_lib.Params,
            actor_params: networks_lib.Params,
            target_critic_params: networks_lib.Params,
            alpha: chex.Array,
            transitions: types.Transition,
            key: networks_lib.PRNGKey,
        ) -> Tuple[chex.Array, Metrics]:

            # size [B, 2]
            q = networks.critic.apply(
                critic_params, transitions.observation, transitions.action
            )

            # now get target Q values
            next_dist_params = networks.actor.apply(
                actor_params, transitions.next_observation
            )
            next_action = networks.sample(next_dist_params, key)
            next_log_prob = networks.log_prob(next_dist_params, next_action)
            next_q = networks.critic.apply(
                target_critic_params, transitions.next_observation, next_action
            )
            next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
            target_q = jax.lax.stop_gradient(
                transitions.reward * reward_scale
                + transitions.discount * discount * next_v
            )

            td_errors = q - jnp.expand_dims(target_q, axis=-1)
            critic_loss = 0.5 * (td_errors**2).mean()

            return critic_loss, {"critic_loss": critic_loss, "q": q.mean()}

        def actor_loss(
            actor_params: networks_lib.Params,
            critic_params: networks_lib.Params,
            alpha: chex.Array,
            transitions: types.Transition,
            key: networks_lib.PRNGKey,
        ) -> Tuple[chex.Array, Metrics]:

            dist_params = networks.actor.apply(actor_params, transitions.observation)
            action = networks.sample(dist_params, key)
            log_prob = networks.log_prob(dist_params, action)

            q = networks.critic.apply(critic_params, transitions.observation, action)
            min_q = jnp.min(q, axis=-1)

            actor_loss = alpha * log_prob - min_q
            loss = jnp.mean(actor_loss)

            return loss, {"actor_loss": loss}

        def update_step(
            state: TrainingState, transitions: types.Transition
        ) -> Tuple[TrainingState, Metrics]:

            # set up update keys
            key, alpha_key, critic_key, actor_key = jax.random.split(state.key, 4)

            # Compute all gradients.
            if adaptive_entropy_coefficient:
                # get the gradients for temperature first
                alpha_grads, alpha_metrics = jax.grad(alpha_loss, has_aux=True)(
                    state.alpha_params, state.actor_params, transitions, alpha_key
                )
                alpha = jnp.exp(state.alpha_params)
            else:
                alpha = entropy_coefficient

            critic_grads, critic_metrics = jax.grad(critic_loss, has_aux=True)(
                state.critic_params,
                state.actor_params,
                state.target_critic_params,
                alpha,
                transitions,
                critic_key,
            )
            actor_grads, actor_metrics = jax.grad(actor_loss, has_aux=True)(
                state.actor_params, state.critic_params, alpha, transitions, actor_key
            )

            # Now apply all gradients.
            actor_updates, actor_opt_state = actor_optimizer.update(
                actor_grads, state.actor_opt_state
            )
            actor_params = optax.apply_updates(state.actor_params, actor_updates)

            critic_updates, critic_opt_state = critic_optimizer.update(
                critic_grads, state.critic_opt_state
            )
            critic_params = optax.apply_updates(state.critic_params, critic_updates)

            target_critic_params = optax.incremental_update(
                critic_params, state.target_critic_params, tau
            )

            metrics = {**actor_metrics, **critic_metrics}

            # set up new training state
            new_state = TrainingState(
                actor_params=actor_params,
                critic_params=critic_params,
                target_critic_params=target_critic_params,
                actor_opt_state=actor_opt_state,
                critic_opt_state=critic_opt_state,
                key=key,
            )
            if adaptive_entropy_coefficient:
                # update alpha params
                alpha_updates, alpha_opt_state = alpha_optimizer.update(
                    alpha_grads, state.alpha_opt_state
                )
                alpha_params = optax.apply_updates(state.alpha_params, alpha_updates)

                alpha_metrics.update({"alpha": jnp.exp(alpha_params)})
                metrics.update(alpha_metrics)

                new_state = new_state._replace(
                    alpha_params=alpha_params, alpha_opt_state=alpha_opt_state
                )

            metrics["reward_mean"] = jnp.mean(transitions.reward, axis=0)
            metrics["reward_std"] = jnp.std(transitions.reward, axis=0)

            return new_state, metrics

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            "learner",
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
            steps_key=self._counter.get_steps_key(),
        )

        # Iterator on demonstration transitions.
        self._iterator = iterator

        # set up update step
        update_step = utils.process_multiple_batches(
            update_step, num_sgd_steps_per_step
        )
        self._update_step = jax.jit(update_step)

        # set up initial state.
        def make_initial_state(key: networks_lib.PRNGKey) -> TrainingState:
            """Sets up the SAC training state."""

            actor_key, critic_key, key = jax.random.split(key, 3)

            actor_params = networks.actor.init(actor_key)
            critic_params = networks.critic.init(critic_key)

            actor_opt_state = actor_optimizer.init(actor_params)
            critic_opt_state = critic_optimizer.init(critic_params)

            state = TrainingState(
                actor_params=actor_params,
                critic_params=critic_params,
                target_critic_params=critic_params,
                actor_opt_state=actor_opt_state,
                critic_opt_state=critic_opt_state,
                key=key,
            )
            if adaptive_entropy_coefficient:
                state = state._replace(
                    alpha_params=log_alpha, alpha_opt_state=alpha_opt_state
                )

            return state

        self._state = make_initial_state(rng)

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    def step(self):
        # sample from data stream.
        sample = next(self._iterator)
        transitions = types.Transition(*sample.data)

        # update.
        self._state, metrics = self._update_step(self._state, transitions)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Increment counts and record the current time
        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        # Attempts to write the logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names: List[Any]) -> List[Any]:
        variables = {
            "actor": self._state.actor_params,
            "critic": self._state.critic_params,
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
