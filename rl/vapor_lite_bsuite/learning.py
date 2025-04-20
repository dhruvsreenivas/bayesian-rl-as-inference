import functools
import time
from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple

import acme
import jax
import jax.numpy as jnp
import optax
import reverb
from acme.jax import networks as networks_lib
from acme.utils import counting, loggers, reverb_utils

from rl.loss import bsuite_vapor_lite_loss, reward_predictor_loss
from rl.types import Metrics
from rl.vapor_lite_bsuite import networks as bsuite_vapor_lite_networks


class RewardPredictorTrainingState(NamedTuple):
    """Reward predictor training state."""

    params: networks_lib.Params
    opt_state: optax.OptState
    prior_params: networks_lib.Params


class RLTrainingState(NamedTuple):
    """Actor-critic training state."""

    params: networks_lib.Params
    opt_state: optax.OptState


class TrainingState(NamedTuple):
    """VAPOR-lite Bsuite training state."""

    actor_critic: RLTrainingState
    reward_ensemble: RewardPredictorTrainingState
    key: networks_lib.PRNGKey


class BsuiteVAPORLiteLearner(acme.Learner):
    """VAPOR-lite for Bsuite."""

    _state: TrainingState

    def __init__(
        self,
        networks: bsuite_vapor_lite_networks.BsuiteVAPORLiteNetworks,
        iterator: Iterator[reverb.ReplaySample],
        actor_critic_optimizer: optax.GradientTransformation,
        reward_ensemble_optimizer: optax.GradientTransformation,
        random_key: networks_lib.PRNGKey,
        discount: float = 0.99,
        reward_std: float = 3.0,
        td_lambda: float = 0.9,
        entropy_cost: float = 0.0,
        reward_counter: Optional[counting.Counter] = None,
        rl_counter: Optional[counting.Counter] = None,
        reward_logger: Optional[loggers.Logger] = None,
        rl_logger: Optional[loggers.Logger] = None,
    ):

        # set up data iterator -- you do reward model updating inside the iterator process sample step
        self._iterator = (self._process_sample(sample) for sample in iterator)

        # set up reward bonus computation
        self._get_uncertainty_bonus = functools.partial(
            bsuite_vapor_lite_networks.compute_uncertainty_bonus, networks=networks
        )

        actor_critic_loss_fn = bsuite_vapor_lite_loss(
            network_fn=networks.actor_critic.apply,
            discount=discount,
            td_lambda=td_lambda,
            entropy_cost=entropy_cost,
        )
        reward_loss_fn = reward_predictor_loss(
            forward_fn=networks.reward_ensemble.apply,
            prior_forward_fn=networks.prior_ensemble.apply,
            reward_std=reward_std,
        )

        # Set up reward update step.
        def reward_step(
            state: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Metrics]:
            """Reward update step."""

            reward_state = state.reward_ensemble

            # Set up PRNGKeys for mapreduce training.
            batch_size = sample.data.observation.shape[0]
            key, *update_keys = jax.random.split(state.key, batch_size + 1)
            update_keys = jnp.array(update_keys)

            # Compute reward gradients.
            grad_fn = jax.grad(reward_loss_fn, has_aux=True)
            grads, metrics = grad_fn(
                reward_state.params, reward_state.prior_params, sample, update_keys
            )

            # TODO: if pmapping, pmean the grads here now.

            # Apply updates.
            updates, new_opt_state = reward_ensemble_optimizer.update(
                grads, reward_state.opt_state
            )
            new_params = optax.apply_updates(reward_state.params, updates)

            new_reward_state = RewardPredictorTrainingState(
                params=new_params,
                opt_state=new_opt_state,
                prior_params=reward_state.prior_params,
            )
            new_state = state._replace(reward_ensemble=new_reward_state, key=key)

            return new_state, metrics

        # TODO: jit or pmap?
        self._reward_step = jax.jit(reward_step)

        # Set up RL training step.
        def rl_step(
            state: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Metrics]:
            """RL update step."""

            actor_critic_state = state.actor_critic

            # Compute gradients.
            grad_fn = jax.grad(actor_critic_loss_fn, has_aux=True)
            grads, metrics = grad_fn(actor_critic_state.params, sample)

            # TODO: if pmapping, pmean the grads here now.

            # Apply updates.
            updates, new_opt_state = actor_critic_optimizer.update(
                grads, actor_critic_state.opt_state
            )
            new_params = optax.apply_updates(actor_critic_state.params, updates)

            new_actor_critic_state = RLTrainingState(
                params=new_params, opt_state=new_opt_state
            )
            new_state = state._replace(actor_critic=new_actor_critic_state)

            return new_state, metrics

        # TODO jit or pmap?
        self._rl_step = jax.jit(rl_step)

        # Make initial state.
        def make_initial_state(key: networks_lib.PRNGKey) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""

            key, rl_key, reward_key, prior_key = jax.random.split(key, 4)

            # Reward ensemble training state
            reward_ensemble_params = networks.reward_ensemble.init(reward_key)
            reward_prior_params = networks.prior_ensemble.init(prior_key)
            reward_ensemble_opt_state = reward_ensemble_optimizer.init(
                reward_ensemble_params
            )
            reward_ensemble = RewardPredictorTrainingState(
                params=reward_ensemble_params,
                opt_state=reward_ensemble_opt_state,
                prior_params=reward_prior_params,
            )

            # RL training state.
            actor_critic_params = networks.actor_critic.init(rl_key)
            actor_critic_opt_state = actor_critic_optimizer.init(actor_critic_params)
            impala = RLTrainingState(
                params=actor_critic_params, opt_state=actor_critic_opt_state
            )

            return TrainingState(
                impala=impala, reward_ensemble=reward_ensemble, key=key
            )

        self._state = make_initial_state(random_key)

        # Set up logging/counting.
        self._reward_counter = reward_counter or counting.Counter()
        self._rl_counter = rl_counter or counting.Counter()

        self._reward_logger = reward_logger or loggers.make_default_logger(
            "reward", steps_key=self._reward_counter.get_steps_key()
        )
        self._rl_logger = rl_logger or loggers.make_default_logger(
            "learner", steps_key=self._rl_counter.get_steps_key()
        )

    def _process_sample(self, sample: reverb.ReplaySample) -> reverb.ReplaySample:
        """Uses the replay sample to train and update the reward predictor."""

        # Update reward on the sample first.
        self._state, metrics = self._reward_step(self._state, sample)

        # Now update the bonuses for RL training.
        transitions = reverb_utils.replay_sample_to_sars_transition(
            sample, is_sequence=False
        )
        bonuses = self._get_uncertainty_bonus(
            self._state.reward_ensemble.params, transitions
        )

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Increment counts and record the current time
        counts = self._reward_counter.increment(steps=1, walltime=elapsed_time)

        # Attempts to write the logs.
        self._reward_logger.write({**metrics, **counts})

        # Now add the uncertainty bonuses to the sample and return it.
        return sample._replace(
            data=sample.data._replace(
                extras=sample.data.extras.update({"sigmas": bonuses})
            )
        )

    def step(self):
        """Does the RL training step."""

        # Get data from our updated iterator (reward training is done there already so don't worry about it)
        samples = next(self._iterator)

        # Do SGD on the batch.
        start = time.time()
        self._state, metrics = self._rl_step(self._state, samples)

        # TODO if pmapping, get metrics from first (host) device with `utils.get_from_first_device`.

        # Update our counts and record them.
        counts = self._rl_counter.increment(steps=1, time_elapsed=time.time() - start)

        # Maybe write logs.
        self._rl_logger.write({**metrics, **counts})

    def get_variables(self, names: Sequence[str]) -> List[networks_lib.Params]:
        variables = {
            "reward_ensemble": self._state.reward_ensemble.params,
            "actor_critic": self._state.actor_critic.params,
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
