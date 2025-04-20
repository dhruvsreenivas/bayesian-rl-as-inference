import functools
import time
from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple

import acme
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
from absl import logging
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting, loggers, reverb_utils

from rl.loss import atari_vapor_lite_loss, reward_predictor_loss
from rl.types import Metrics
from rl.vapor_lite_atari import networks as atari_vapor_lite_networks

_PMAP_AXIS_NAME = "data"


class RewardPredictorTrainingState(NamedTuple):
    """Reward predictor training state."""

    params: networks_lib.Params
    opt_state: optax.OptState
    prior_params: networks_lib.Params


class RLTrainingState(NamedTuple):
    """Impala training state."""

    params: networks_lib.Params
    opt_state: optax.OptState


class TrainingState(NamedTuple):
    """VAPOR-lite training state."""

    impala: RLTrainingState
    reward_ensemble: RewardPredictorTrainingState
    key: networks_lib.PRNGKey


class AtariVAPORLiteLearner(acme.Learner):
    """Learner for Atari VAPOR-lite."""

    _state: TrainingState

    def __init__(
        self,
        networks: atari_vapor_lite_networks.AtariVAPORLiteNetworks,
        iterator: Iterator[reverb.ReplaySample],
        impala_optimizer: optax.GradientTransformation,
        reward_ensemble_optimizer: optax.GradientTransformation,
        random_key: networks_lib.PRNGKey,
        is_sequence_based: bool = False,
        discount: float = 0.995,
        entropy_cost: float = 0.0,
        baseline_cost: float = 1.0,
        max_abs_reward: float = np.inf,
        reward_std: float = 0.1,
        td_lambda: float = 0.9,
        reward_counter: Optional[counting.Counter] = None,
        rl_counter: Optional[counting.Counter] = None,
        reward_logger: Optional[loggers.Logger] = None,
        rl_logger: Optional[loggers.Logger] = None,
        devices: Optional[Sequence[jax.Device]] = None,
    ):

        # this is for pmapping -- if we're not doing this, then don't worry about it
        local_devices = jax.local_devices()
        process_id = jax.process_index()
        logging.info("Learner process id: %s. Devices passed: %s", process_id, devices)
        logging.info(
            "Learner process id: %s. Local devices from JAX API: %s",
            process_id,
            local_devices,
        )
        self._devices = devices or local_devices
        self._local_devices = [d for d in self._devices if d in local_devices]

        # set up data iterators
        self._iterator = (self._process_sample(sample) for sample in iterator)
        self._is_sequence_based = is_sequence_based

        # set up reward bonus computation
        self._get_uncertainty_bonus = functools.partial(
            atari_vapor_lite_networks.compute_uncertainty_bonus, networks=networks
        )

        def unroll_without_rng(
            params: networks_lib.Params,
            observations: networks_lib.Observation,
            initial_state: networks_lib.RecurrentState,
        ) -> Tuple[networks_lib.NetworkOutput, networks_lib.RecurrentState]:

            unused_rng = jax.random.PRNGKey(0)
            return networks.policy_network.unroll(
                params, unused_rng, observations, initial_state
            )

        impala_loss_fn = atari_vapor_lite_loss(
            unroll_fn=unroll_without_rng,
            discount=discount,
            max_abs_reward=max_abs_reward,
            baseline_cost=baseline_cost,
            entropy_cost=entropy_cost,
            td_lambda=td_lambda,
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
            grads = jax.lax.pmean(grads, _PMAP_AXIS_NAME)

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
        self._reward_step = jax.pmap(
            reward_step, axis_name=_PMAP_AXIS_NAME, devices=self._devices
        )

        # Set up IMPALA training step.
        def impala_step(
            state: TrainingState, sample: reverb.ReplaySample
        ) -> Tuple[TrainingState, Metrics]:
            """IMPALA update step."""

            impala_state = state.impala

            # Compute gradients.
            grad_fn = jax.grad(impala_loss_fn, has_aux=True)
            grads, metrics = grad_fn(impala_state.params, sample)

            # TODO: if pmapping, pmean the grads here now.
            grads = jax.lax.pmean(grads, _PMAP_AXIS_NAME)

            # Apply updates.
            updates, new_opt_state = impala_optimizer.update(
                grads, impala_state.opt_state
            )
            new_params = optax.apply_updates(impala_state.params, updates)

            new_impala_state = RLTrainingState(
                params=new_params, opt_state=new_opt_state
            )
            new_state = state._replace(impala=new_impala_state)

            return new_state, metrics

        # TODO jit or pmap?
        self._rl_step = jax.pmap(
            impala_step, axis_name=_PMAP_AXIS_NAME, devices=self._devices
        )

        # Make initial state.
        def make_initial_state(key: networks_lib.PRNGKey) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""

            key, impala_key, reward_key, prior_key = jax.random.split(key, 4)

            # Reward ensemble training state.
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
            impala_params = networks.policy_network.init(impala_key)
            impala_opt_state = impala_optimizer.init(impala_params)
            impala = RLTrainingState(params=impala_params, opt_state=impala_opt_state)

            return TrainingState(
                impala=impala, reward_ensemble=reward_ensemble, key=key
            )

        state = make_initial_state(random_key)
        self._state = utils.replicate_in_all_devices(state, self._local_devices)

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
        """Uses the replay sample to train and update the reward predictor, as well as provide uncertainty estimates to the RL learner."""

        # Update reward on the sample first.
        self._state, metrics = self._reward_step(self._state, sample)

        transitions = reverb_utils.replay_sample_to_sars_transition(
            sample, is_sequence=self._is_sequence_based
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
        metrics = utils.get_from_first_device(metrics)

        # Update our counts and record them.
        counts = self._rl_counter.increment(steps=1, time_elapsed=time.time() - start)

        # Maybe write logs.
        self._rl_logger.write({**metrics, **counts})

    def get_variables(self, names: Sequence[str]) -> List[networks_lib.Params]:
        variables = {
            "reward_ensemble": utils.get_from_first_device(
                [self._state.reward_ensemble.params]
            ),
            "policy": utils.get_from_first_device([self._state.impala.params]),
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return utils.get_from_first_device(self._state)

    def restore(self, state: TrainingState):
        self._state = utils.replicate_in_all_devices(state, self._local_devices)
