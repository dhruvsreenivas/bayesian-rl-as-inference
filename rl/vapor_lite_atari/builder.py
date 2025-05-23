"""Atari VAPOR-lite Builder."""

from typing import Any, Callable, Generic, Iterator, List, Optional

import acme
import jax
import optax
import reverb
from absl import logging
from acme import adders, core, specs
from acme.adders import reverb as reverb_adders
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors as actors_lib
from acme.agents.jax import builders
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils
from acme.utils import counting, loggers

from rl.vapor_lite_atari import acting
from rl.vapor_lite_atari import config as atari_vapor_lite_config
from rl.vapor_lite_atari import learning
from rl.vapor_lite_atari import networks as atari_vapor_lite_networks


class AtariVAPORLiteBuilder(
    Generic[actor_core_lib.RecurrentState],
    builders.ActorLearnerBuilder[
        atari_vapor_lite_networks.AtariVAPORLiteNetworks,
        acting.AtariVAPORLitePolicy,
        reverb.ReplaySample,
    ],
):
    """Atari VAPOR-lite Builder. Built off of IMPALA Builder in Acme."""

    def __init__(
        self,
        config: atari_vapor_lite_config.AtariVAPORLiteConfig,
        table_extension: Optional[Callable[[], Any]] = None,
    ):
        """Creates a VAPOR-lite Atari learner."""
        self._config = config
        self._sequence_length = self._config.sequence_length
        self._table_extension = table_extension

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: acting.AtariVAPORLitePolicy,
    ) -> List[reverb.Table]:
        """The queue; use XData or INFO log."""
        dummy_actor_state = policy.init(jax.random.PRNGKey(0))
        signature = reverb_adders.SequenceAdder.signature(
            environment_spec,
            policy.get_extras(dummy_actor_state),
            sequence_length=self._config.sequence_length,
        )

        # Maybe create rate limiter.
        # Setting the samples_per_insert ratio less than the default of 1.0, allows
        # the agent to drop data for the benefit of using data from most up-to-date
        # policies to compute its learner updates.
        samples_per_insert = self._config.samples_per_insert
        if samples_per_insert:
            if samples_per_insert > 1.0 or samples_per_insert <= 0.0:
                raise ValueError(
                    "Atari VAPOR-lite requires a samples_per_insert ratio in the range (0, 1],"
                    f" but received {samples_per_insert}."
                )
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                samples_per_insert=samples_per_insert,
                min_size_to_sample=1,
                error_buffer=self._config.batch_size,
            )
        else:
            limiter = reverb.rate_limiters.MinSize(1)

        # Make two queues for online and offline replay fraction.
        tables = []

        replay_extensions = []
        queue_extensions = []
        if self._table_extension is not None:
            queue_extensions = [self._table_extension()]

        if self._config.replay_fraction > 0:
            replay = reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Prioritized(1.0),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_queue_size,
                max_times_sampled=1,
                rate_limiter=limiter,
                extensions=replay_extensions,
                signature=signature,
            )
            tables.append(replay)

            logging.info(
                "Creating off-policy replay buffer with replay fraction %g "
                "of batch %d",
                self._config.replay_fraction,
                self._config.batch_size,
            )
        if self._config.replay_fraction < 1:
            queue = reverb.Table.queue(
                name="queue_table",
                max_size=self._config.max_queue_size,
                extensions=queue_extensions,
                signature=signature,
            )
            tables.append(queue)

            logging.info(
                "Creating online replay queue with queue fraction %g " "of batch %d",
                1.0 - self._config.replay_fraction,
                self._config.batch_size,
            )

        return tables

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:
        """Creates a dataset."""
        batch_size_per_learner = self._config.batch_size // jax.process_count()
        batch_size_per_device, ragged = divmod(
            self._config.batch_size, jax.device_count()
        )
        if ragged:
            raise ValueError(
                "Learner batch size must be divisible by total number of devices!"
            )

        dataset = datasets.make_reverb_dataset(
            table={
                self._config.replay_table_name: self._config.replay_fraction,
                "queue_table": 1.0 - self._config.replay_fraction,
            },
            server_address=replay_client.server_address,
            batch_size=batch_size_per_device,
            num_parallel_calls=None,
            max_in_flight_samples_per_worker=2 * batch_size_per_learner,
        )

        return utils.multi_device_put(dataset.as_numpy_iterator(), jax.local_devices())

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[acting.AtariVAPORLitePolicy],
    ) -> Optional[adders.Adder]:
        """Creates an adder which handles observations."""
        del environment_spec, policy
        # Note that the last transition in the sequence is used for bootstrapping
        # only and is ignored otherwise. So we need to make sure that sequences
        # overlap on one transition, thus "-1" in the period length computation.
        return reverb_adders.SequenceAdder(
            client=replay_client,
            priority_fns={self._config.replay_table_name: None},
            period=self._config.sequence_period or (self._sequence_length - 1),
            sequence_length=self._sequence_length,
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: atari_vapor_lite_networks.AtariVAPORLiteNetworks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        del environment_spec, replay_client

        impala_optimizer = optax.chain(
            optax.clip_by_global_norm(self._config.max_gradient_norm),
            optax.adam(
                self._config.learning_rate,
                b1=self._config.adam_momentum_decay,
                b2=self._config.adam_variance_decay,
                eps=self._config.adam_eps,
                eps_root=self._config.adam_eps_root,
            ),
        )
        reward_ensemble_optimizer = optax.chain(
            optax.clip_by_global_norm(self._config.max_gradient_norm),
            optax.adam(
                self._config.learning_rate,
                b1=self._config.adam_momentum_decay,
                b2=self._config.adam_variance_decay,
                eps=self._config.adam_eps,
                eps_root=self._config.adam_eps_root,
            ),
        )

        return learning.AtariVAPORLiteLearner(
            networks=networks,
            iterator=dataset,
            impala_optimizer=impala_optimizer,
            reward_ensemble_optimizer=reward_ensemble_optimizer,
            random_key=random_key,
            discount=self._config.discount,
            entropy_cost=self._config.entropy_cost,
            baseline_cost=self._config.baseline_cost,
            max_abs_reward=self._config.max_abs_reward,
            td_lambda=self._config.td_lambda,
            reward_std=self._config.reward_std,
            reward_counter=counter,
            rl_counter=counter,
            reward_logger=logger_fn("reward"),
            rl_logger=logger_fn("learner"),
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: acting.AtariVAPORLitePolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> acme.Actor:
        del environment_spec
        variable_client = variable_utils.VariableClient(
            client=variable_source,
            key="network",
            update_period=self._config.variable_update_period,
        )
        return actors_lib.GenericActor(policy, random_key, variable_client, adder)

    def make_policy(
        self,
        networks: atari_vapor_lite_networks.AtariVAPORLiteNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> acting.AtariVAPORLitePolicy:
        return acting.get_actor_core(networks, environment_spec, evaluation)
