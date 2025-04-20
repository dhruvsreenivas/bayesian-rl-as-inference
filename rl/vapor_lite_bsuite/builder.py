from typing import Iterator, List, Optional

import acme
import jax
import optax
import reverb
from absl import logging
from acme import adders, core, specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, builders
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils
from acme.utils import counting, loggers

from rl.vapor_lite_bsuite import config as bsuite_vapor_lite_config
from rl.vapor_lite_bsuite import learning
from rl.vapor_lite_bsuite import networks as bsuite_vapor_lite_networks


class BsuiteVAPORLiteBuilder(
    builders.ActorLearnerBuilder[
        bsuite_vapor_lite_networks.BsuiteVAPORLiteNetworks,
        actor_core_lib.FeedForwardPolicyWithExtra,
        reverb.ReplaySample,
    ]
):
    """Bsuite VAPOR-lite builder. Built similarly to A2C in Bsuite repo, with addition of replay as is done in MPO (from Acme)."""

    def __init__(
        self,
        config: bsuite_vapor_lite_config.BsuiteVAPORLiteConfig,
    ):
        self._config = config

    def make_policy(
        self,
        networks: bsuite_vapor_lite_networks.BsuiteVAPORLiteNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.FeedForwardPolicy:
        """Makes the policy."""

        del environment_spec
        return bsuite_vapor_lite_networks.apply_policy_and_sample(
            networks, eval_mode=evaluation
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: actor_core_lib.FeedForwardPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> acme.Actor:
        """Makes actor."""

        del environment_spec
        assert variable_source is not None

        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
        variable_client = variable_utils.VariableClient(
            variable_source, "policy", device="cpu"
        )

        return actors.GenericActor(
            actor_core, random_key, variable_client, adder, backend="cpu"
        )

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: actor_core_lib.FeedForwardPolicy,
    ) -> List[reverb.Table]:
        """Create tables to insert data into. Here we handle the replay fraction."""

        del policy

        if self._config.samples_per_insert:
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=max(error_buffer, 2 * self._config.samples_per_insert),
            )
        else:
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        # Reverb loves Acme.
        replay_extensions = []
        queue_extensions = []

        # Create replay tables.
        tables = []
        if self._config.replay_fraction > 0:
            replay_table = reverb.Table(
                name=adders_reverb.DEFAULT_PRIORITY_TABLE,
                sampler=reverb.selectors.Prioritized(1.0),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_replay_size,
                rate_limiter=limiter,
                extensions=replay_extensions,
                signature=adders_reverb.NStepTransitionAdder.signature(
                    environment_spec
                ),
            )
            tables.append(replay_table)

            logging.info(
                "Creating off-policy replay buffer with replay fraction %g "
                "of batch %d",
                self._config.replay_fraction,
                self._config.batch_size,
            )

        if self._config.replay_fraction < 1:
            queue = reverb.Table.queue(
                name="queue_table",
                max_size=self._config.online_queue_capacity,
                extensions=queue_extensions,
                signature=adders_reverb.NStepTransitionAdder.signature(
                    environment_spec
                ),
            )
            tables.append(queue)

            logging.info(
                "Creating online replay queue with queue fraction %g " "of batch %d",
                1.0 - self._config.replay_fraction,
                self._config.batch_size,
            )

        return tables

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[actor_core_lib.FeedForwardPolicy],
    ) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the actor/environment."""

        del environment_spec, policy

        priority_fns = {}
        if self._config.replay_fraction > 0:
            priority_fns[adders_reverb.DEFAULT_PRIORITY_TABLE] = None
        if self._config.replay_fraction < 1:
            priority_fns["queue_table"] = None

        return adders_reverb.NStepTransitionAdder(
            priority_fns=priority_fns,
            client=replay_client,
            n_step=1,
            discount=self._config.discount,
        )

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:

        dataset = datasets.make_reverb_dataset(
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            table={
                adders_reverb.DEFAULT_PRIORITY_TABLE: self._config.replay_fraction,
                "queue_table": 1.0 - self._config.replay_fraction,
            },
            prefetch_size=4,
        )
        return utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: bsuite_vapor_lite_networks.BsuiteVAPORLiteNetworks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        """Builds the learner."""

        del environment_spec, replay_client

        # set up optimizers
        actor_critic_optimizer = optax.adam(self._config.learning_rate)
        reward_ensemble_optimizer = optax.adam(self._config.learning_rate)

        return learning.BsuiteVAPORLiteLearner(
            networks=networks,
            iterator=dataset,
            actor_critic_optimizer=actor_critic_optimizer,
            reward_ensemble_optimizer=reward_ensemble_optimizer,
            random_key=random_key,
            discount=self._config.discount,
            reward_std=self._config.reward_std,
            entropy_cost=self._config.entropy_cost,
            td_lambda=self._config.td_lambda,
            reward_counter=counter,
            rl_counter=counter,
            reward_logger=logger_fn("reward"),
            rl_logger=logger_fn("learner"),
        )
