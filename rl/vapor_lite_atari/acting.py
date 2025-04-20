# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IMPALA actor implementation."""

from typing import Generic, Mapping, Tuple

import chex
import jax
import jax.numpy as jnp
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types

from rl.vapor_lite_atari import networks as atari_vapor_lite_networks

AtariVAPORLiteExtras = Mapping[str, jnp.ndarray]


@chex.dataclass(frozen=True, mappable_dataclass=False)
class AtariVAPORLiteActorState(Generic[actor_core_lib.RecurrentState]):
    rng: jax_types.PRNGKey
    logits: networks_lib.Logits
    recurrent_state: actor_core_lib.RecurrentState
    prev_recurrent_state: actor_core_lib.RecurrentState


AtariVAPORLitePolicy = actor_core_lib.ActorCore[
    AtariVAPORLiteActorState[actor_core_lib.RecurrentState], AtariVAPORLiteExtras
]


def get_actor_core(
    networks: atari_vapor_lite_networks.AtariVAPORLiteNetworks,
    environment_spec: specs.EnvironmentSpec,
    evaluation: bool = False,
) -> AtariVAPORLitePolicy:
    """Creates a VAPOR-lite ActorCore."""

    dummy_logits = jnp.zeros(environment_spec.actions.num_values)

    def init(
        rng: jax_types.PRNGKey,
    ) -> AtariVAPORLiteActorState[actor_core_lib.RecurrentState]:
        rng, init_state_rng = jax.random.split(rng)
        initial_state = networks.policy_network.init_recurrent_state(
            init_state_rng, None
        )
        return AtariVAPORLiteActorState(
            rng=rng,
            logits=dummy_logits,
            recurrent_state=initial_state,
            prev_recurrent_state=initial_state,
        )

    def select_action(
        params: networks_lib.Params,
        observation: networks_lib.Observation,
        state: AtariVAPORLiteActorState[actor_core_lib.RecurrentState],
    ) -> Tuple[
        networks_lib.Action, AtariVAPORLiteActorState[actor_core_lib.RecurrentState]
    ]:

        rng, apply_rng, policy_rng = jax.random.split(state.rng, 3)
        (logits, _), new_recurrent_state = networks.policy_network.apply(
            params,
            apply_rng,
            observation,
            state.recurrent_state,
        )

        if evaluation:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = jax.random.categorical(policy_rng, logits)

        return action, AtariVAPORLiteActorState(
            rng=rng,
            logits=logits,
            recurrent_state=new_recurrent_state,
            prev_recurrent_state=state.recurrent_state,
        )

    def get_extras(
        state: AtariVAPORLiteActorState[actor_core_lib.RecurrentState],
    ) -> AtariVAPORLiteExtras:
        return {"logits": state.logits, "core_state": state.prev_recurrent_state}

    return actor_core_lib.ActorCore(
        init=init, select_action=select_action, get_extras=get_extras
    )
