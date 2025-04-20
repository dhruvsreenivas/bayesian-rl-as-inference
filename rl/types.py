from typing import Callable, Mapping

import chex

ActivationFn = Callable[[chex.Array], chex.Array]
Metrics = Mapping[str, chex.Array]


@chex.dataclass
class ActorCriticOutput:
    logits: chex.Array
    value: chex.Array
