import dataclasses
from typing import Optional, Union

import numpy as np
import optax
from acme import types
from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class BsuiteVAPORLiteConfig:
    """Configuration options for VAPOR-lite on Bsuite."""

    seed: int = 0
    discount: float = 0.99

    # Optimizer configuration.
    learning_rate: Union[float, optax.Schedule] = 3e-3

    # Loss configuration.
    entropy_cost: float = 1.0
    reward_std: float = 0.1
    td_lambda: float = 0.9

    # Uncertainty bonus configuration.
    sigma_scale: float = 3.0

    # Replay options
    samples_per_insert: Optional[float] = 1.0
    min_replay_size: int = 100_000
    max_replay_size: int = 1_000_000
    online_queue_capacity: int = 0  # If not set, will use 4 * online_batch_size.
    replay_fraction: float = 0.995
    batch_size: int = 16
