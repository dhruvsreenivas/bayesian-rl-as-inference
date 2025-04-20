"""Atari VAPOR-lite config."""

import dataclasses
from typing import Optional, Union

import numpy as np
import optax
from acme import types
from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class AtariVAPORLiteConfig:
    """Configuration options for VAPOR-lite on Atari."""

    seed: int = 0
    discount: float = 0.99
    sequence_length: int = 20
    sequence_period: Optional[int] = None
    variable_update_period: int = 1000

    # Optimizer configuration.
    batch_size: int = 32
    learning_rate: Union[float, optax.Schedule] = 1e-4
    adam_momentum_decay: float = 0.0
    adam_variance_decay: float = 0.99
    adam_eps: float = 1e-8
    adam_eps_root: float = 0.0
    max_gradient_norm: float = 40.0

    # Loss configuration.
    baseline_cost: float = 0.5
    entropy_cost: float = 0.01
    max_abs_reward: float = np.inf
    reward_std: float = 0.1
    td_lambda: float = 0.9

    # Replay options
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
    num_prefetch_threads: Optional[int] = None
    samples_per_insert: Optional[float] = 1.0
    max_queue_size: Union[int, types.Batches] = types.Batches(10)
    replay_fraction: float = 0.9

    def __post_init__(self):
        if isinstance(self.max_queue_size, types.Batches):
            self.max_queue_size *= self.batch_size
        assert (
            self.max_queue_size > self.batch_size + 1
        ), """
        max_queue_size must be strictly larger than the batch size:
        - during the last step in an episode we might write 2 sequences to
          Reverb at once (that's how SequenceAdder works)
        - Reverb does insertion/sampling in multiple threads, so data is
          added asynchronously at unpredictable times. Therefore we need
          additional buffer size in order to avoid deadlocks."""
