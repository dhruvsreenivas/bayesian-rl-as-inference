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

"""Shared helpers for rl_continuous and rl_discrete experiments."""

import functools
from typing import Tuple

import bsuite
import dm_env
import gym
from acme import wrappers

_VALID_TASK_SUITES = ("gym", "control")


def make_control_environment(suite: str, task: str) -> dm_env.Environment:
    """Makes the requested continuous control environment.

    Args:
      suite: One of 'gym' or 'control'.
      task: Task to load. If `suite` is 'control', the task must be formatted as
        f'{domain_name}:{task_name}'

    Returns:
      An environment satisfying the dm_env interface expected by Acme agents.
    """

    if suite not in _VALID_TASK_SUITES:
        raise ValueError(
            f"Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}"
        )

    if suite == "gym":
        env = gym.make(task)
        # Make sure the environment obeys the dm_env.Environment interface.
        env = wrappers.GymWrapper(env)

    elif suite == "control":
        # Load dm_suite lazily not require Mujoco license when not using it.
        from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top

        domain_name, task_name = task.split(":")
        env = dm_suite.load(domain_name, task_name)
        env = wrappers.ConcatObservationWrapper(env)

    # Wrap the environment so the expected continuous action spec is [-1, 1].
    # Note: this is a no-op on 'control' tasks.
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    return env


def make_atari_environment(
    level: str = "Pong",
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False,
    oar_wrapper: bool = False,
    num_stacked_frames: int = 4,
    flatten_frame_stack: bool = False,
    grayscaling: bool = True,
    to_float: bool = True,
    scale_dims: Tuple[int, int] = (84, 84),
) -> dm_env.Environment:
    """Loads the Atari environment."""
    # Internal logic.
    version = "v0" if sticky_actions else "v4"
    level_name = f"{level}NoFrameskip-{version}"
    env = gym.make(level_name, full_action_space=True)

    wrapper_list = [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            scale_dims=scale_dims,
            to_float=to_float,
            max_episode_len=108_000,
            num_stacked_frames=num_stacked_frames,
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=grayscaling,
            zero_discount_on_life_loss=zero_discount_on_life_loss,
        ),
        wrappers.SinglePrecisionWrapper,
    ]

    if oar_wrapper:
        # E.g. IMPALA and R2D2 use this particular variant.
        wrapper_list.append(wrappers.ObservationActionRewardWrapper)

    return wrappers.wrap_all(env, wrapper_list)


def make_bsuite_environment(
    bsuite_id: str, results_dir: str, overwrite: bool
) -> dm_env.Environment:
    raw_environment = bsuite.load_and_record_to_csv(
        bsuite_id=bsuite_id,
        results_dir=results_dir,
        overwrite=overwrite,
    )
    environment = wrappers.SinglePrecisionWrapper(raw_environment)

    return environment
