"""Example running SAC on continuous control tasks."""

import launchpad as lp
from absl import app, flags
from acme import specs
from acme.agents.jax import normalization
from acme.jax import experiments
from acme.utils import lp_utils

from rl import environment as control
from rl import sac
from rl.sac import builder

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "run_distributed",
    True,
    "Should an agent be executed in a distributed "
    "way. If False, will run single-threaded.",
)
flags.DEFINE_string("env_name", "gym:HalfCheetah-v2", "What environment to run")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 1_000_000, "Number of env steps to run.")
flags.DEFINE_integer("eval_every", 50_000, "How often to run evaluation.")
flags.DEFINE_integer("evaluation_episodes", 10, "Evaluation episodes.")


def build_experiment_config():
    """Builds SAC experiment config which can be executed in different ways."""
    # Create an environment, grab the spec, and use it to create networks.

    suite, task = FLAGS.env_name.split(":", 1)
    environment = control.make_control_environment(suite, task)

    environment_spec = specs.make_environment_spec(environment)
    network_factory = lambda spec: sac.make_networks(
        spec, hidden_layer_sizes=(256, 256, 256)
    )

    # Construct the agent.
    config = sac.SACConfig(
        learning_rate=3e-4,
        n_step=2,
        target_entropy=sac.target_entropy_from_env_spec(environment_spec),
        input_normalization=normalization.NormalizationConfig(),
    )
    sac_builder = builder.SACBuilder(config)

    return experiments.ExperimentConfig(
        builder=sac_builder,
        environment_factory=lambda seed: control.make_control_environment(suite, task),
        network_factory=network_factory,
        seed=FLAGS.seed,
        max_num_actor_steps=FLAGS.num_steps,
    )


def main(_):
    config = build_experiment_config()
    if FLAGS.run_distributed:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=4
        )
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        experiments.run_experiment(
            experiment=config,
            eval_every=FLAGS.eval_every,
            num_eval_episodes=FLAGS.evaluation_episodes,
        )


if __name__ == "__main__":
    app.run(main)
