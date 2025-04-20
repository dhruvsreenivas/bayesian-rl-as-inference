from absl import app, flags
from acme import specs
from acme.jax import experiments

from rl import environment as bsuite_environment
from rl import vapor_lite_bsuite
from rl.vapor_lite_bsuite import builder as bsuite_vapor_lite_builder

FLAGS = flags.FLAGS


ENV_NAME = flags.DEFINE_string("env_name", "deepsea/0", "What environment to run")
RESULTS_DIR = flags.DEFINE_string(
    "results_dir", "/network/scratch/d/dhruv.sreenivas/bsuite", "CSV results directory."
)
OVERWRITE = flags.DEFINE_boolean(
    "overwrite", False, "Whether to overwrite csv results."
)
SEED = flags.DEFINE_integer("seed", 0, "Random seed (experiment).")
NUM_ACTOR_STEPS = flags.DEFINE_integer(
    "num_steps", 1_000_000, "Number of env steps to run."
)

_BATCH_SIZE = 16


def build_experiment_config():
    """Builds Bsuite VAPOR-lite experiment config which can be executed in different ways."""

    # Create an environment, grab the spec, and use it to create networks.
    env_name = ENV_NAME.value

    def env_factory(seed):
        del seed
        return bsuite_environment.make_bsuite_environment(
            bsuite_id=env_name, results_dir=RESULTS_DIR.value, overwrite=OVERWRITE.value
        )

    # Construct the agent.
    config = vapor_lite_bsuite.BsuiteVAPORLiteConfig(
        batch_size=_BATCH_SIZE, learning_rate=1e-4, td_lambda=0.9
    )
    return experiments.ExperimentConfig(
        builder=bsuite_vapor_lite_builder.BsuiteVAPORLiteBuilder(config),
        environment_factory=env_factory,
        network_factory=vapor_lite_bsuite.make_networks,
        seed=SEED.value,
        max_num_actor_steps=NUM_ACTOR_STEPS.value,
    )


def main(_):
    experiment_config = build_experiment_config()
    experiments.run_experiment(experiment_config)


if __name__ == "__main__":
    app.run(main)
