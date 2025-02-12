import unittest
import ray
from practice_rays.prisoner_guard_env import PrisonerGuardEnv
from practice_rays.callbacks import EpisodeReturn, ExampleEnvCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.train import RunConfig


ray.init()
test_env = PrisonerGuardEnv()
prisoner_obs = test_env.observation_spaces["prisoner"]
prisoner_act = test_env.action_spaces["prisoner"]
guard_obs = test_env.observation_spaces["guard"]
guard_act = test_env.action_spaces["guard"]
config = (
    PPOConfig()
    # Use your custom class directly
    .environment(env=PrisonerGuardEnv)
    # use gpu
    .multi_agent(
        policies={
            "prisoner": (None, prisoner_obs, prisoner_act, None),
            "guard": (None, guard_obs, guard_act, None),
        },
        # Simple mapping - each agent gets its own policy
        policy_mapping_fn=lambda agent_id, episode, 
            **kwargs: agent_id,
        # Different hyperparameters for each agent type
        algorithm_config_overrides_per_module={
            "prisoner": PPOConfig.overrides(gamma=0.85),
            "guard": PPOConfig.overrides(lr=0.00001),
        },
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs={
            "prisoner": RLModuleSpec(),
            "guard": RLModuleSpec(),
        }),
    )
    .training(
        lr=tune.grid_search([0.01, 0.001]),
    )
    .callbacks(
        callbacks_class=[EpisodeReturn, ExampleEnvCallback],
    )
)

# algo = config.build()
# print(algo.train())

# train with tune
# https://docs.ray.io/en/latest/tune/tutorials/tune-output.html


tuner = tune.Tuner(
    "PPO",
    param_space=config
)

tuner.fit()
