import torch
import gymnasium
import numpy as np
import torch.nn as nn
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
import ray

from typing import Set, Dict
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from practice_rays.hiearchial_env import HiearchialEnvV2 as HierarchicalEnv
from practice_rays.hiearchial_env import ActionMaskingHRLEnv
from practice_rays.normalize_wrapper import NormalizeObservationWrapper
from ray.rllib.examples.envs.classes.six_room_env import (
    HierarchicalSixRoomEnv
)

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from ray.rllib.models import ModelCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.connectors.env_to_module.mean_std_filter import MeanStdFilter

import numpy as np


class TorchMaskedActions(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)

        # Define the internal model for processing observations
        self.internal_model = TorchFC(
            obs_space['observations'], action_space, num_outputs, model_config, name + "_internal"
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the action mask from the observation
        action_mask = input_dict["obs"]["action_mask"]

        # Extract the actual observations
        observations = input_dict["obs"]["observations"]

        # Compute the logits for all actions using the internal model
        logits, _ = self.internal_model({"obs": observations})

        # Apply the action mask: set logits of invalid actions to a very low value
        inf_mask = torch.clamp(torch.log(action_mask), -
                               1e10, np.finfo(np.float32).max)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


def create_env(config):
    return HierarchicalSixRoomEnv(config)


# def test_hrl_room_env() -> None:
#     env = HierarchicalSixRoomEnv({})
#     for i in range(10):
#         # random action
#         if env._high_level_action is None:
#             action_sample = env.action_spaces['high_level_agent'].sample()
#             action = {"high_level_agent": action_sample}

#         print("high level action", env._high_level_action)
#         obs, rewards, dones, _, infos = env.step(action)
#         print("observation", obs)
#         print("rewards", rewards)
#         print("dones", dones)
#         print("\n")


# Register the custom model
ModelCatalog.register_custom_model("masked_action_model", TorchMaskedActions)


# For debugging purposes
ray.init(local_mode=True)

# tune.register_env("env", HierarchicalEnv)
tune.register_env("mask_env", ActionMaskingHRLEnv)


def policy_mapping_fn(agent_id, episode, **kwarg):
    """Map agents to their respective policies."""
    if agent_id == "high_level_agent":
        return "high_level_agent"
    elif agent_id == "low_attack_agent":
        return "low_attack_agent"
    elif agent_id == "low_avoid_agent":
        return "low_avoid_agent"
    else:
        raise ValueError(f"Unknown agent ID: {agent_id}")


def train_heuristic_hrl() -> None:
    """
    Train HRL hiearchial model
    """
    env = HierarchicalEnv(env_config=None)
    high_level_obs_space = env.observation_spaces['high_level_agent']
    high_level_act_space = env.action_spaces['high_level_agent']

    low_attk_obs_space = env.observation_spaces['low_attack_agent']
    low_avoid_obs_space = env.observation_spaces['low_avoid_agent']\

    low_attk_act_space = env.action_spaces['low_attack_agent']
    low_avoid_act_space = env.action_spaces['low_avoid_agent']

    # Define the multi-agent RL training setup
    config = (
        PPOConfig()
        .environment(env="env")
        .multi_agent(
            policies={
                "high_level_agent": (None, high_level_obs_space, high_level_act_space, {}),
                "low_attack_agent": (None, low_attk_obs_space, low_attk_act_space, {}),
                "low_avoid_agent": (None, low_avoid_obs_space, low_avoid_act_space, {}),
            },
            policy_mapping_fn=policy_mapping_fn  # ✅ Correctly map actions
        )
        .env_runners(num_env_runners=3)
        .resources(num_gpus=1)
    )

    run_config = tune.RunConfig(
        stop={"training_iteration": 100},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=20,  # Save every n iterations
            checkpoint_at_end=True,  # Always save final checkpoint
            num_to_keep=5,  # Keep only the top n checkpoint
        ),
    )
    tuner = tune.Tuner("PPO", param_space=config,
                       run_config=run_config)
    tuner.fit()


def train_ppo_mask_hrl():
    """
    Pass

    This is how you would set up a training environment
    for maskable PPO agents in a HRL environment.

    Biggest thing you want to look at is look at how
    rl_module is set up in the config. This is where
    you can set up the different RL modules for each
    agent in the environment
    """
    mask_env = ActionMaskingHRLEnv(env_config=None)
    high_level_obs_space = mask_env.observation_spaces['high_level_agent']
    high_level_act_space = mask_env.action_spaces['high_level_agent']

    low_attk_obs_space = mask_env.observation_spaces['low_attack_agent']
    low_avoid_obs_space = mask_env.observation_spaces['low_avoid_agent']\

    low_attk_act_space = mask_env.action_spaces['low_attack_agent']
    low_avoid_act_space = mask_env.action_spaces['low_avoid_agent']

    # # Define the run configuration
    run_config = tune.RunConfig(
        stop={"training_iteration": 100},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True,
            num_to_keep=5,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
        ),
    )

    config = (
        PPOConfig()
        .environment(env="mask_env")
        .framework("torch")
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "high_level_agent": RLModuleSpec(
                        observation_space=high_level_obs_space,
                        action_space=high_level_act_space,
                        model_config={}
                    ),
                    "low_attack_agent": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=low_attk_obs_space,
                        action_space=low_attk_act_space,
                        model_config={}
                    ),
                    "low_avoid_agent": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=low_avoid_obs_space,
                        action_space=low_avoid_act_space,
                        model_config={}
                    ),
                }
            )
        )
        .multi_agent(
            policies=["high_level_agent",
                      "low_attack_agent", "low_avoid_agent"],
            policy_mapping_fn=policy_mapping_fn
        )
        .resources(num_gpus=1)
    )

    # Initialize and run the tuner
    tuner = tune.Tuner("PPO", param_space=config, run_config=run_config)
    tuner.fit()


def train_ppo_mask_hrl_normalized():
    """
    Sets up and runs training of maskable PPO agents in a hierarchical RL environment
    that uses a normalization wrapper to scale observations.

    This function registers a normalized environment with RLlib, extracts the
    observation and action spaces from a temporary instance, and builds a PPO config
    that uses these spaces.
    """
    # First define an environment creator that returns your normalized environment.
    def normalized_env_creator(env_config):
        base_env = ActionMaskingHRLEnv(env_config=env_config)
        # Wrap the environment so that observations are normalized.
        return NormalizeObservationWrapper(base_env)

    # Register the normalized environment with RLlib.
    tune.register_env("normalized_mask_env", normalized_env_creator)

    # Create a temporary environment instance to extract observation and action spaces.
    temp_env = ActionMaskingHRLEnv(env_config={})

    high_level_obs_space = temp_env.observation_spaces['high_level_agent']
    high_level_act_space = temp_env.action_spaces['high_level_agent']

    low_attk_obs_space = temp_env.observation_spaces['low_attack_agent']
    low_avoid_obs_space = temp_env.observation_spaces['low_avoid_agent']

    low_attk_act_space = temp_env.action_spaces['low_attack_agent']
    low_avoid_act_space = temp_env.action_spaces['low_avoid_agent']

    temp_env.close()  # Clean up the temporary environment.

    # Define the run configuration for Ray Tune.
    run_config = tune.RunConfig(
        stop={"training_iteration": 100},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True,
            num_to_keep=5,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
        ),
    )

    # Build the PPO configuration.
    config = (
        PPOConfig()
        # Use the registered normalized env.
        .environment(env="mask_env")
        .framework("torch")
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "high_level_agent": RLModuleSpec(
                        observation_space=high_level_obs_space,
                        action_space=high_level_act_space,
                        model_config={}
                    ),
                    "low_attack_agent": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=low_attk_obs_space,
                        action_space=low_attk_act_space,
                        model_config={}
                    ),
                    "low_avoid_agent": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=low_avoid_obs_space,
                        action_space=low_avoid_act_space,
                        model_config={}
                    ),
                }
            )
        )
        .multi_agent(
            policies=["high_level_agent",
                      "low_attack_agent", "low_avoid_agent"],
            policy_mapping_fn=policy_mapping_fn
        )
        .resources(num_gpus=1)
        .env_runners(observation_filter="MeanStdFilter")
    )

    # Initialize and run the training using Ray Tune.
    tuner = tune.Tuner("PPO", param_space=config, run_config=run_config)
    tuner.fit()


if __name__ == "__main__":
    # train_heuristic_hrl()
    # train_ppo_mask_hrl()
    train_ppo_mask_hrl_normalized()
