import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from practice_rays.hiearchial_env import HierarchicalEnv


env = HierarchicalEnv(env_config=None)
high_level_obs_space = env.observation_spaces['high_level_agent']
high_level_act_space = env.action_spaces['high_level_agent']

low_attk_obs_space = env.observation_spaces['low_attack_agent']
low_avoid_obs_space = env.observation_spaces['low_avoid_agent']\
    
low_attk_act_space = env.action_spaces['low_attack_agent']
low_avoid_act_space = env.action_spaces['low_avoid_agent']

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents to their respective policies."""
    if agent_id == "high_level_agent":
        return "high_level_agent"
    elif agent_id == "low_attack_agent":
        return "low_attack_agent"
    elif agent_id == "low_avoid_agent":
        return "low_avoid_agent"
    else:
        raise ValueError(f"Unknown agent ID: {agent_id}")

# Define the multi-agent RL training setup
config = (
    PPOConfig()
    .environment(env=HierarchicalEnv)
    .multi_agent(
        policies={
            "high_level_agent": (None, high_level_obs_space, high_level_act_space, {}),
            "low_attack_agent": (None, low_attk_obs_space, low_attk_act_space, {}),
            "low_avoid_agent": (None, low_avoid_obs_space, low_avoid_act_space, {}),
        },
        policy_mapping_fn=policy_mapping_fn  # ✅ Correctly map actions
    )
)

tuner = tune.Tuner("PPO", param_space=config)
tuner.fit()