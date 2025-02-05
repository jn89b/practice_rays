import numpy as np
from ray.rllib.policy.policy import Policy

class MetaPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.use_heuristic = config.get("use_heuristic", False)  # Allow switching

    def compute_actions(self, obs_batch, **kwargs):
        actions = []
        for obs in obs_batch:
            if self.use_heuristic:
                # Example heuristic: Attack if close, Avoid if far
                action = 0 if obs[0] < 10 else 1  # obs[0] = distance_to_target
            else:
                # Placeholder for learned policy (random for now, should be RL)
                action = np.random.choice([0, 1])
            actions.append(action)
        return np.array(actions), [], {}
