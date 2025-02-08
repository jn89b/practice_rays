import gymnasium as gym
import numpy as np
import pickle


class NormalizeObservationWrapper(gym.Wrapper):
    """
    A wrapper that normalizes the observations for a multi-agent environment.

    For each agent, it keeps running estimates of the mean and variance of the 
    observation vectors. At every reset and step it updates these estimates and 
    returns a normalized observation. In case an agent’s observation is a dictionary 
    (as for our low-level agents), only the numeric part (the value under key "observations")
    is normalized.

    The wrapper also includes a method to save the running mean and standard deviation.
    """

    def __init__(self, env):
        super(NormalizeObservationWrapper, self).__init__(env)
        # We assume that the underlying env has an attribute `agents` listing the agent names.
        self.agents = env.agents

        # For each agent, we keep track of the running mean, M2 (for variance calculation), and count.
        self.running_means = {}
        self.running_M2 = {}
        self.counts = {}

        for agent in self.agents:
            # Figure out the shape of the observation we want to normalize.
            # For low-level agents, we assume the observation is a dict with an "observations" key.
            obs_space = (env.observation_spaces.spaces[agent]
                         if isinstance(env.observation_spaces, gym.spaces.Dict)
                         else env.observation_spaces)

            if isinstance(obs_space, gym.spaces.Dict):
                # For dict observations, we only normalize the "observations" field.
                shape = obs_space.spaces["observations"].shape
            else:
                shape = obs_space.shape

            self.running_means[agent] = np.zeros(shape, dtype=np.float32)
            self.running_M2[agent] = np.zeros(shape, dtype=np.float32)
            # Small epsilon to avoid division by zero
            self.counts[agent] = 1e-4

    def _update_stats(self, agent, x):
        """
        Update running statistics for the given agent.

        Uses Welford’s online algorithm.
        """
        x = np.array(x, dtype=np.float32)
        count = self.counts[agent] + 1
        delta = x - self.running_means[agent]
        new_mean = self.running_means[agent] + delta / count
        delta2 = x - new_mean
        new_M2 = self.running_M2[agent] + delta * delta2

        self.running_means[agent] = new_mean
        self.running_M2[agent] = new_M2
        self.counts[agent] = count

    def _normalize(self, agent, x):
        """
        Normalize observation x for the given agent.
        """
        # Calculate variance (and then standard deviation) from M2
        variance = self.running_M2[agent] / self.counts[agent]
        std = np.sqrt(variance)
        return (x - self.running_means[agent]) / (std + 1e-8)

    def reset(self, **kwargs):
        # Reset the underlying environment.
        obs, info = self.env.reset(**kwargs)
        norm_obs = {}
        for agent, o in obs.items():
            if isinstance(o, dict) and "observations" in o:
                raw_obs = o["observations"]
                self._update_stats(agent, raw_obs)
                norm_obs[agent] = {
                    "observations": self._normalize(agent, raw_obs),
                    "action_mask": o["action_mask"]
                }
            else:
                self._update_stats(agent, o)
                norm_obs[agent] = self._normalize(agent, o)
        return norm_obs, info

    def step(self, action_dict):
        # Take a step in the underlying environment.
        obs, rewards, terminateds, truncateds, infos = self.env.step(
            action_dict)
        norm_obs = {}
        for agent, o in obs.items():
            if isinstance(o, dict) and "observations" in o:
                raw_obs = o["observations"]
                self._update_stats(agent, raw_obs)
                norm_obs[agent] = {
                    "observations": self._normalize(agent, raw_obs),
                    "action_mask": o["action_mask"]
                }
            else:
                self._update_stats(agent, o)
                norm_obs[agent] = self._normalize(agent, o)

        print("norm_obs", norm_obs)
        return norm_obs, rewards, terminateds, truncateds, infos

    def save_stats(self, filename):
        """
        Save the current running mean and standard deviation for each agent to disk.
        """
        stats = {
            "means": self.running_means,
            "stds": {
                agent: np.sqrt(self.running_M2[agent] / self.counts[agent])
                for agent in self.agents
            },
            "counts": self.counts
        }
        with open(filename, "wb") as f:
            pickle.dump(stats, f)
        print(f"Normalization statistics saved to {filename}")

# -----------------------
# Example usage:
# -----------------------
# from your_module import ActionMaskingHRLEnv  (assuming you have imported your env)
#
# env = ActionMaskingHRLEnv(env_config)
# env = NormalizeObservationWrapper(env)
#
# # In your training loop, call env.reset() and env.step() as usual.
#
# # At the end of training you can save the normalization statistics:
# env.save_stats("normalization_stats.pkl")
