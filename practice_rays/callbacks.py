# https://docs.ray.io/en/latest/rllib/rllib-callback.html
import math
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        # Keep some global state in between individual callback events.
        self.overall_sum_of_rewards = 0.0

    def on_episode_end(self, *, episode, **kwargs):
        self.overall_sum_of_rewards += episode.get_return()
        print(f"Episode done. R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")

class ExampleEnvCallback(RLlibCallback):
    def on_episode_step(self, *, episode, env, **kwargs):
        # First get the angle from the env (note that `env` is a VectorEnv).
        # See https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/acrobot.py
        # for the env source code.
        # cos_theta1, sin_theta1 = env.envs[0].unwrapped.state[0], env.envs[0].unwrapped.state[1]
        # # Convert cos/sin/tan into degree.
        # deg_theta1 = math.degrees(math.atan2(sin_theta1, cos_theta1))

        # # Log the theta1 degree value in the episode object, temporarily.
        # episode.add_temporary_timestep_data("theta1", deg_theta1)
        print(env.envs[0].unwrapped.state)
        

    # def on_episode_end(self, *, episode, metrics_logger, **kwargs):
    #     # Get all the logged theta1 degree values and average them.
    #     theta1s = episode.get_temporary_timestep_data("theta1")
    #     avg_theta1 = np.mean(theta1s)

    #     # Log the final result - per episode - to the MetricsLogger.
    #     # Report with a sliding/smoothing window of 50.
    #     metrics_logger.log_value("theta1_mean", avg_theta1, reduce="mean", window=50)