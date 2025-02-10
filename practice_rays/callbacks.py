# https://docs.ray.io/en/latest/rllib/rllib-callback.html
import math
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

"""
Toy examples of how to use RLlib callbacks.
https://discuss.ray.io/t/log-or-record-custom-env-data-via-rllib/4674
"""

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

class DemoCallback(RLlibCallback):
    def __init__(self) -> None:
        # keep track of number of wins between the two agents
        self.guard_win: int = 0
        self.prisoner_win: int = 0
        self.prisoner_loss: int = 0
        
    def print_goal(self, env, print_goal: bool = True):    
        # If I want to grab a specific attribute from my environment such as
        # goal positon here's what I can do
        goal_x = env.unwrapped.goal_x
        goal_y = env.unwrapped.goal_y
        if print_goal:  
            print(f"Goal position: ({goal_x}, {goal_y})")
            
    def get_win(self, env) -> bool:
        return env.unwrapped.prisoner_won
            
    def get_type(self, env):
        """
        Used to check if environment is a vectorized environment
        for easier processing
        """    
        if hasattr(env, "envs"):
            env = env.envs[0]
            #infos = env.unwrapped.get_info()
        else:
            env = env
            
        return env

    # def on_episode_start(self, *,
    #                      episode) -> None:
    #     """
    #     """        
    #     episode.hist_data["prisoner_wins"] = []
    #     episode.hist_data["prisoner_losses"] = []
    
    def on_episode_start(
        self, *, episode,  **kwargs
    ):
        episode.hist_data["prisoner_wins"] = []
        episode.hist_data["prisoner_losses"] = []
        print("Episode history data: ", episode.hist_data)

    def on_episode_step(self, *, episode, env, **kwargs) -> None:
        """
        """
        # if hasattr(env, "envs"):
        #     env = env.envs[0]
        #     #infos = env.unwrapped.get_info()
        # else:
        #     env = env
        env = self.get_type(env)
        self.print_goal(env, print_goal=False)
        
    def on_episode_end(self, *, episode, env, **kwargs) -> None:
        """
        """
        env = self.get_type(env)
        episode = episode
        observations = episode.get_observations()
        infos = episode.get_infos()
        rewards = episode.get_rewards()

        # let's log the win count
        high_level_reward  = rewards['high_level_agent'][0]
        # who_won = infos["prisoner_won"]
        # print(f"Prisoner won: {who_won}")
        prisoner_won = self.get_win(env)
        if prisoner_won:
            self.prisoner_win += 1
        else:
            self.prisoner_loss += 1
    
        # # Log the theta1 degree value in the episode object, temporarily.
        episode.hist_data["prisoner_wins"].append(self.prisoner_win)
        episode.hist_data["prisoner_losses"].append(self.prisoner_loss)
        
        prisoner_wins = episode.hist_data["prisoner_wins"]
        prisoner_losses = episode.hist_data["prisoner_losses"]

        sum_wins = sum(prisoner_wins)
        sum_losses = sum(prisoner_losses)
        
        print("sum wins: ", sum_wins)
        print("sum losses: ", sum_losses)
        
        # episode.add_temporary_timestep_data("prisoner_wins", self.prisoner_win)
        # episode.add_temporary_timestep_data("prisoner_losses", self.prisoner_loss)
       
        # print(f"Episode {episode.episode_id} ended with length {episode.length}")

    def on_train_result(self, result: dict, **kwargs) -> None:
        # https://discuss.ray.io/t/rllib-callbacks-to-get-custom-metrics-such-as-observation-reward-etc-in-each-episode-from-singleagentepisode-and-access-it-in-the-trainer/16022/2
        # Log the metrics for this iteration.
        result.setdefault("custom_metrics", {})
        # get win from episode  
        # result["custom_metrics"]["prisoner_wins"] = self.prisoner_win
        # result["custom_metrics"]["prisoner_losses"] = self.prisoner_loss

        # # Calculate win rate if there are any episodes.
        # total = self.prisoner_win + self.prisoner_loss
        # win_rate = self.prisoner_win / total if total > 0 else 0.0
        # result["custom_metrics"]["prisoner_win_rate"] = win_rate

        # print(
        #     f"Iteration {result.get('training_iteration', 'unknown')}: "
        #     f"Win rate = {win_rate:.2f} "
        #     f"(Wins: {self.prisoner_win}, Losses: {self.prisoner_loss})"
        # )

        # # Reset the counters for the next iteration.
        # self.prisoner_win = 0
        # self.prisoner_loss = 0        
        
    # def on_episode_end(self, *, episode, metrics_logger, **kwargs):
    #     # Get all the logged theta1 degree values and average them.
    #     theta1s = episode.get_temporary_timestep_data("theta1")
    #     avg_theta1 = np.mean(theta1s)

    #     # Log the final result - per episode - to the MetricsLogger.
    #     # Report with a sliding/smoothing window of 50.
    #     metrics_logger.log_value("theta1_mean", avg_theta1, reduce="mean", window=50)