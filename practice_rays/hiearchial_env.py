import numpy as np
import gymnasium as gym
from typing import List, Dict, Optional, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

MIN_X: int = 0
MIN_Y: int = 0
MAX_X: int = 10
MAX_Y: int = 10

ATTACK_IDX: int = 0
AVOID_IDX: int = 1

class HierarchicalEnv(MultiAgentEnv):
    """
    This environment consists of two policies:
        - A high level meta policy to decide whether to attack or avoid
        - A low level policy to execute the action
    """
    def __init__(self, env_config):
        super().__init__()
        
        self.agents:List[str] = ["high_level_agent", "low_attack_agent", 
                                 "low_avoid_agent"]
        
        # observation space for high level agent
        low_obs: np.ndarray = np.array([MIN_X, MIN_Y, 0], dtype=np.float32)
        high_obs: np.ndarray = np.array(
            [MAX_X, MAX_Y, np.sqrt(MAX_X ** 2 + MAX_Y ** 2)], dtype=np.float32)
        
        self.action_map: Dict[int, int] = {0: (-1, 0),  # Left
                                           1: (1, 0),  # Right
                                           2: (0, -1),  # Down
                                           3: (0, 1)}  # Up
        
        self.observation_spaces: Dict[str, gym.spaces.Box] = {
            "high_level_agent": gym.spaces.Discrete(2),  # Attack or Avoid,
            "low_attack_agent": gym.spaces.Box(low=low_obs, high=high_obs,
                                               dtype=np.float32),
            "low_avoid_agent": gym.spaces.Box(low=low_obs, high=high_obs,
                                              dtype=np.float32)
        }
        
        self.action_spaces: Dict[str, gym.spaces.Discrete] = {
            "high_level_agent": gym.spaces.Discrete(2),
            "low_attack_agent": gym.spaces.Discrete(4),
            "low_avoid_agent": gym.spaces.Discrete(4)
        }
        self.start_time:int = 0
        self.time_limit:int = 100
        self.spawn_agents()
        self._high_level_action: Optional[int] = None
        
    def spawn_agents(self):
        self.prisoner_x: int = 1  # np.random.randint(MIN_X, MAX_X)
        self.prisoner_y: int = 1  # np.random.randint(MIN_Y, MAX_Y)

        self.guard_x: int = np.random.randint(MIN_X, MAX_X)
        self.guard_y: int = np.random.randint(MIN_Y, MAX_Y)

        self.goal_x:int  = 6
        self.goal_y:int  = 6
        
    def observe(self, agent:str) -> np.ndarray:
        distance_to_goal: float = np.sqrt(
            self.prisoner_x - self.goal_x) ** 2 + (self.prisoner_y - self.goal_y) ** 2
        distance_from_guard: float = np.sqrt(
            self.guard_x - self.prisoner_x) ** 2 + (self.guard_y - self.prisoner_y) ** 2
        
        if agent == "high_level_agent":
            return np.array([distance_to_goal, distance_from_guard],
                dtype=np.float32)
        elif agent == "low_attack_agent":
            return np.array([self.prisoner_x, self.prisoner_y, distance_to_goal],
                dtype=np.float32)
        elif agent == "low_avoid_agent":
            return np.array([self.prisoner_x, self.prisoner_y, distance_from_guard],
                dtype=np.float32)
        
    def reset(self, *, seed=None, options=None):
        self.start_time = 0
        self.spawn_agents()

        # Return high-level observation.
        return {
            "high_level_agent": self.observe("high_level_agent"),
        }, {}
        
        
    def get_move(self, action: int) -> tuple:
        if action not in self.action_map:
            raise ValueError(f"Invalid action {action}")

        return self.action_map[action]
        
    def is_out_of_bounds(self, x: int, y: int) -> bool:
        return x < MIN_X or x > MAX_X or y < MIN_Y or y > MAX_Y
            
    def get_rewards(self,) -> Dict[str, float]:
        pass
            
    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, Dict]:
        terminateds = {agent: False for agent in self.agents}
        # rewards = {agent: 0 for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}
        rewards = {}
        high_level_action: int = action_dict["high_level_agent"]
        print(f"Action dict: {action_dict}")
        print(f"High level action: {high_level_action}")
        
        # Figure out the high level policy
        if "high_level_agent" in action_dict:
            # This will yield a high level policy where 0 is attack and 1 is avoid
            self._high_level_action = action_dict["high_level_agent"]
            if self._high_level_action == ATTACK_IDX:
                low_agent = "low_attack_agent"
            else:
                low_agent = "low_avoid_agent"
            
            observation = self.observe(low_agent)
            rewards["high_level_agent"] = 0
            
            return observation, rewards, terminateds, truncateds, {}
        
        # this is the low level agent
        else:
            self.start_time += 1
            
            if self._high_level_action is None:
                raise ValueError("High level action not set")
            elif self._high_level_action == ATTACK_IDX:
                agent = "low_attack_agent"
            else:
                agent = "low_avoid_agent"
            
            dx, dy = self.get_move(action_dict[agent])    
            self.prisoner_x += dx
            self.prisoner_y += dy
            observation = self.observe(agent)
            
            if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
                rewards["high_level_agent"] = -10.0
                rewards[agent] = -10.0
                terminateds[agent] = True
            elif self.prisoner_x == self.goal_x and self.prisoner_y == self.goal_y:
                rewards["high_level_agent"] = 10.0
                rewards[agent] = 10.0
                terminateds[agent] = True
            
            elif self.start_time >= self.time_limit:
                rewards["high_level_agent"] = 0.0
                rewards[agent] = 0.0
                terminateds[agent] = True
            elif self.is_out_of_bounds(self.prisoner_x, self.prisoner_y):
                rewards["high_level_agent"] = -10.0
                rewards[agent] = -10.0
                terminateds[agent] = True
                
            return observation, rewards, terminateds, truncateds, {}    
            