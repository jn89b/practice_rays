import gymnasium as gym
import numpy as np
from typing import List, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv


MIN_X: int = 0
MAX_X: int = 10
MIN_Y: int = 0
MAX_Y: int = 10


class PrisonerGuardEnv(MultiAgentEnv):
    """
    Two agents, one is the prisoner and the other is the guard.
    The prisoner has to escape the guard to get a reward.
    The guard has to catch the prisoner

    Observation spaces are :
        - x, y position of ego agent
        - distance between ego agent and other agent

    """

    def __init__(self, config=None):
        self.config = config

        self.agents: List[str] = ["prisoner", "guard"]
        # agent iterator
        self.agent_iter = iter(self.agents)
        # get the next agent

        self.observation_spaces: Dict[str, gym.spaces.Box] = {}
        self.action_spaces: Dict[str, gym.spaces.Discrete] = {}

        for agent in self.agents:
            low_obs: np.ndarray = np.array([MIN_X, MIN_Y, 0])
            high_obs: np.ndarray = np.array(
                [MAX_X, MAX_Y, np.sqrt(MAX_X ** 2 + MAX_Y ** 2)])

            self.observation_spaces[agent] = gym.spaces.Box(
                low=low_obs, high=high_obs, dtype=np.float32
            )

        for agent in self.agents:
            self.action_spaces[agent] = gym.spaces.Discrete(4)

        self.action_map: Dict[int, int] = {0: (-1, 0),  # Left
                                           1: (1, 0),  # Right
                                           2: (0, -1),  # Down
                                           3: (0, 1)}  # Up

        self.last_move = None
        self.start_time: int = 0
        self.max_steps: int = 100
        self.spawn_agents()
        self.current_player: str = next(self.agent_iter)

    def spawn_agents(self):
        self.prisoner_x: int = 0  # np.random.randint(MIN_X, MAX_X)
        self.prisoner_y: int = 0  # np.random.randint(MIN_Y, MAX_Y)

        self.guard_x: int = np.random.randint(MIN_X, MAX_X)
        self.guard_y: int = np.random.randint(MIN_Y, MAX_Y)

        self.goal_x = 6
        self.goal_y = 6

    def reset(self, *, seed=None, options=None) -> Dict[str, np.ndarray]:
        self.start_time = 0
        self.spawn_agents()

        # The first observation should not matter (none of the agents has moved yet).
        # Set them to 0.
        # self.current_player = next(self.agent_iter)
        # current_obs = self.observe(self.current_player)
        observations = {agent: self.observe(agent) for agent in self.agents}

        return observations, {}

    def observe(self, agent: str) -> np.ndarray:
        if agent == "prisoner":
            return np.array([self.prisoner_x, self.prisoner_y, np.sqrt(
                (self.guard_x - self.prisoner_x) ** 2 + (self.guard_y - self.prisoner_y) ** 2)],
                dtype=np.float32)
        else:
            return np.array([self.guard_x, self.guard_y, np.sqrt(
                (self.prisoner_x - self.guard_x) ** 2 + (self.prisoner_y - self.guard_y) ** 2)],
                dtype=np.float32)

    def get_move(self, action: int) -> tuple:
        if action not in self.action_map:
            raise ValueError(f"Invalid action {action}")

        return self.action_map[action]

    def is_out_of_bounds(self, x: int, y: int) -> bool:
        return x < MIN_X or x > MAX_X or y < MIN_Y or y > MAX_Y

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(self.action_spaces)

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(self.observation_spaces)

    def step(self, action_dict: Dict[str, int]) -> Dict[str, tuple]:
        # action = action_dict[self.current_player]
        # move = self.get_move(action)

        # rewards
        rewards = {agent: 0 for agent in self.agents}
        terminateds = {agent: False for agent in self.agents}
        for agent in self.agents:
            x_cmd, y_cmd = self.get_move(action_dict[agent])
            if agent == "prisoner":
                self.prisoner_x += x_cmd
                self.prisoner_y += y_cmd
            else:
                self.guard_x += x_cmd
                self.guard_y += y_cmd

        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards["prisoner"] = -1
            rewards["guard"] = 1
            # Terminate the entire episode (for all agents) once 10 moves have been made.
            terminateds = {"__all__": True}
        elif self.prisoner_x == self.goal_x and self.prisoner_y == self.goal_y:
            rewards["prisoner"] = 1
            rewards["guard"] = -1
            terminateds = {"__all__": True}

        elif self.is_out_of_bounds(self.prisoner_x, self.prisoner_y):
            rewards["prisoner"] = -1
            rewards["guard"] = 1
            terminateds = {"__all__": True}
        elif self.is_out_of_bounds(self.guard_x, self.guard_y):
            rewards["prisoner"] = 1
            rewards["guard"] = -1
            terminateds = {"__all__": True}

        if self.start_time <= self.max_steps:
            terminateds = {"__all__": True}

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminateds, {}, infos
