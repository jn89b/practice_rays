import numpy as np
import gymnasium as gym
from typing import List, Dict, Optional, Tuple, Any
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
# https://discuss.ray.io/t/multi-agent-setting-different-step-sizes-for-agents-and-how-actions-are-passed/5906

MIN_X: int = 0
MIN_Y: int = 0
MAX_X: int = 8
MAX_Y: int = 8

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

        self.agents: List[str] = ["high_level_agent", "low_attack_agent",
                                  "low_avoid_agent"]

        # observation space for high level agent
        low_obs: np.ndarray = np.array([MIN_X, MIN_Y, 0], dtype=np.float32)
        high_obs: np.ndarray = np.array(
            [MAX_X, MAX_Y, np.sqrt(MAX_X ** 2 + MAX_Y ** 2)], dtype=np.float32)

        low_high_obs: np.ndarray = np.array([0, 0], dtype=np.float32)
        high_high_obs: np.ndarray = np.array(
            [np.sqrt(MAX_X ** 2 + MAX_Y ** 2),
             np.sqrt(MAX_X ** 2 + MAX_Y ** 2)],
            dtype=np.float32)

        self.action_map: Dict[int, int] = {0: (-1, 0),  # Left
                                           1: (1, 0),  # Right
                                           2: (0, -1),  # Down
                                           3: (0, 1)}  # Up

        self.observation_spaces: Dict[str, gym.spaces.Box] = {
            # Attack or Avoid,
            "high_level_agent": gym.spaces.Box(low=low_high_obs, high=high_high_obs,
                                               dtype=np.float32),
            "low_attack_agent": gym.spaces.Box(low=low_obs, high=high_obs,
                                               dtype=np.float32),
            "low_avoid_agent": gym.spaces.Box(low=low_obs, high=high_obs,
                                              dtype=np.float32)
        }

        self.action_spaces: Dict[str, gym.spaces.Discrete] = {
            "high_level_agent": gym.spaces.Discrete(2),  # Attack or Avoid
            "low_attack_agent": gym.spaces.Discrete(4),
            "low_avoid_agent": gym.spaces.Discrete(4)
        }
        self.start_time: int = 0
        self.time_limit: int = 100
        self.spawn_agents()
        self._high_level_action: Optional[int] = None

    def spawn_agents(self):
        self.prisoner_x: int = 1  # np.random.randint(MIN_X, MAX_X)
        self.prisoner_y: int = 1  # np.random.randint(MIN_Y, MAX_Y)

        self.guard_x: int = np.random.randint(MIN_X, MAX_X)
        self.guard_y: int = np.random.randint(MIN_Y, MAX_Y)

        self.goal_x: int = 6
        self.goal_y: int = 6

    def observe(self, agent: str) -> np.ndarray:
        distance_to_goal: float = np.sqrt(
            (self.prisoner_x - self.goal_x) ** 2 +
            (self.prisoner_y - self.goal_y) ** 2
        )

        distance_from_guard: float = np.sqrt(
            (self.guard_x - self.prisoner_x) ** 2 +
            (self.guard_y - self.prisoner_y) ** 2
        )

        if agent == "high_level_agent":
            return np.array([distance_to_goal, distance_from_guard],
                            dtype=np.float32)
        elif agent == "low_attack_agent":
            return np.array([self.prisoner_x, self.prisoner_y, distance_to_goal],
                            dtype=np.float32)
        elif agent == "low_avoid_agent":
            return np.array([self.prisoner_x, self.prisoner_y, distance_from_guard],
                            dtype=np.float32)
        else:
            raise ValueError(f"Unknown agent {agent}")

    def reset(self, *, seed=None, options=None):
        self.start_time = 0
        self.spawn_agents()
        self._high_level_action = None

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
        """"""
        terminateds = {agent: False for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        observations = {}

        print(f"Action dict: {action_dict}")

        # 🔹 Step 1: High-Level Agent Decides Which Low-Level Agent Acts
        if "high_level_agent" in action_dict:
            print("🔹 High-level agent is deciding which low-level agent to activate.")
            high_level_decision = action_dict["high_level_agent"]

            # Store high-level decision
            self._high_level_action = high_level_decision

            # Choose which low-level agent will act
            if self._high_level_action == ATTACK_IDX:
                active_low_agent = "low_attack_agent"
            else:
                active_low_agent = "low_avoid_agent"

            # Provide observations for both high-level and newly activated low-level agent
            observations = {
                "high_level_agent": self.observe("high_level_agent"),
                active_low_agent: self.observe(active_low_agent),
            }

            # Small reward for transitioning
            rewards["high_level_agent"] = -0.01
            print(f"🎯 High-level agent chose {active_low_agent}.")
            return observations, rewards, terminateds, truncateds, {}

        # 🔹 Step 2: Low-Level Agent Executes an Action
        elif self._high_level_action is not None:
            # elif any(agent in action_dict for agent in ["low_attack_agent", "low_avoid_agent"]):
            print("🔹 Low-level agent is executing an action.")
            self.start_time += 1

            # Determine which low-level agent is active
            if self._high_level_action == ATTACK_IDX:
                agent = "low_attack_agent"
            else:
                agent = "low_avoid_agent"

            # Validate that the chosen agent has an action
            if agent not in action_dict:
                raise ValueError(
                    f"Expected action for {agent}, but not found in action_dict.")

            # Move the prisoner based on the selected action
            dx, dy = self.get_move(action_dict[agent])
            self.prisoner_x += dx
            self.prisoner_y += dy

            # Update the observation for the low-level agent
            observations[agent] = self.observe(agent)

            # 🔹 Step 3: Check for Termination Conditions
            if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
                rewards["high_level_agent"] = -10.0
                rewards[agent] = -10.0
                terminateds["high_level_agent"] = True
                terminateds[agent] = True
                print("⚠️ Prisoner caught by guard! Penalizing agent.")

            elif self.prisoner_x == self.goal_x and self.prisoner_y == self.goal_y:
                rewards["high_level_agent"] = 10.0
                rewards[agent] = 10.0
                terminateds["high_level_agent"] = True
                terminateds[agent] = True
                print("🏆 Prisoner reached the goal! Rewarding agent.")

            elif self.start_time >= self.time_limit:
                rewards["high_level_agent"] = 0.0
                rewards[agent] = 0.0
                terminateds["high_level_agent"] = True
                terminateds[agent] = True
                print("⏳ Time limit reached. Ending episode.")

            elif self.is_out_of_bounds(self.prisoner_x, self.prisoner_y):
                rewards["high_level_agent"] = -10.0
                rewards[agent] = -10.0
                terminateds["high_level_agent"] = True
                terminateds[agent] = True
                print("🚧 Agent moved out of bounds! Penalizing.")

            self._high_level_action = None

        # # 🔹 Step 4: Signal Termination for RLlib if Needed
        # terminateds["__all__"] = all(terminateds.values())
        return observations, rewards, terminateds, truncateds, {}


class HiearchialEnvV2(MultiAgentEnv):

    def __init__(self,
                 env_config: Dict = None,
                 use_action_mask: bool = False) -> None:
        super().__init__()

        self.agents: List[str] = ["high_level_agent",
                                  "low_attack_agent", "low_avoid_agent"]

        self.possible_agents: List[str] = self.agents[:]

        # observation space for high level agent
        low_obs: np.ndarray = np.array([MIN_X, MIN_Y, 0], dtype=np.float32)
        high_obs: np.ndarray = np.array(
            [MAX_X, MAX_Y, np.sqrt(MAX_X ** 2 + MAX_Y ** 2)], dtype=np.float32)

        low_high_obs: np.ndarray = np.array([0, 0], dtype=np.float32)
        high_high_obs: np.ndarray = np.array(
            [np.sqrt(MAX_X ** 2 + MAX_Y ** 2),
             np.sqrt(MAX_X ** 2 + MAX_Y ** 2)],
            dtype=np.float32)

        self.action_map: Dict[int, int] = {0: (-1, 0),  # Left
                                           1: (1, 0),  # Right
                                           2: (0, -1),  # Down
                                           3: (0, 1)}  # Up

        self.observation_spaces: Dict[str, gym.spaces.Box] = {
            # Attack or Avoid,
            "high_level_agent": gym.spaces.Box(low=low_high_obs, high=high_high_obs,
                                               dtype=np.float32),
            "low_attack_agent": gym.spaces.Box(low=low_obs, high=high_obs,
                                               dtype=np.float32),
            "low_avoid_agent": gym.spaces.Box(low=low_obs, high=high_obs,
                                              dtype=np.float32)
        }

        self.action_spaces: Dict[str, gym.spaces.Discrete] = {
            "high_level_agent": gym.spaces.Discrete(2),  # Attack or Avoid
            "low_attack_agent": gym.spaces.Discrete(4),
            "low_avoid_agent": gym.spaces.Discrete(4)
        }
        self.start_time: int = 0
        self.time_limit: int = 100
        self._high_level_action: Optional[int] = None

        self.spawn_agents()
        self.reset()

    def spawn_agents(self):
        self.prisoner_x: int = 1  # np.random.randint(MIN_X, MAX_X)
        self.prisoner_y: int = 1  # np.random.randint(MIN_Y, MAX_Y)

        self.guard_x: int = np.random.randint(MIN_X, MAX_X)
        self.guard_y: int = np.random.randint(MIN_Y, MAX_Y)

        # choose a goal location taht is not the same as the guard
        self.goal_x: int = np.random.randint(MIN_X, MAX_X)
        self.goal_y: int = np.random.randint(MIN_Y, MAX_Y)

        while self.goal_x == self.guard_x and self.goal_y == self.guard_y:
            self.goal_x = np.random.randint(MIN_X, MAX_X)
            self.goal_y = np.random.randint(MIN_Y, MAX_Y)

    def observe(self, agent: str) -> np.ndarray:
        distance_to_goal: float = np.sqrt(
            (self.prisoner_x - self.goal_x) ** 2 +
            (self.prisoner_y - self.goal_y) ** 2
        )

        distance_from_guard: float = np.sqrt(
            (self.guard_x - self.prisoner_x) ** 2 +
            (self.guard_y - self.prisoner_y) ** 2
        )

        if agent == "high_level_agent":
            return np.array([distance_to_goal, distance_from_guard],
                            dtype=np.float32)
        elif agent == "low_attack_agent":
            return np.array([self.prisoner_x, self.prisoner_y, distance_to_goal],
                            dtype=np.float32)
        elif agent == "low_avoid_agent":
            return np.array([self.prisoner_x, self.prisoner_y, distance_from_guard],
                            dtype=np.float32)
        else:
            raise ValueError(f"Unknown agent {agent}")

    def reset(self, *, seed=None, options=None):
        self.start_time = 0
        self.spawn_agents()
        self._high_level_action = None

        # Return high-level observation.
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations, {}

    def greedy_pursuit_guard(self):
        """
        #TODO: Implement as Heuristic Policy later on
        Heuristic to move the guard towards the prisoner.
        """
        dx, dy = 0, 0
        if self.guard_x < self.prisoner_x:
            dx = 1
        elif self.guard_x > self.prisoner_x:
            dx = -1

        if self.guard_y < self.prisoner_y:
            dy = 1
        elif self.guard_y > self.prisoner_y:
            dy = -1

        self.guard_x += dx
        self.guard_y += dy

        if self.is_out_of_bounds(self.guard_x, self.guard_y):
            # check if it's dx or dy
            if self.guard_x < MIN_X or self.guard_x > MAX_X:
                self.guard_x -= dx
            if self.guard_y < MIN_Y or self.guard_y > MAX_Y:
                self.guard_y -= dy

            self.guard_x -= dx
            self.guard_y -= dy

    def get_move(self, action: int) -> tuple:
        if action not in self.action_map:
            raise ValueError(f"Invalid action {action}")

        return self.action_map[action]

    def is_out_of_bounds(self, x: int, y: int) -> bool:
        return x < MIN_X or x > MAX_X or y < MIN_Y or y > MAX_Y

    def is_caught(self, x: int, y: int) -> bool:
        return x == self.guard_x and y == self.guard_y

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        For HRL to work you need to specify which agent is action,
        the return will need to return which agent will be the next to act.
        """
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}

        rewards = {agent: 0.0 for agent in self.agents}
        infos = {}

        dx, dy = (0, 0)

        if "high_level_agent" in action_dict:
            self._high_level_action = action_dict["high_level_agent"]
            # we're going to choose the next low-level agent to activate
            if self._high_level_action == ATTACK_IDX:
                next_active_agent = "low_attack_agent"
            else:
                next_active_agent = "low_avoid_agent"

            observation = {next_active_agent: self.observe(next_active_agent)}
            rewards["high_level_agent"] = -0.01

            return observation, rewards, terminateds, truncateds, infos
        else:
            if self._high_level_action == ATTACK_IDX:
                agent_policy = "low_attack_agent"
            else:
                agent_policy = "low_avoid_agent"

            self.greedy_pursuit_guard()
            dx, dy = self.get_move(action_dict[agent_policy])
            self.prisoner_x += dx
            self.prisoner_y += dy
            # check if we're out of bounds
            if self.is_out_of_bounds(self.prisoner_x, self.prisoner_y):
                rewards["high_level_agent"] = -10.0
                rewards[agent_policy] = -10.0
                terminateds = {"__all__": True}
                print("🚧 Agent moved out of bounds! Penalizing.")
            elif self.is_caught(self.prisoner_x, self.prisoner_y):
                rewards["high_level_agent"] = -10.0
                rewards[agent_policy] = -10.0
                terminateds = {"__all__": True}
                print("🚧 Agent has been caught! Penalizing.")
            elif self.prisoner_x == self.goal_x and self.prisoner_y == self.goal_y:
                rewards["high_level_agent"] = 10.0
                rewards[agent_policy] = 10.0
                terminateds = {"__all__": True}
                print("🏆 Prisoner reached the goal! Rewarding agent.")
            elif self.start_time >= self.time_limit:
                rewards["high_level_agent"] = 0.0
                rewards[agent_policy] = 0.0
                truncateds = {"__all__": True}
                print("⏳ Time limit reached. Ending episode.")

            # return only the high level agent
            observations = {
                "high_level_agent": self.observe("high_level_agent")}
            # infos = {agent: self.observe(agent) for agent in self.agents}
            infos = {}
            return observations, rewards, terminateds, truncateds, infos


class ActionMaskingHRLEnv(MultiAgentEnv):
    """
    This is a concrete implementation of a multi-agent environment
    with action masking for hierarchical reinforcement learning.
    """

    def __init__(self,
                 env_config: Dict = None) -> None:
        super().__init__()

        self.agents: List[str] = ["high_level_agent",
                                  "low_attack_agent", "low_avoid_agent"]

        self.possible_agents: List[str] = self.agents[:]

        # observation space for high level agent
        low_obs: np.ndarray = np.array(
            [MIN_X, MIN_Y, 0, -MAX_X, -MAX_Y], dtype=np.float32)
        high_obs: np.ndarray = np.array(
            [MAX_X, MAX_Y, np.sqrt(MAX_X ** 2 + MAX_Y ** 2),
             MAX_X, MAX_Y], dtype=np.float32)

        low_high_obs: np.ndarray = np.array([0, 0], dtype=np.float32)
        high_high_obs: np.ndarray = np.array(
            [np.sqrt(MAX_X ** 2 + MAX_Y ** 2),
             np.sqrt(MAX_X ** 2 + MAX_Y ** 2)],
            dtype=np.float32)

        self.action_map: Dict[int, int] = {0: (-1, 0),  # Left
                                           1: (1, 0),  # Right
                                           2: (0, -1),  # Down
                                           3: (0, 1)}  # Up

        original_obs_space = gym.spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32)

        self.action_spaces: Dict[str, gym.spaces.Discrete] = {
            "high_level_agent": gym.spaces.Discrete(2),  # Attack or Avoid
            "low_attack_agent": gym.spaces.Discrete(4),
            "low_avoid_agent": gym.spaces.Discrete(4)
        }

        # self.observation_spaces = {
        #     "high_level_agent": gym.spaces.Box(low=low_high_obs, high=high_high_obs, dtype=np.float32),
        #     "low_attack_agent": gym.spaces.Dict({
        #         "observations": gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
        #         "action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_spaces["low_attack_agent"].n,), dtype=np.float32)
        #     }),
        #     "low_avoid_agent": gym.spaces.Dict({
        #         "observations": gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
        #         "action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_spaces["low_avoid_agent"].n,), dtype=np.float32)
        #     })
        # }

        # Define the per-agent observation spaces:
        agent_obs_spaces = {
            "high_level_agent": gym.spaces.Box(low=low_high_obs, high=high_high_obs, dtype=np.float32),
            "low_attack_agent": gym.spaces.Dict({
                "observations": gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0.0, high=1.0,
                                              shape=(
                                                  self.action_spaces["low_attack_agent"].n,),
                                              dtype=np.float32)
            }),
            "low_avoid_agent": gym.spaces.Dict({
                "observations": gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0.0, high=1.0,
                                              shape=(
                                                  self.action_spaces["low_avoid_agent"].n,),
                                              dtype=np.float32)
            })
        }

        # Wrap the outer dictionary in a gym.spaces.Dict:
        self.observation_spaces = gym.spaces.Dict(agent_obs_spaces)

        self.start_time: int = 0
        self.time_limit: int = 100
        self._high_level_action: Optional[int] = None
        self.spawn_agents()

        self.old_distance_to_goal: float = self.compute_distance(
            self.prisoner_x, self.prisoner_y, self.goal_x, self.goal_y)

        self.old_distance_from_guard: float = self.compute_distance(
            self.prisoner_x, self.prisoner_y, self.guard_x, self.guard_y)

        self.reset()

    def spawn_agents(self):
        self.prisoner_x: int = np.random.randint(MIN_X, MAX_X)
        self.prisoner_y: int = np.random.randint(MIN_Y, MAX_Y)

        self.guard_x: int = np.random.randint(MIN_X, MAX_X)
        self.guard_y: int = np.random.randint(MIN_Y, MAX_Y)

        while self.guard_x == self.prisoner_x and self.guard_y == self.prisoner_y:
            self.guard_x = np.random.randint(MIN_X, MAX_X)
            self.guard_y = np.random.randint(MIN_Y, MAX_Y)

        # choose a goal location taht is not the same as the guard
        self.goal_x: int = np.random.randint(MIN_X, MAX_X)
        self.goal_y: int = np.random.randint(MIN_Y, MAX_Y)

        while self.goal_x == self.guard_x and self.goal_y == self.guard_y:
            self.goal_x = np.random.randint(MIN_X, MAX_X)
            self.goal_y = np.random.randint(MIN_Y, MAX_Y)

    def compute_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def observe(self, agent: str) -> np.ndarray:
        distance_to_goal: float = np.sqrt(
            (self.prisoner_x - self.goal_x) ** 2 +
            (self.prisoner_y - self.goal_y) ** 2
        )

        distance_from_guard: float = np.sqrt(
            (self.guard_x - self.prisoner_x) ** 2 +
            (self.guard_y - self.prisoner_y) ** 2
        )

        # Compute the original observation
        if agent == "high_level_agent":
            distance_to_goal = np.sqrt(
                (self.prisoner_x - self.goal_x) ** 2 + (self.prisoner_y - self.goal_y) ** 2)
            distance_from_guard = np.sqrt(
                (self.guard_x - self.prisoner_x) ** 2 + (self.guard_y - self.prisoner_y) ** 2)
            return np.array([distance_to_goal, distance_from_guard], dtype=np.float32)
        elif agent in ["low_attack_agent", "low_avoid_agent"]:

            dx = self.guard_x - self.prisoner_x
            dy = self.guard_y - self.prisoner_y
            observation = np.array(
                [self.prisoner_x, self.prisoner_y, distance_to_goal,
                 dx, dy], dtype=np.float32)
            action_mask = self._generate_action_mask(agent)
            return {"observations": observation, "action_mask": action_mask}
        else:
            raise ValueError(f"Unknown agent {agent}")

    def _generate_action_mask(self, agent: str) -> np.ndarray:
        valid_actions = np.ones(self.action_spaces[agent].n,
                                dtype=np.float32)
        if self.prisoner_x <= MIN_X:
            valid_actions[0] = 0.0  # Can't move left
        if self.prisoner_x >= MAX_X:
            valid_actions[1] = 0.0  # Can't move right
        if self.prisoner_y <= MIN_Y:
            valid_actions[2] = 0.0  # Can't move down
        if self.prisoner_y >= MAX_Y:
            valid_actions[3] = 0.0  # Can't move up
        return valid_actions

    def reset(self, *, seed=None, options=None):
        self.start_time = 0
        self.spawn_agents()
        self._high_level_action = None
        self.old_distance_to_goal: float = self.compute_distance(
            self.prisoner_x, self.prisoner_y, self.goal_x, self.goal_y)
        self.current_distance_to_goal: float = self.compute_distance(
            self.prisoner_x, self.prisoner_y, self.goal_x, self.goal_y)

        # Return high-level observation.
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations, {}

    def greedy_pursuit_guard(self):
        """
        #TODO: Implement as Heuristic Policy later on
        Heuristic to move the guard towards the prisoner.
        """
        dx, dy = 0, 0
        if self.guard_x < self.prisoner_x:
            dx = 1
        elif self.guard_x > self.prisoner_x:
            dx = -1

        if self.guard_y < self.prisoner_y:
            dy = 1
        elif self.guard_y > self.prisoner_y:
            dy = -1

        self.guard_x += dx
        self.guard_y += dy

        if self.is_out_of_bounds(self.guard_x, self.guard_y):
            # check if it's dx or dy
            if self.guard_x < MIN_X or self.guard_x > MAX_X:
                self.guard_x -= dx
            if self.guard_y < MIN_Y or self.guard_y > MAX_Y:
                self.guard_y -= dy

            self.guard_x -= dx
            self.guard_y -= dy

    def get_move(self, action: int) -> tuple:
        if action not in self.action_map:
            raise ValueError(f"Invalid action {action}")

        return self.action_map[action]

    def is_out_of_bounds(self, x: int, y: int) -> bool:
        return x < MIN_X or x > MAX_X or y < MIN_Y or y > MAX_Y

    def is_caught(self, x: int, y: int) -> bool:
        return x == self.guard_x and y == self.guard_y

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        For HRL to work you need to specify which agent is action,
        the return will need to return which agent will be the next to act.
        """
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}

        rewards = {agent: 0.0 for agent in self.agents}
        infos = {}

        dx, dy = (0, 0)
        terminal_reward: float = 100.0

        if "high_level_agent" in action_dict:
            self._high_level_action = action_dict["high_level_agent"]
            # we're going to choose the next low-level agent to activate
            if self._high_level_action == ATTACK_IDX:
                next_active_agent = "low_attack_agent"
            else:
                next_active_agent = "low_avoid_agent"

            observation = {next_active_agent: self.observe(next_active_agent)}
            rewards["high_level_agent"] = -0.01

            return observation, rewards, terminateds, truncateds, infos
        else:
            if self._high_level_action == ATTACK_IDX:
                agent_policy = "low_attack_agent"
            else:
                agent_policy = "low_avoid_agent"

            self.greedy_pursuit_guard()

            curr_obs = self.observe(agent_policy)
            action_mask = curr_obs["action_mask"]
            action = action_dict[agent_policy]
            # Check if the action is valid
            if action_mask[action] == 0.0:
                raise ValueError(
                    f"Invalid action {action} taken by {agent_policy}")

            dx, dy = self.get_move(action_dict[agent_policy])
            self.prisoner_x += dx
            self.prisoner_y += dy
            # check if we're out of bounds
            if self.is_out_of_bounds(self.prisoner_x, self.prisoner_y):
                rewards["high_level_agent"] = -terminal_reward
                rewards[agent_policy] = -terminal_reward
                terminateds = {"__all__": True}
                print("🚧 Agent moved out of bounds! Penalizing.")
            elif self.is_caught(self.prisoner_x, self.prisoner_y):
                rewards["high_level_agent"] = -terminal_reward
                rewards[agent_policy] = -terminal_reward
                terminateds = {"__all__": True}
                print("🚧 Agent has been caught! Penalizing.")
            elif self.prisoner_x == self.goal_x and self.prisoner_y == self.goal_y:
                rewards["high_level_agent"] = terminal_reward
                rewards[agent_policy] = 1.0
                terminateds = {"__all__": True}
                print("🏆 Prisoner reached the goal! Rewarding agent.")
            elif self.start_time >= self.time_limit:
                rewards["high_level_agent"] = 0.0
                rewards[agent_policy] = 0.0
                truncateds = {"__all__": True}
                terminateds = {"__all__": True}
                print("⏳ Time limit reached. Ending episode.")
            else:
                # so we want a reward funciton based on what policy we are
                # using
                # if the high level agent is using the attack policy
                distance_from_guard = self.compute_distance(
                    self.prisoner_x, self.prisoner_y, self.guard_x, self.guard_y)
                distance_to_goal = self.compute_distance(
                    self.prisoner_x, self.prisoner_y, self.goal_x, self.goal_y)

                delta_distance_to_goal: float = distance_to_goal - \
                    self.old_distance_to_goal
                delta_distance_from_guard: float = distance_from_guard - \
                    self.old_distance_from_guard

                # so if we are increasing distance from guard that is good
                # if we are decreasing distance to goal that is good
                # We are also clipping the reward to be between -1 and 1
                intermediate_reward = np.tanh(
                    -(delta_distance_to_goal) + (delta_distance_from_guard*0.2))

                rewards[agent_policy] = intermediate_reward
                rewards["high_level_agent"] = intermediate_reward
                self.old_distance_from_guard = distance_from_guard
                self.old_distance_to_goal = distance_to_goal

                # return only the high level agent
            observations = {
                "high_level_agent": self.observe("high_level_agent")
            }
            # infos = {agent: self.observe(agent) for agent in self.agents}
            infos = {}
            return observations, rewards, terminateds, truncateds, infos
