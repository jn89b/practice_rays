import unittest
# from .prisoner_guard_env import PrisonerGuardEnv
from practice_rays.prisoner_guard_env import PrisonerGuardEnv
from practice_rays.callbacks import EpisodeReturn, ExampleEnvCallback
# from practice_rays.hiearchial_env import HierarchicalEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.envs.classes.six_room_env import (
    HierarchicalSixRoomEnv,
    SixRoomEnv,
)


class TestGeneratedData(unittest.TestCase):

    def setUp(self):
        self.test_env = PrisonerGuardEnv()
        self.hrl_env = HierarchicalEnv(env_config=None)

    def test_spawn_agents(self):
        self.assertTrue(self.test_env.guard_x >=
                        0 and self.test_env.guard_x <= 10)
        self.assertTrue(self.test_env.guard_y >=
                        0 and self.test_env.guard_y <= 10)

        print("Guard x: ", self.test_env.guard_x)
        print("Guard y: ", self.test_env.guard_y)
        print("Prisoner x: ", self.test_env.prisoner_x)
        print("Prisoner y: ", self.test_env.prisoner_y)

    def test_simple_move(self):
        self.test_env.reset()
        n_steps = 30

        for i in range(n_steps):
            one_action = self.test_env.action_spaces[self.test_env.current_player].sample(
            )
            second_action = self.test_env.action_spaces[self.test_env.current_player].sample(
            )
            actions = {"prisoner": one_action,
                       "guard": second_action}

            observations, rewards, terminateds, _, infos = self.test_env.step(
                actions)
            print("action space", self.test_env.action_spaces)
            print("type of observations: ", type(observations))
            print("type of rewards: ", type(rewards))
            if terminateds['__all__']:
                break

    def test_hrl_move(self):
        self.hrl_env.reset()
        n_steps = 30

        actions = {"high_level_agent": 0}

        for i in range(n_steps):
            print("action: ", actions)

            self.hrl_env.step(actions)

    # def test_train(self):

    #     prisoner_obs = self.test_env.observation_spaces["prisoner"]
    #     prisoner_act = self.test_env.action_spaces["prisoner"]
    #     guard_obs = self.test_env.observation_spaces["guard"]
    #     guard_act = self.test_env.action_spaces["guard"]
    #     config = (
    #         PPOConfig()
    #         # Use your custom class directly
    #         .environment(env=PrisonerGuardEnv)
    #         .multi_agent(
    #             policies={
    #                 "prisoner": (None, prisoner_obs, prisoner_act, None),
    #                 "guard": (None, guard_obs, guard_act, None),
    #             },
    #             # Simple mapping - each agent gets its own policy
    #             policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
    #             # Different hyperparameters for each agent type
    #             algorithm_config_overrides_per_module={
    #                 "prisoner": PPOConfig.overrides(gamma=0.85),
    #                 "guard": PPOConfig.overrides(lr=0.00001),
    #             },
    #         )
    #         .rl_module(
    #             rl_module_spec=MultiRLModuleSpec(rl_module_specs={
    #                 "prisoner": RLModuleSpec(),
    #                 "guard": RLModuleSpec(),
    #             }),
    #         )
    #     )

    #     algo = config.build()
    #     print(algo.train())


if __name__ == '__main__':
    unittest.main()
