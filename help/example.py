from typing import Dict, Optional
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from ray.rllib.agents.pg import PGConfig

from pathlib import Path
import imageio
import numpy as np


class MyCallback(DefaultCallbacks):

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        # create an empty list for keeping the frames
        episode.hist_data["frames"] = []

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:

        if worker.policy_config["in_evaluation"]:
            # call your custom env logging procedure here
            img = np.random.randint(0, 256, size=(64, 64, 3)).astype('uint8')
            episode.hist_data["frames"].append(img)

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:

        if worker.policy_config["in_evaluation"]:
            log_dir = Path(worker.io_context.log_dir)
            frames = episode.hist_data["frames"]
            imageio.mimsave(
                log_dir / f'{episode.env_id}_{episode.env_id}.gif',
                frames
            )


if __name__ == '__main__':
    config = (
        PGConfig()
        .framework('torch')
        .callbacks(callbacks_class=MyCallback)
        .environment(env='CartPole-v0')
        .evaluation(evaluation_interval=1)
    )

    algo = config.build()
    algo.train()