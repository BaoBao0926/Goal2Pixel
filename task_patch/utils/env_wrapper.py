from typing import Union, Tuple, Dict

from task_patch.utils.filter_traj import (filter_episodes_by_diff_floor_obj,
                                          filter_episodes_by_nav_area,
                                          filter_episodes_by_geodesic_distance,
                                          filter_episodes_by_reply_length,
                                          filter_no_goal,
                                          filter_episodes_by_reply_action,
                                          filter_objectgoal,
                                          filter_seen16_object_goal,
                                          filter_unseen5_object_goal,
                                          filter_fail_HD
                                          )
from habitat.core.environments import RLTaskEnv, RLTaskEnvObsType
from habitat import logger

class SimpleRLEnv(RLTaskEnv):

    def get_info(self, observations):
        metrics = self.habitat_env.get_metrics()
        past_limit = self.habitat_env._past_limit()
        stop_called = self.habitat_env.task.is_stop_called
        elasp_step = self.habitat_env._elapsed_steps
        metrics["past_limit"] = past_limit
        metrics["stop_called"] = stop_called
        metrics["elasp_step"] = elasp_step

        return metrics

class CleanEnv(SimpleRLEnv):
    def __init__(self, config, dataset):
        # Initialize the base environment
        super().__init__(config=config, dataset=dataset)

        logger.info(f"Number of episodes before filtering: {len(self.habitat_env.episodes)}")

        self = filter_episodes_by_nav_area(self)

        logger.info(f"Number of episodes after filtering: {len(self.habitat_env.episodes)}")

        pass


class FilteredEnv(SimpleRLEnv):
    def __init__(self, config, dataset):
        # Initialize the base environment
        super().__init__(config=config, dataset=dataset)

        logger.info(f"Number of episodes before filtering: {len(self.habitat_env.episodes)}")

        """
        Iterate over all episodes, applying several checks to determine if each episode is valid.

        Checks:
        - The position is not on a small island of navigable space (ISLAND_RADIUS_LIMIT).
            [the starting position of the agent is in a sufficiently large area of navigable space]
            [instead of being trapped in a very small, isolated navigable region (or "island") within the environment]

        - Geodesic distance (i.e., shortest path considering obstacles) to goals must be valid and within a certain range (1.0 - 30.0).
            [the agent's starting position is neither too close nor too far from a goal]

        - The episode must have a valid navigable path (sim.pathfinder.find_path()).
            [exists a feasible path that the agent can take from its starting point to its designated goal in the given environment]

        - Elevation (height) change between points must be below a threshold (0.25). -- same floor
        """

        # Apply filtering to episodes
        self.filter_invalid_episodes()

        # Apply filtering to remove failed episodes based on human demo success
        self.filter_failed_episodes()

        # Reset the episode iterator after filtering
        self.reset_episode_iterator()

    def filter_failed_episodes(self):
        """
        Filters out failed episodes based on human demonstration success.
        """
        self = filter_fail_HD(self)

    def filter_invalid_episodes(self):
        """
        Filters out invalid episodes using a series of custom filtering functions.
        The sequence of checks is as follows:
        - Filter episodes by reply length.
        - Filter episodes where the height change between points exceeds a threshold.
        - Filter episodes where the starting position is on a small navigable island.
        - Filter episodes based on geodesic distance and presence of a valid path.
        - Filter episodes with no valid goals.
        """
        self = filter_no_goal(self) # output no filtered
        # self = filter_episodes_by_diff_floor_obj(self)
        # self = filter_episodes_by_nav_area(self) # have problem
        # self = filter_episodes_by_geodesic_distance(self)  # have problem
        self = filter_episodes_by_reply_length(self)
        self = filter_episodes_by_reply_action(self) # output no filtered

        self = filter_objectgoal(self)

        # self = filter_seen16_object_goal(self)
        # self = filter_unseen5_object_goal(self)


    def reset_episode_iterator(self):
        """
        Resets the episode iterator after filtering, ensuring that subsequent episodes
        are selected from the updated set.
        """
        self.habitat_env._current_episode = None
        self.habitat_env._episode_iterator = None
        self.habitat_env._episode_from_iter_on_reset = True
        self.habitat_env._episode_force_changed = False

        # If the dataset is available, set up the episode iterator
        if self.habitat_env._dataset:
            assert (
                len(self.habitat_env._dataset.episodes) > 0
            ), "Dataset should have a non-empty episode list"

            self.habitat_env._setup_episode_iterator()
            self.habitat_env.current_episode = next(self.habitat_env.episode_iterator)

            # Update the number of episodes
            self.number_of_episodes = len(self.habitat_env.episodes)

import math

class NanCheckedEnv(FilteredEnv):
    def step(self, *args, **kwargs):
        observations, reward, done, info = super().step(*args, **kwargs)

        if math.isnan(reward):
            breakpoint()
            breakpoint()
            breakpoint()
        if math.isnan(info["spl"]):
            breakpoint()

        return observations, reward, done, info