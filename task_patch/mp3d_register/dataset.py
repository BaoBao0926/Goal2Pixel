#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from habitat import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation
from omegaconf import DictConfig

from scripts.run_utils.constant import NO_USED_CATEGORIES, C5_categories, C16_categories

from .attributes import MP3DEpisode, ObjectInScene, SceneState

HEIGHT_DIFFERENCE_THRESHOLD = 0.25
EPSILON = 1e-5
ONE_FLOOR_HEIGHT = 1000.0  # in meters
# ONE_FLOOR_HEIGHT = 0.3  # in meters


@registry.register_dataset(name="ObjectNav-MP3D")
class ObjectNavDatasetMP3D(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""

    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[MP3DEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]
    gibson_to_mp3d_category_map: Dict[str, str] = {
        "couch": "sofa",
        "toilet": "toilet",
        "bed": "bed",
        "tv": "tv_monitor",
        "potted plant": "plant",
        "chair": "chair",
    }

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = MP3DEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def experiment_config_check(self, config):
        if self.experiment_mode == "C16":
            # assert config['data_path'].split("/")[-3] == "objectnav_mp3d_35k"
            # assert config['split'] == "train"
            self.keep_goals = C16_categories

        elif self.experiment_mode == "C5":
            # assert config['data_path'].split("/")[-3] == "objectnav_mp3d_35k"
            # assert config['split'] == "train"
            self.keep_goals = C5_categories

        elif self.experiment_mode == "S28":
            # assert config['data_path'].split("/")[-3] == "objectnav_mp3d_35k"
            # assert config['split'] == "train"
            self.keep_goals = C16_categories + C5_categories

        elif self.experiment_mode == "S11":
            assert config["data_path"].split("/")[-3] == "mp3d"
            assert config["split"] == "val"

        elif self.experiment_mode == "debug":
            logger.info("Debug mode: no data filtering applied.")

        else:
            raise ValueError(
                f"Unknown experiment mode: {self.experiment_mode}. "
                "Supported modes are: C16, C5, S28, S11, debug."
            )

    def filter(self, episode_data):
        # Filter 1: Skip by object category early
        if episode_data.get("object_category") in self.filtered_goals:
            return
        if self.experiment_mode in ["C16", "C5"]:
            if episode_data.get("object_category") not in self.keep_goals:
                return

        # Filter 2: Skip by replay length
        replay = episode_data.get("reference_replay")
        if not replay and self.experiment_mode in ["C16", "C5", "S28"]:
            raise ValueError("Episode reference replay is None, but it should not be.")

        if not self.experiment_mode == "S11":
            if len(replay) > self.max_replay_steps:
                return

            # Filter 3: Remove LOOK_UP and LOOK_DOWN actions from replay
            replay = [
                step
                for step in replay
                if step["action"] not in ["LOOK_UP", "LOOK_DOWN"]
            ]

            # Filter 4: Skip by height change in replay steps
            skip_episode = False
            initial_height = replay[0]["agent_state"]["position"][1]
            invalid_step_indices = []

            for i in range(2, len(replay) - 1):
                curr_step = replay[i]
                curr_action = curr_step["action"]
                curr_state = curr_step["agent_state"]
                prev_state = replay[i - 1]["agent_state"]

                # height check
                current_height = curr_state["position"][1]
                if abs(current_height - initial_height) > HEIGHT_DIFFERENCE_THRESHOLD:
                    skip_episode = True
                    break

                # behavior check (does the agent move?)
                pos_moved = np.linalg.norm(
                    np.array(curr_state["position"]) - np.array(prev_state["position"])
                )
                rot_changed = np.linalg.norm(
                    np.array(curr_state["rotation"]) - np.array(prev_state["rotation"])
                )

                if curr_action == "MOVE_FORWARD":
                    if pos_moved == 0.0 and rot_changed != 0.0:
                        raise ValueError(
                            f"Invalid action in replay: {curr_action}. "
                            "Expected agent to move forward, but rotation changed without position change."
                        )
                elif curr_action == "TURN_LEFT" or curr_action == "TURN_RIGHT":
                    if rot_changed == 0.0 and pos_moved != 0.0:
                        raise ValueError(
                            f"Invalid action in replay: {curr_action}. "
                            "Expected agent to turn, but position changed without rotation change."
                        )
                else:
                    raise ValueError(
                        f"Unknown action in replay: {curr_action}. "
                        "Expected MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT."
                    )
                if pos_moved < EPSILON and rot_changed < EPSILON:
                    # logging.warning(
                    #     f"Episode {episode_data['episode_id']} step {i} has no movement or rotation change. "
                    # )
                    # skip this step ( record the step index in the episode)
                    invalid_step_indices.append(i)

            # Remove invalid steps from replay (these are steps where the agent did not move or rotate)
            replay = [
                step
                for idx, step in enumerate(replay)
                if idx not in invalid_step_indices
            ]

            if skip_episode:
                return

            episode_data["reference_replay"] = replay

        return episode_data

    def __init__(
        self, config: Optional["DictConfig"] = None, exp_mode: str = "C16"
    ) -> None:
        self.goals_by_category = {}
        self.max_replay_steps = config.max_replay_steps
        self.experiment_mode = exp_mode

        # check for each experiment mode, is the data loading correct?
        self.experiment_config_check(config)

        # 28 object categories in MP3D --> to 21 task categories first
        self.filtered_goals = NO_USED_CATEGORIES

        # load from traj dataset and scene dir PointNavDatasetV1
        # from_json will be called
        super().__init__(config)

        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:

        # load the json string contains
        # category_to_mp3d_category_id and category_to_task_category_id
        deserialized = json.loads(json_str)

        # Load mappings
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        # If no episodes are present, return early
        if len(deserialized["episodes"]) == 0:
            return

        # Deduplicate goals if necessary
        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        # Deserialize all goals_by_category first
        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for episode_data in deserialized["episodes"]:
            if "_shortest_path_cache" in episode_data:
                del episode_data["_shortest_path_cache"]

            if "gibson" in episode_data["scene_id"]:
                episode["scene_id"] = "gibson_semantic/{}".format(
                    episode_data["scene_id"].split("/")[-1]
                )

            episode_data = self.filter(episode_data)
            if not episode_data:
                continue

            # Now instantiate episode object
            episode = MP3DEpisode(**episode_data)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if not episode.is_thda:
                ## obtain which goal the agent is actually reaching in the hdt

                potential_goals = self.goals_by_category.get(episode.goals_key, [])
                gt_stop_state = np.array(
                    episode.reference_replay[-2]["agent_state"]["position"]
                )

                matched_goal = None
                min_dist = float("inf")  # start with infinity

                for goal in potential_goals:
                    gpos = np.array(goal.position)

                    # check height first
                    if abs(gpos[1] - gt_stop_state[1]) < ONE_FLOOR_HEIGHT:
                        dist = np.linalg.norm(gpos[[0, 2]] - gt_stop_state[[0, 2]])
                        if dist < min_dist:  # keep the closest so far
                            matched_goal = goal
                            min_dist = dist

                episode.gt_reach_goal = (matched_goal, min_dist)

                ##

                episode.goals = self.goals_by_category[episode.goals_key]
                if episode.scene_dataset == "gibson":
                    episode.object_category = self.gibson_to_mp3d_category_map[
                        episode.object_category
                    ]
            else:
                goals = []
                for g in episode.goals:
                    g = ObjectGoal(**g)
                    for vidx, view in enumerate(g.view_points):
                        view_location = ObjectViewLocation(**view)  # type: ignore
                        view_location.agent_state = AgentState(
                            **view_location.agent_state
                        )  # type: ignore
                        g.view_points[vidx] = view_location
                    goals.append(g)
                episode.goals = goals

                objects = [ObjectInScene(**o) for o in episode.scene_state["objects"]]
                scene_state = [SceneState(objects=objects).__dict__]
                episode.scene_state = scene_state

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            # Passed all filters — keep this episode
            self.episodes.append(episode)  # type: ignore


import gzip
from typing import List, Optional

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode


@registry.register_dataset(name="R2RVLN-v2")
class VLNDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            episode = VLNEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            agent_start_pos = episode.start_position

            valid_goals = []
            for goal in episode.goals:
                gpos = np.array(goal["position"])
                # check height first
                if abs(gpos[1] - agent_start_pos[1]) < ONE_FLOOR_HEIGHT:
                    valid_goals.append(NavigationGoal(**goal))

            # for different height filter
            one_floor_flag = True
            # for re_waypoint in episode.reference_path:
            #     re_waypoint_pos = np.array(re_waypoint)
            #     # check height first
            #     if abs(re_waypoint_pos[1] - agent_start_pos[1]) >= ONE_FLOOR_HEIGHT:
            #         one_floor_flag = False

            # if episode.scene_id not in ['../../VLN_dataset/data/scene_datasets/mp3d/ZMojNkEp431/ZMojNkEp431.glb']:
            #     continue

            # Only append if exactly one valid goal remains and all reference path waypoints are on the same floor
            if len(valid_goals) == 1 and one_floor_flag:
                episode.goals = valid_goals
                self.episodes.append(episode)


ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
ALL_EPISODES_MASK = "*"

from habitat.core.dataset import ALL_SCENES_MASK
from habitat.tasks.vln.vln import ExtendedInstructionData, VLNExtendedEpisode


@registry.register_dataset(name="RxRVLN-v2")
class RxRVLNCEDatasetV1(Dataset):
    """Loads the RxR VLN-CE Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                config.data_path.format(split=config.split, role=role), "rt"
            ) as f:
                self.from_json(f.read(), scenes_dir=config.scenes_dir)

        if ALL_SCENES_MASK not in config.content_scenes:
            scenes_to_load = set(config.content_scenes)
            self.episodes = [
                e
                for e in self.episodes
                if self.scene_from_scene_path(e.scene_id) in scenes_to_load
            ]

        if ALL_LANGUAGES_MASK not in config.LANGUAGES:
            languages_to_load = set(config.LANGUAGES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]
            pass

        if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:

        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(**episode.instruction)
            episode.instruction.split = self.config.split

            # specify some scene to run
            # if episode.scene_id not in ['../../VLN_dataset/data/scene_datasets/mp3d/S9hNv5qa7GM/S9hNv5qa7GM.glb']:
            #     continue

            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config: DictConfig) -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted({cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes})

    @classmethod
    def extract_roles_from_config(cls, config: DictConfig) -> List[str]:
        if ALL_ROLES_MASK in config.ROLES:
            return cls.annotation_roles
        assert set(config.ROLES).issubset(set(cls.annotation_roles))
        return config.ROLES

    @classmethod
    def check_config_paths_exist(cls, config: DictConfig) -> bool:
        return all(
            os.path.exists(config.data_path.format(split=config.split, role=role))
            for role in cls.extract_roles_from_config(config)
        ) and os.path.exists(config.scenes_dir)

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        """Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> str:
        return episode.instruction.language
