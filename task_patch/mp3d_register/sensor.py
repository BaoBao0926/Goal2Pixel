from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from gym import spaces
from habitat import logger
import numpy as np
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.object_nav_task import ObjectGoal
from omegaconf import DictConfig
from .dataset import ObjectNavDatasetMP3D
from .attributes import (AgentStateSpec,ReplayActionSpec, MP3DEpisode,
                         ObjectInScene, SceneState
                         )
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.simulator import Observations

@registry.register_sensor
class MP3DObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "mp3d_objectgoal_sensor"

    def __init__(
            self,
            sim,
            config: "DictConfig",
            dataset: "ObjectNavDatasetMP3D",
            *args: Any,
            **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.goal_spec_max_val - 1
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
            self,
            observations,
            *args: Any,
            episode: MP3DEpisode,
            **kwargs: Any,
    ) -> Optional[int]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.goal_spec == "object_id":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor."
            )


@registry.register_sensor
class MP3DDemonstrationSensor(Sensor):

    cls_uuid: str = "mp3d_demonstration_sensor"

    def __init__(self, **kwargs):
        # obtain the next action from reference replay
        self.observation_space = spaces.Discrete(1)
        self.prev_action = 0
        self.timestep = 0
        self.uuid = self._get_uuid(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation(
            self,
            observations,
            episode: MP3DEpisode,
            task: EmbodiedTask,
            **kwargs
    ):

        # Fetch next action as observation
        if not task.is_episode_active:  # reset
            self.timestep = 1

        if self.timestep < len(episode.reference_replay):
            action_name = episode.reference_replay[self.timestep]['action'].lower()
            action_to_index = {action: idx for idx, action in enumerate(task.actions.keys())}
            action = action_to_index[action_name]
        else:
            action = 0

        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor
class MP3DInflectionWeightSensor(Sensor):

    cls_uuid: str = "mp3d_inflection_weight_sensor"

    def __init__(self, config: "DictConfig", **kwargs):

        self.observation_space = spaces.Discrete(1)
        self.uuid = self._get_uuid(**kwargs)
        self._config = config
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation(
            self,
            observations,
            episode: MP3DEpisode,
            task: EmbodiedTask,
            **kwargs
    ):

        # Fetch next action as observation
        if not task.is_episode_active:  # reset
            self.timestep = 0

        inflection_weight = 1.0
        if self.timestep == 0:
            inflection_weight = 1.0
        elif self.timestep >= len(episode.reference_replay):
            inflection_weight = 1.0
        elif episode.reference_replay[self.timestep - 1]['action'] != episode.reference_replay[self.timestep]['action']:
            inflection_weight = self._config.inflection_coef

        self.timestep += 1
        return inflection_weight

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

#
# @registry.register_sensor(name="GlobalGPSSensor")
# class GlobalGPSSensor(Sensor):
#     """Current agent location in global coordinate frame"""
#
#     cls_uuid: str = "globalgps"
#
#     def __init__(
#         self, *args: Any, sim, config: "DictConfig", **kwargs: Any
#     ):
#         self._sim = sim
#         self._dimensionality = config.dimensionality
#         super().__init__(config=config)
#
#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         return self.cls_uuid
#
#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return SensorTypes.position
#
#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         return spaces.Box(
#             low=np.finfo(np.float).min,
#             high=np.finfo(np.float).max,
#             shape=(self._dimensionality,),
#             dtype=np.float,
#         )
#
#     def get_observation(self, *args: Any, **kwargs: Any):
#         agent_position = self._sim.get_agent_state().position
#         if self._dimensionality == 2:
#             agent_position = np.array([agent_position[0], agent_position[2]])
#         return agent_position.astype(np.float32)
#
#
# @registry.register_sensor
# class VLNOracleProgressSensor(Sensor):
#     """Relative progress towards goal"""
#
#     cls_uuid: str = "progress"
#
#     def __init__(
#         self, sim, config: "DictConfig", **kwargs: Any
#     ) -> None:
#         self._sim = sim
#         super().__init__(config=config)
#
#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid
#
#     def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
#         return SensorTypes.MEASUREMENT
#
#     def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
#         return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)
#
#     def get_observation(self, *args: Any, episode, **kwargs: Any) -> float:
#         distance_to_target = self._sim.geodesic_distance(
#             self._sim.get_agent_state().position.tolist(),
#             episode.goals[0].position,
#         )
#
#         # just in case the agent ends up somewhere it shouldn't
#         if not np.isfinite(distance_to_target):
#             return np.array([0.0])
#
#         distance_from_start = episode.info["geodesic_distance"]
#         return np.array(
#             [(distance_from_start - distance_to_target) / distance_from_start]
#         )
#
#
# @registry.register_sensor
# class AngleFeaturesSensor(Sensor):
#     """Returns a fixed array of features describing relative camera poses based
#     on https://arxiv.org/abs/1806.02724. This encodes heading angles but
#     assumes a single elevation angle.
#     """
#
#     cls_uuid: str = "angle_features"
#
#     def __init__(self, *args: Any, config: "DictConfig", **kwargs: Any) -> None:
#         self.cameras = config.CAMERA_NUM
#         super().__init__(config)
#         orient = [np.pi * 2 / self.cameras * i for i in range(self.cameras)]
#         self.angle_features = np.stack(
#             [np.array([np.sin(o), np.cos(o), 0.0, 1.0]) for o in orient]
#         )
#
#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid
#
#     def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
#         return SensorTypes.HEADING
#
#     def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
#         return spaces.Box(
#             low=-1.0,
#             high=1.0,
#             shape=(self.cameras, 4),
#             dtype=np.float,
#         )
#
#     def get_observation(self, *args: Any, **kwargs: Any) -> ndarray:
#         return deepcopy(self.angle_features)
#
#
# @registry.register_sensor
# class ShortestPathSensor(Sensor):
#     """Provides the next action to follow the shortest path to the goal."""
#
#     cls_uuid: str = "shortest_path_sensor"
#
#     def __init__(
#         self, *args: Any, sim, config: "DictConfig", **kwargs: Any
#     ):
#         super().__init__(config=config)
#         cls = ShortestPathFollower
#         if config.USE_ORIGINAL_FOLLOWER:
#             cls = ShortestPathFollowerCompat
#         self.follower = cls(sim, config.GOAL_RADIUS, return_one_hot=False)
#
#     def _get_uuid(self, *args: Any, **kwargs: Any):
#         return self.cls_uuid
#
#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return SensorTypes.TACTILE
#
#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)
#
#     def get_observation(self, *args: Any, episode, **kwargs: Any):
#         best_action = self.follower.get_next_action(episode.goals[0].position)
#         if best_action is None:
#             best_action = HabitatSimActions.STOP
#         return np.array([best_action])
#
#
# @registry.register_sensor
# class RxRInstructionSensor(Sensor):
#     """Loads pre-computed intruction features from disk in the baseline RxR
#     BERT file format.
#     https://github.com/google-research-datasets/RxR/tree/7a6b87ba07959f5176aa336192a8c5dc85ca1b8e#downloading-bert-text-features
#     """
#
#     cls_uuid: str = "rxr_instruction"
#
#     def __init__(self, *args: Any, config: "DictConfig", **kwargs: Any):
#         self.features_path = config.features_path
#         super().__init__(config=config)
#
#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid
#
#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return SensorTypes.MEASUREMENT
#
#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         return spaces.Box(
#             low=np.finfo(np.float).min,
#             high=np.finfo(np.float).max,
#             shape=(512, 768),
#             dtype=np.float,
#         )
#
#     def get_observation(
#         self, *args: Any, episode: VLNExtendedEpisode, **kwargs
#     ):
#         features = np.load(
#             self.features_path.format(
#                 split=episode.instruction.split,
#                 id=int(episode.instruction.instruction_id),
#                 lang=episode.instruction.language.split("-")[0],
#             )
#         )
#         feats = np.zeros((512, 768), dtype=np.float32)
#         s = features["features"].shape
#         feats[: s[0], : s[1]] = features["features"]
#         return feats
#
