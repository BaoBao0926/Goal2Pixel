from dataclasses import dataclass, field
from typing import Optional

from habitat.config.default_structured_configs import (HabitatConfig,
                                                         FogOfWarConfig,
                                                       DatasetConfig,
                                                       LabSensorConfig,
                                                       MeasurementConfig,
                                                       EnvironmentConfig,)


from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

cs = ConfigStore.instance()


##########################################################################
# Sensors
##########################################################################
@dataclass
class MP3DObjectGoalSensorConfig(LabSensorConfig):
    type: str = "MP3DObjectGoalSensor"
    goal_spec: str = "TASK_CATEGORY_ID"
    goal_spec_max_val: int = 50

@dataclass
class MP3DDemonstrationSensorConfig(LabSensorConfig):
    type: str = "MP3DDemonstrationSensor"

@dataclass
class MP3DInflectionWeightSensorConfig(LabSensorConfig):
    type: str = "MP3DInflectionWeightSensor"
    inflection_coef: float = 0.5
##########################################################################
# Dataset
##########################################################################
@dataclass
class MP3DDatasetConfig(DatasetConfig):
    max_replay_steps: int = 250
    ROLES: list = field(default_factory=lambda: ["guide"]) # ["*", "guide", "follower"]
    LANGUAGES: list = field(default_factory=lambda: ["*"]) # ["*", "te-IN", "hi-IN", "en-US", "en-IN"]
    EPISODES_ALLOWED: list = field(default_factory=lambda: ["*"]) # list of episode ids allowed to load, empty means all episodes
    

##########################################################################
# Measures
##########################################################################
@dataclass
class OracleSPLMeasurementConfig(MeasurementConfig):
    type: str = "OracleSPL"


@dataclass
class OracleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OracleSuccess"
    success_distance: float = 3.0


@dataclass
class OracleNavigationErrorMeasurementConfig(MeasurementConfig):
    type: str = "OracleNavigationError"


@dataclass
class PathLengthMeasurementConfig(MeasurementConfig):
    type: str = "PathLength"

@dataclass
class StepsTakenMeasurementConfig(MeasurementConfig):
    type: str = "StepsTaken"

@dataclass
class NDTWMeasurementConfig(MeasurementConfig):
    type: str = "NDTW"
    split: str = "val_seen"
    fdtw: bool = True
    gt_path: str = "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz"
    success_distance: float = 3.0

@dataclass
class SDTWMeasurementConfig(MeasurementConfig):
    type: str = "SDTW"
@dataclass
class TopDownMapVLNCEMeasurementConfig(MeasurementConfig):
    type: str = "TopDownMapVLNCE"
    max_episode_steps: int = (
        EnvironmentConfig().max_episode_steps
    )  # TODO : Use OmegaConf II()
    map_padding: int = 3
    map_resolution: int = 1024
    draw_source_and_target: bool = True
    draw_reference_path: bool = True
    draw_fixed_waypoints: bool = True
    draw_mp3d_agent_path: bool = True
    graphs_file: str = "data/connectivity_graphs.pkl"
    draw_source: bool = True
    draw_border: bool = True
    draw_shortest_path: bool = True
    draw_view_points: bool = True
    draw_goal_positions: bool = True
    # axes aligned bounding boxes
    draw_goal_aabbs: bool = True
    fog_of_war: FogOfWarConfig = FogOfWarConfig()



# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------

cs.store(
    package="habitat.dataset",
    group="habitat/dataset",
    name="dataset_config_schema",
    node=MP3DDatasetConfig,
)

cs.store(
    package="habitat.task.lab_sensors.mp3d_objectgoal_sensor",
    group="habitat/task/lab_sensors",
    name="mp3d_objectgoal_sensor",
    node=MP3DObjectGoalSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.mp3d_demonstration_sensor",
    group="habitat/task/lab_sensors",
    name="mp3d_demonstration_sensor",
    node=MP3DDemonstrationSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.mp3d_inflection_weight_sensor",
    group="habitat/task/lab_sensors",
    name="mp3d_inflection_weight_sensor",
    node=MP3DInflectionWeightSensorConfig,
)

cs.store(
    package="habitat.task.measurements.oracle_spl",
    group="habitat/task/measurements",
    name="oracle_spl",
    node=OracleSPLMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.oracle_success",
    group="habitat/task/measurements",
    name="oracle_success",
    node=OracleSuccessMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.oracle_navigation_error",
    group="habitat/task/measurements",
    name="oracle_navigation_error",
    node=OracleNavigationErrorMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.ndtw",
    group="habitat/task/measurements",
    name="ndtw",
    node=NDTWMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.sdtw",
    group="habitat/task/measurements",
    name="sdtw",
    node=SDTWMeasurementConfig,
)   

cs.store(
    package="habitat.task.measurements.path_length",
    group="habitat/task/measurements",
    name="path_length",
    node=PathLengthMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.steps_taken",
    group="habitat/task/measurements",
    name="steps_taken",
    node=StepsTakenMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.top_down_map_vlnce",
    group="habitat/task/measurements",
    name="top_down_map_vlnce",
    node=TopDownMapVLNCEMeasurementConfig,
)



class HabitatConfigPlugin_mp3d(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://config/dataset/",
        )
        search_path.append(
            provider="habitat",
            path="pkg://config/tasks/",
        )