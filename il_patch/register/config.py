from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_store import ConfigStore
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from habitat_baselines.config.default_structured_configs import (HabitatBaselinesConfig,
HabitatBaselinesILConfig,
                                                                 ResizeShortestEdgeConfig,
                                                                 CenterCropperConfig,
HabitatBaselinesBaseConfig
                                                                 )

@dataclass
class HabitatBaselinesILConfig_MP3D(HabitatBaselinesILConfig):
    ckpt_no: Optional[int] = None
    results_dir: str = "results"


cs = ConfigStore.instance()

cs.store(
    group="habitat_baselines",
    name="habitat_baselines_il_config_mp3d",
    node=HabitatBaselinesILConfig_MP3D,
)

cs.store(
    group="habitat_baselines/il/policy/obs_transforms",
    name="center_cropper_base",
    node=CenterCropperConfig,
)
cs.store(
    group="habitat_baselines/il/policy/obs_transforms",
    name="resize_shortest_edge_base",
    node=ResizeShortestEdgeConfig,
)

class HabitatBaselinesConfigPlugin_BC(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://habitat_baselines/config/",
        )