from habitat.tasks.nav.nav import NavigationEpisode
import attr
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation


"""
from ovon repo
"""

@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state."""

    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action."""

    action: str = attr.ib(default=None, validator=not_none_validator)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)


@attr.s(auto_attribs=True)
class OVONObjectViewLocation(ObjectViewLocation):
    r"""OVONObjectViewLocation

    Args:
        raidus: radius of the circle
    """

    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode
    :param object_category: Category of the obect
    """

    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "hm3d"
    scene_dataset_config: Optional[str] = ""
    additional_obj_config_paths: Optional[List] = []
    attempts: Optional[int] = 1

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"

@attr.s(auto_attribs=True, kw_only=True)
class OVONEpisode(ObjectGoalNavEpisode):
    r"""OVON Episode

    :param children_object_categories: Category of the object
    """

    children_object_categories: Optional[List[str]] = []