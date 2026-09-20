import attr
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask
)
from habitat.core.simulator import AgentState
"""
from habitat-imitation-baselines repo
"""

@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state.
    """
    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action.
    """
    action: str = attr.ib(default=None, validator=not_none_validator)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectInScene:
    object_id: int = attr.ib(default=None, validator=not_none_validator)
    semantic_category_id: int = attr.ib(default=None)
    object_template: str = attr.ib(default=None, validator=not_none_validator)
    scale: float = attr.ib(default=None)
    position: List[float] = attr.ib(default=None)
    rotation: List[float] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class SceneState:
    objects: List[ObjectInScene] = attr.ib(default=None)



@attr.s(auto_attribs=True, kw_only=True)
class MP3DEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None

    # new attributes for MP3D-HD..?
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state: Optional[List[SceneState]] = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "mp3d"

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"




