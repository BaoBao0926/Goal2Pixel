import gym
from typing import Optional, Tuple, Dict
import numpy as np
import quaternion
from omegaconf import DictConfig
from gym.core import ActType, ObsType
import torch


import habitat
from habitat import Dataset
from habitat.core.environments import GymHabitatEnv
from habitat import logger

from scripts.run_utils.constant import category_to_mp3d_category_id, category_to_task_category_id
import scripts.run_utils.pose_utils as pu



class AnnotatedGymHabitatEnv(GymHabitatEnv):
    def __init__(
        self,
        config: "DictConfig",
        dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.episode_no = -1

        # Get camera height from config
        # For VLN tasks, camera is typically at agent's eye level (same as agent height)
        self.camera_height = config.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1]


        # Episode Dataset info
        self.goal_name = None

        # Episode tracking info
        self.timestep = None # number of steps taken in the current episode
        self.stopped = None # whether the agent has issued a stop action
        self.path_length = None
        self.trajectory_states = []
        self.info = {}

        # pose tracking info
        self.world_last_agent_pose = None
        self.world_curr_agent_pose = None

        super().__init__(config, dataset)

        # core.env.Env
        self.habitat_env = self.env.env.habitat_env

        possible_action_env = self.habitat_env.task.actions.keys()
        logger.info("Possible actions in the environment: {}".format(possible_action_env))


    def reset(self, config_mapping, return_info: bool = False, **kwargs):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        self.episode_no += 1

        # Episode tracking info - Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.world_last_agent_pose = None
        self.world_curr_agent_pose = None
        self.trajectory_states = []
        self.info = {}

        obs, info = self.env.reset(return_info=return_info)
        self.info.update(info)

        self.current_episode = self.habitat_env.current_episode
        self.scene_id = self.current_episode.scene_id.split("/")[-1][:-4]
        self.ep_id = self.current_episode.episode_id
        if hasattr(self.current_episode, "object_category"):
            self.object_goal = self.current_episode.object_category
            self.task_goal_idx = category_to_task_category_id[self.object_goal] # constant.py
            self.mp40_goal_idx = category_to_mp3d_category_id[self.object_goal] # constant.py

        # 1_5_432
        if hasattr(self.current_episode, "gt_reach_goal"):
            self.gt_goal_object = self.current_episode.gt_reach_goal[0].object_name
            self.gt_goal_position = self.current_episode.gt_reach_goal[0].position

        # upstair or downstair check
        self.world_curr_agent_pose = self.get_agent_pose()
        self.start_height = self.world_curr_agent_pose['position'][1]
        # Agent height for point cloud transform: camera height relative to agent base
        # Camera is ALWAYS 0.88m above agent's feet/wheels, regardless of world Y
        # Ground is always ~0.88m below camera, so transform puts ground at Z≈0
        self.agent_height = self.camera_height

        self.start_position = self.world_curr_agent_pose['position']
        self.start_rotation = self.world_curr_agent_pose['rotation']
        
        # Initialize last agent pose to current location
        self.world_last_agent_pose = self.get_agent_location()

        # Set info
        if hasattr(self.current_episode, "gt_reach_goal"):
            self.info['object_goal'] = self.object_goal
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        # self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['goal_name'] = self.goal_name
        self.info['agent_height'] = self.agent_height
        self.info['episode_no'] = self.episode_no

        # logger.info(f"Episode: {self.episode_no}, Search for: {self.object_goal}")

        # semantic annotation
        # obtain mapping from instance id to semantic label id
        scene = self.habitat_env.sim.semantic_annotations()
        # for obj in scene.objects:
        #     # region_id
        #     region_id = obj.id.split("_")[-2]
        #     object_id = int(obj.id.split("_")[-1])
        #     obj_category_index = obj.category.index()
        #     obj_category_name = obj.category.name()
        #
        #     a = obj.region.category.index()
        #     b = obj.region.category.name()
        #
        #     pass


        instance_id_to_obj_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        instance_id_to_obj_name = {int(obj.id.split("_")[-1]): obj.category.name() for obj in scene.objects}

        # instance_id_to_region_id = {int(obj.id.split("_")[-1]): obj.region.category.index() for obj in scene.objects}
        # instance_id_to_region_name = {int(obj.id.split("_")[-1]): obj.region.category.name() for obj in scene.objects}

        self.mapping_id = np.array([instance_id_to_obj_id[i] for i in range(len(instance_id_to_obj_id))])
        self.mapping_name = np.array([instance_id_to_obj_name[i] for i in range(len(instance_id_to_obj_name))])
        # self.mapping_region_id = np.array([instance_id_to_region_id[i] for i in range(len(instance_id_to_region_id))])
        # self.mapping_region_name = np.array([instance_id_to_region_name[i] for i in range(len(instance_id_to_region_name))])

        if 'semantic' in obs:
            obs['semantic_id'] = np.take(self.mapping_id, obs['semantic'])
            obs['semantic_name'] = np.take(self.mapping_name, obs['semantic'])
            # obs['semantic_region_id'] = np.take(self.mapping_region_id, obs['semantic'])
            # obs['semantic_region_name'] = np.take(self.mapping_region_name, obs['semantic'])

        # check whether the ground truth object has correct id mapping
        if hasattr(self.current_episode, "gt_reach_goal"):
            for g in self.current_episode.goals:
                assert self.mapping_id[g.object_id] == self.mp40_goal_idx

        return obs, self.info


    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        if action == 0:
            self.stopped = True

        obs, reward, done, info = self.env.step(action)
        assert done == self.habitat_env.episode_over

        # update all info into self.info
        self.info.update(info)
        self.info['reward'] = reward

        if 'semantic' in obs:
            obs['semantic_id'] = np.take(self.mapping_id, obs['semantic'])
            obs['semantic_name'] = np.take(self.mapping_name, obs['semantic'])
            # obs['semantic_region_id'] = np.take(self.mapping_region_id, obs['semantic'])
            # obs['semantic_region_name'] = np.take(self.mapping_region_name, obs['semantic'])

        self.world_curr_agent_pose = self.get_agent_pose()
        # Agent height stays constant: camera is always 0.88m above agent base
        # Depth measurements are relative to camera, so ground is always ~0.88m below
        # No adjustment needed for stairs - the relative geometry is unchanged
        self.agent_height = self.camera_height
        self.info['agent_height'] = self.agent_height

        # Get location change
        dx, dy, do = self.get_location_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        self.timestep += 1
        self.info['time'] = self.timestep

        return obs, done, self.info

    def get_agent_pose(self):
        """
        Returns position and rotation of the agent in the world coordinate frame.
        """
        agent_state = self.habitat_env.sim.get_agent_state()
        return {'position': agent_state.position, 'rotation': agent_state.rotation}

    def get_agent_location(self):
        """Returns x, y, o pose of the agent."""

        agent_pose = self.get_agent_pose()
        x = -agent_pose['position'][2]
        y = -agent_pose['position'][0]
        axis = quaternion.as_euler_angles(agent_pose['rotation'])[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_pose['rotation'])[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_pose['rotation'])[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_location_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_agent_location = self.get_agent_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_agent_location, self.world_last_agent_pose)
        self.world_last_agent_pose = curr_agent_location
        return dx, dy, do

