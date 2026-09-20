import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import skimage.morphology
import torch
from skimage.draw import line_aa
import os
from pathlib import Path
import re

from habitat import logger
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

import scripts.run_utils.pose_utils as pu
from scripts.run_utils.mapping.visualization import draw_line, get_contour_points, init_vis_image
from scripts.run_utils.tools import _parse_obj_id, seg_idx_to_onehot
from task_patch.utils.vis_utils import convert_depth_to_rgb
from scripts.agent.base_agent import BaseAgent

SAVE_MASKRCNN_SEMANTIC = False
SAVE_GROUNDINGSAM2_SEMANTIC = False
if SAVE_MASKRCNN_SEMANTIC:
    from scripts.run_utils.SemanticPredMaskRCNN_semantic_prediction import SemanticPredMaskRCNN




class IF_Agent(BaseAgent):

    def __init__(self, config, envs):
        self.config = config
        self.envs = envs
        self.possible_action_env = list(envs.env.env.habitat_env.task.actions.keys())

        self.device = config.device

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((config.mapping.frame_height,config.mapping.frame_width),
                               interpolation=Image.NEAREST)])

        # Downscaling factor
        self.ds = config.mapping.env_frame_width // config.mapping.frame_width  # Downscaling factor

        if SAVE_MASKRCNN_SEMANTIC:
            self.sem_pred = SemanticPredMaskRCNN(config.semantic_prediction)
            self.save_maskrcnn = True
            # SemanticPredMaskRCNN semantic prediction (only 16 classes and not open vocabulary)

        self.selem = skimage.morphology.disk(3)

        self.rgbd = None
        self.info = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None

        self.global_width = config.mapping.global_width
        self.global_height = config.mapping.global_height
        self.local_width = config.mapping.local_width
        self.local_height = config.mapping.local_height

        self.num_sem_categories = config.mapping.num_sem_categories

        if SAVE_MASKRCNN_SEMANTIC:
            self.save_maskrcnn = True
            # SemanticPredMaskRCNN semantic prediction (only 16 classes and not open vocabulary)
            sem_pred = SemanticPredMaskRCNN(config.semantic_prediction)

        if SAVE_GROUNDINGSAM2_SEMANTIC:
            self.save_groundingsam2 = True
            from scripts.run_utils.groundingsam2 import detect_segment
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from grounding_dino.groundingdino.util.inference import load_model

            device = "cuda" if torch.cuda.is_available() else "cpu"

            sam2_model = build_sam2(config.semantic_prediction.SAM2_MODEL_CONFIG,
                                    config.semantic_prediction.SAM2_CHECKPOINT,
                                    device)
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            grounding_model = load_model(config.semantic_prediction.GROUNDING_DINO_CONFIG,
                                         config.semantic_prediction.GROUNDING_DINO_CHECKPOINT,
                                         device)


    def reset(self, output_dir='outputs/tmp'):
        """Resets the environment to a new episode.
        Inputs:
            output_dir (str): log/data_path/split/experiment_mode

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        config_mapping = self.config.mapping

        # 640 x 480
        obs, info = self.envs.reset(config_mapping, return_info=True)
        
        self.observation = obs
        self.info = info

        obs['world_pos'] = self.envs.habitat_env.sim.get_agent_state().position
        obs['world_rotation'] = self.envs.habitat_env.sim.get_agent_state().rotation
        self.info["agent_world_position"] = self.envs.habitat_env.sim.get_agent_state().position
        self.info["agent_world_rotation"] = self.envs.habitat_env.sim.get_agent_state().rotation
        self.info["camera_world_position"] = self.envs.habitat_env.sim.get_agent_state().sensor_states['rgb'].position
        self.info["camera_world_rotation"] = self.envs.habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation


        self.seg_idx_obj = obs['semantic_id']
        self.seg_name_obj = obs['semantic_name']
        # self.seg_idx_region = obs['  ']
        # self.seg_name_region = obs['semantic_region_name']

        # 4 x 480 x 640
        rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2)

        # 20 x 120 x 160
        rgbd, seg_predictions, = self.preprocess_obs(rgbd, use_seg=True, sem_idx_gt=self.seg_idx_obj)

        self.rgbd = rgbd

        self.obs_shape = rgbd.shape

        # Episode initializations
        map_shape = (int(config_mapping.map_size / config_mapping.map_resolution),
                     int(config_mapping.map_size / config_mapping.map_resolution))
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [config_mapping.map_size / 2.0,
                         config_mapping.map_size / 2.0, 0.]
        self.last_action = None
        self.pred_box = []
        self.been_stuck = False
        self.stuck_goal = None
        self.frontier_vis = None

        self.reference_path = self.envs.habitat_env.current_episode.reference_path
        self.start_position = self.envs.habitat_env.current_episode.start_position
        self.start_rotation = self.envs.habitat_env.current_episode.start_rotation


        # create folder to save vis data
        self.current_episode = self.envs.env.habitat_env.current_episode
        self.scene_id = self.current_episode.scene_id.split("/")[-1][:-4]
        self.ep_id = self.current_episode.episode_id

        base_dir = f"{self.scene_id}/{self.ep_id}"
        if self.config.environment == 'habitat':
            subfolders = ["rgb", "depth", "semantic", 
                          "top_down_map", 
                          'pixel', "his_multip_color", 'rgb_debug',
                            # for visualization
                          "semantic", "semantic_room",
                          "full_occupancy_explore", "local_occupancy_explore", "full_occupancy_explore_frontier", 
                          "full_occupancy_explore_frontier_gt", "local_occupancy_explore_frontier", "local_occupancy_explore_frontier_gt",
                          "combined", "point_cloud_debug", "fov", "his_KF_multi_color",  "his_KF_single_color", "his_KF_no_color", "local_occupancy_explore_multi_color",
                          ]
        elif self.config.environment == 'real_world':
            # TODO
            subfolders = ["rgb", "depth", "semantic", "maskrcnn", "bev", "occ"]


        self.paths = {}
        for folder in subfolders:
            path = Path(output_dir) / folder / base_dir
            path.mkdir(parents=True, exist_ok=True)
            self.paths[folder] = str(path)

        return obs, rgbd, info

    def step(self, best_action):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        self.gt_action = best_action

        obs, done, info = self.envs.step(self.gt_action)

        obs['world_pos'] = self.envs.habitat_env.sim.get_agent_state().position
        obs['world_rotation'] = self.envs.habitat_env.sim.get_agent_state().rotation

        self.observation = obs
        self.info = info

        self.seg_idx_obj = obs['semantic_id']
        self.seg_name_obj = obs['semantic_name']

        self.info["agent_world_position"] = self.envs.habitat_env.sim.get_agent_state().position
        self.info["agent_world_rotation"] = self.envs.habitat_env.sim.get_agent_state().rotation
        self.info["camera_world_position"] = self.envs.habitat_env.sim.get_agent_state().sensor_states['rgb'].position
        self.info["camera_world_rotation"] = self.envs.habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation
        

        # Process observations
        rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2)
        self.raw_obs = rgbd[:, :, :3]
        self.raw_depth = rgbd[:, :, 3]

        # rgbd, seg_predictions = self.preprocess_obs(rgbd)
        rgbd, seg_predictions, = self.preprocess_obs(rgbd, use_seg=True, sem_idx_gt=self.seg_idx_obj)

        self.last_action = self.gt_action
        self.rgbd = rgbd


        return obs, rgbd, done, self.envs.info
