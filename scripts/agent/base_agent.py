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

SAVE_MASKRCNN_SEMANTIC = False
SAVE_GROUNDINGSAM2_SEMANTIC = False
if SAVE_MASKRCNN_SEMANTIC:
    from scripts.run_utils.SemanticPredMaskRCNN_semantic_prediction import SemanticPredMaskRCNN

class BaseAgent:
    """
    Base agent class with common functionality for navigation agents.
    Subclasses should implement episode-specific logic.
    """

    def __init__(self, config, envs):
        self.config = config
        self.envs = envs
        self.possible_action_env = list(envs.env.env.habitat_env.task.actions.keys())

        self.device = config.device

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((config.mapping.frame_height, config.mapping.frame_width),
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

        obs['world_pos'] = self.envs.habitat_env.sim.get_agent_state().position
        obs['world_rotation'] = self.envs.habitat_env.sim.get_agent_state().rotation

        self.observation = obs
        self.info = info
        self.info["agent_world_position"] = self.envs.habitat_env.sim.get_agent_state().position
        self.info["agent_world_rotation"] = self.envs.habitat_env.sim.get_agent_state().rotation
        self.info["camera_world_position"] = self.envs.habitat_env.sim.get_agent_state().sensor_states['rgb'].position
        self.info["camera_world_rotation"] = self.envs.habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation

        self.seg_idx_obj = obs['semantic_id']
        self.seg_name_obj = obs['semantic_name']
        # self.seg_idx_region = obs['  ']
        # self.seg_name_region = obs['semantic_region_name']

        # Process RGBD observations
        rgbd = np.concatenate((obs['rgb'].astype(np.uint8), obs['depth']), axis=2)
        rgbd, seg_predictions = self.preprocess_obs(rgbd, use_seg=True, sem_idx_gt=self.seg_idx_obj)

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

        # Create output directories
        self._create_output_directories(output_dir)

        return obs, rgbd, info

    def _create_output_directories(self, output_dir):
        """
        Create output directories for visualization.
        Subclasses can override to customize directory structure.
        """
        base_dir = self._get_episode_base_dir()

        if self.config.environment == 'habitat':
            subfolders = ["rgb", "depth", "semantic", 
                          "full_occupancy_explore", 
                          "local_occupancy_explore",
                          "full_occupancy_explore_frontier", 
                          "full_occupancy_explore_frontier_gt", 
                          "local_occupancy_explore_frontier",
                          "local_occupancy_explore_frontier_gt",
                          "fov",
                          "top_down_map",
                        #   "point_cloud_debug", 
                          "combined", 
                        #   "history_rgbs"
                          ]
        elif self.config.environment == 'real_world':
            # TODO
            subfolders = ["rgb", "depth", "semantic", "maskrcnn", "bev", "occ"]

        self.paths = {}
        for folder in subfolders:
            path = Path(output_dir) / folder / base_dir
            path.mkdir(parents=True, exist_ok=True)
            self.paths[folder] = str(path)

    def _get_episode_base_dir(self):
        """
        Get the base directory for episode outputs.
        Override in subclasses for different naming schemes.
        """
        return f"{self.scene_id}/{self.ep_id}"

    def preprocess_obs(self, obs, use_seg=True, sem_idx_gt: np.ndarray = None):
        """
        Preprocess observations (RGB, depth, semantics).

        Args:
            obs: Raw observations (C x H x W)
            use_seg: Whether to use semantic segmentation
            sem_idx_gt: Ground truth semantic indices

        Returns:
            state: Preprocessed state (C x H x W)
            seg_predictions: Segmentation predictions (if any)
        """
        # Extract RGB and depth
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        self.depth_vis = convert_depth_to_rgb(depth)

        if sem_idx_gt is not None:
            # Use GT semantics
            sem_seg_pred = seg_idx_to_onehot(sem_idx_gt, self.num_sem_categories + 1)[..., 1:]
            self.rgb_vis = rgb[:, :, ::-1]
            seg_predictions = None
        else:
            # Fallback to predicted semantics
            sem_seg_pred, seg_predictions = self.pred_sem(rgb.astype(np.uint8),
                                                          depth=None,
                                                          use_seg=use_seg,
                                                          pred_bbox=False)

        # Downscale if needed
        if self.ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[self.ds // 2::self.ds, self.ds // 2::self.ds]
            sem_seg_pred = sem_seg_pred[self.ds // 2::self.ds, self.ds // 2::self.ds]

        # depth = np.expand_dims(depth, axis=2)

        # Concatenate to state
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2)

        return state, seg_predictions

    def get_action(self, planner_inputs):
        """
        Plan and return action based on planner inputs.

        Args:
            planner_inputs (dict): Planning information

        Returns:
            action (int): action id
        """
        config_mapping = self.config.mapping
        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs['exp_pred'])
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r / config_mapping.map_resolution - gx1),
                 int(c / config_mapping.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape) \
 \
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r / config_mapping.map_resolution - gx1),
                      int(c / config_mapping.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        # self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
        #                                start[1] - 0:start[1] + 1] = 1
        rr, cc, _ = line_aa(last_start[0], last_start[1], start[0], start[1])
        self.visited[gx1:gx2, gy1:gy2][rr, cc] += 1

        self.visited_vis[gx1:gx2, gy1:gy2] = \
            draw_line(last_start, start,
                      self.visited_vis[gx1:gx2, gy1:gy2])

        # relieve the stuck goal
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        if abs(x1 - x2) >= 0.05 or abs(y1 - y2) >= 0.05:
            self.been_stuck = False
            self.stuck_goal = None

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                    self.been_stuck = True
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < config_mapping.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                             ((i + buf) * np.cos(np.deg2rad(t1))
                              + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                             ((i + buf) * np.sin(np.deg2rad(t1))
                              - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r / config_mapping.map_resolution), \
                            int(c / config_mapping.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

    # def visualize(self, inputs, save_idx, dir='outputs/tmp'):
    #     """
    #     Visualize the current state of navigation.

    #     Args:
    #         depth: (H, W, 1) normalized float32 in [0,1] from Habitat.
    #         min_d: minimum depth in meters
    #         max_d: maximum depth in meters

    #     Returns:
    #         depth_m: (H, W) float32, depth in meters.

    #     """
    #     assert depth.ndim == 3 and depth.shape[2] == 1
    #     d = depth[:, :, 0].astype(np.float32)  # (H, W)

    #     # 1) Clamp to [0,1] to avoid weird values
    #     d = np.clip(d, 0.0, 1.0)

    #     # 2) Mark invalid pixels for later filtering
    #     # Far-plane (>0.99) and zero readings should not create obstacles
    #     # but should still mark free space as explored
    #     far_plane = (d > 0.99)
    #     invalid = (d < 0.01)

    #     # Clamp invalid to max depth (will be used for free space carving only)
    #     d[far_plane | invalid] = 1.0

    #     # 3) Map normalized depth to meters: [0,1] -> [min_d, max_d]
    #     depth_m = min_d + d * (max_d - min_d)  # (H, W) in meters

    #     return depth_m


    def visualize(self, inputs, save_idx, dir='outputs/tmp'):
        config_mapping = self.config.mapping

        color_palette = [
            1.0, 1.0, 1.0,
            0.6, 0.6, 0.6,
            0.95, 0.95, 0.95,
            0.96, 0.36, 0.26,
            0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
            0.9400000000000001, 0.7818, 0.66,
            0.8882000000000001, 0.9400000000000001, 0.66,
            0.66, 0.9400000000000001, 0.8518000000000001,
            0.7117999999999999, 0.66, 0.9400000000000001,
            0.9218, 0.66, 0.9400000000000001,
            0.9400000000000001, 0.66, 0.748199999999999]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # add a check with collision map
        map_pred[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 1

        sem_map += 5

        no_cat_mask = sem_map == 11
        # no_cat_mask = np.logical_or(no_cat_mask, 1 - no_cat_mask)
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        # vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        # # <goal>
        # selem = skimage.morphology.disk(4)
        # goal_mat = 1 - skimage.morphology.binary_dilation(
        #     goal, selem) != True
        #
        # goal_mask = goal_mat == 1
        # sem_map[goal_mask] = 4
        # # </goal>

        locs = np.array(self.envs.habitat_env.current_episode.goals[0].position[:2]) + np.array([18, 18])
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r / config_mapping.map_resolution),
                        int(c / config_mapping.map_resolution)]

        if gx1 + 1 <= loc_c < gx2 - 1 and gy1 + 1 <= loc_r < gy2 - 1:
            sem_map[loc_r - gy1 - 1:loc_r - gy1 + 2, loc_c - gx1 - 1:loc_c - gx1 + 2] = [255, 0, 0]

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        # sem_map_vis = insert_s_goal(self.s_goal, sem_map_vis, goal)
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        # 360 x 480 x 3
        rgb_vis = cv2.resize(self.rgb_vis, (320, 240),
                             interpolation=cv2.INTER_NEAREST)
        depth_vis = cv2.resize(self.depth_vis, (320, 240))
        # concatenate rgb and depth
        rgb_visualization = np.vstack((rgb_vis, depth_vis))

        # pad rgb_visualization to 480 x 480
        h, w, c = rgb_visualization.shape
        target_w = 480

        # 需要补多少列
        pad_total = target_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        # 用白色(255,255,255)来填充背景
        rgb_visualization = cv2.copyMakeBorder(
            rgb_visualization,
            top=0, bottom=0, left=pad_left, right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)  # 白色
        )

        vis_image = self.vis_image_background.copy()

        vis_image[50:530, 770:1250] = sem_map_vis

        vis_image[50:530, 265:745] = rgb_visualization

        cv2.rectangle(vis_image, (25, 50), (240, 265), (128, 128, 128), 1)
        cv2.rectangle(vis_image, (25, 315), (240, 530), (128, 128, 128), 1)

        cv2.rectangle(vis_image, (770, 50), (1250, 530), (128, 128, 128), 1)
        cv2.rectangle(vis_image, (265, 50), (745, 530), (128, 128, 128), 1)

        TOP = 50
        MAP_LEFT = 770
        MAP_SIZE = 480  # 480x480 panel

        # ---- window sizes (rows, cols) ----
        H_win = gx2 - gx1  # rows
        W_win = gy2 - gy1  # cols

        # 全局索引（行/列）
        row_idx = start_y / config_mapping.map_resolution  # rows
        col_idx = start_x / config_mapping.map_resolution  # cols

        # 相对窗口的索引
        row_rel = row_idx - gx1
        col_rel = col_idx - gy1

        # 映射到面板像素（注意 y 翻转，因为做了 flipud）
        x_pix = col_rel * (MAP_SIZE / W_win)
        y_pix = (H_win - row_rel) * (MAP_SIZE / H_win)

        theta = np.deg2rad(-start_o)

        pos = (float(x_pix), float(y_pix), float(theta))
        agent_arrow = get_contour_points(pos, origin=(MAP_LEFT, TOP))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        os.makedirs(dir, exist_ok=True)
        height, width, layers = vis_image.shape
        image_name = f"bev_map_{save_idx:03d}.png"
