import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
from skimage import measure
import math
from typing import Optional
import os
import habitat_sim

from .mapping_utils import *
from .visualization_refined import write_gpu_point_cloud, write_map_vanilla, COLORS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scripts.run_utils.constant import mp3d_category

"""
BEV_Map.local_map

    Shape: (num_scenes, C, H_local, W_local) where C = num_sem_categories + 4.
    Channel layout (fixed across the code):

        local_map[:, 0, :, :] → fp_map_pred (obstacle/occupancy)
            Built from height-projected voxels around the agent height and the “under-floor” heuristic.
            Thresholded and clamped to [0,1]. Interpreted as occupied / non-traversable estimate per cell.

        local_map[:, 1, :, :] → fp_exp_pred (explored mask)
            Sum over all heights → cells the sensor has observed (“seen” area).
            Also clamped to [0,1]. Interpreted as explored/visible cells (regardless of free/occupied).

    These above two is to distinguish “free because observed empty” from “free because never observed.”

        local_map[:, 2, :, :] → agent current location
            Reset to 0 every step (fill_(0.)), then a 5×5 square centered at the current cell is set to 1.
            Binary indicator of current agent footprint.

        local_map[:, 3, :, :] → agent trajectory / ever-visited footprint
            Not reset each step; the same 5×5 square is written every step and accumulates.
            Effectively a trail (past and current positions).

        local_map[:, 4:4+num_sem_categories, :, :] → per-category semantic evidence
            From agent_height_proj[:, 1:, :, :], scaled by cat_pred_threshold and clamped to [0,1].
            Each channel k (offset by 4) corresponds to semantic category k (the same order as in obs[:, 4:, :, :]).
            2 is floor


BEV_Map.planner_pose_inputs

    Shape: (num_scenes, 7)
    Per row (scene) layout:
        [0] → global x position (meters)
        [1] → global y position (meters)
        [2] → global heading (degrees, normalized to [-180, 180])
        Comes from local_pose + origins and the internal pose update; heading is handled in degrees in get_new_pose_batch.

        [3] → gx1 (left/top row index of the local-window in the full map, in grid cells)
        [4] → gx2 (right/bottom row index, exclusive bound)
        [5] → gy1 (left column index of the local-window in the full map, in grid cells)
        [6] → gy2 (right column index, exclusive bound)
"""

class BEV_Map():
    def __init__(self, args):
        self.args = args

        nc = args.num_sem_categories + 4  # num channels
        self.device = args.device
        self.pcd_device = o3d.core.Device(self.device.upper())

        self.map_size = args.map_size
        self.map_center = self.map_size / 2.0
        self.global_width = args.global_width
        self.global_height = args.global_height
        self.local_width = args.local_width
        self.local_height = args.local_height
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.pcd_resolution = args.map_resolution
        self.local_map_size_cells = int(args.map_size / args.map_resolution) // args.global_downscaling
        self.vision_range = args.vision_range
        self.fov = args.hfov
        self.du_scale = args.du_scale

        self.cat_pred_threshold = args.cat_pred_threshold # 5.0, we sum up the semantic feature of each bev column, so the one-hot vectors sums up. If there are more than 5 cells, we consider it to exist.
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(3.6 / self.z_resolution)  # 3.6 meters (was 360 cm)
        self.min_height = int(-0.8 / self.z_resolution)  # -0.8 meters (was -80 cm)
        self.voxel_dimension = [self.global_width, self.global_height, self.max_height - self.min_height]
        self.agent_height = args.camera_height  # 0.88 meters - camera height for baseline transform
        self.agent_radius = args.agent_radius  # meters
        # Height ranges for obstacle and floor detection (configurable)
        self.obstacle_height_min = args.obstacle_height_min  # 0.1 meters
        self.obstacle_height_max = args.obstacle_height_max  # 1.8 meters
        self.floor_height = args.floor_height  # 0.0 meters
        self.intrinsic_matrix = get_intrinsic_matrix(
            self.screen_w, self.screen_h, self.fov)
        self.max_depth = args.max_depth  # meters
        self.blind_area_max_distance = args.blind_area_max_distance  # meters

        self.agent_world_position = np.zeros(3)
        self.agent_world_rotation = np.zeros((3, 3))

        # Initializing full and local map
        self.full_map = np.zeros((self.global_height, self.global_width, nc))
        self.local_map = np.zeros((self.local_height, self.local_width, nc))

        # Initial full and local pose
        self.full_pose = np.zeros(3)
        self.local_pose = np.zeros(3)

        # Origin of local map
        self.origins = np.zeros(3)

        # Local Map Boundaries
        self.local_map_boundary = np.zeros(4).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros(7)

    def get_local_map_boundaries(self, agent_loc):
        loc_r, loc_c = agent_loc

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - self.local_height // 2, loc_c - self.local_width // 2
            gx2, gy2 = gx1 + self.local_height, gy1 + self.local_width
            if gx1 < 0:
                gx1, gx2 = 0, self.local_height
            if gx2 > self.global_height:
                gx1, gx2 = self.global_height - self.local_height, self.global_height
            if gy1 < 0:
                gy1, gy2 = 0, self.local_width
            if gy2 > self.global_width:
                gy1, gy2 = self.global_width - self.local_width, self.global_width
        else:
            gx1, gx2, gy1, gy2 = 0, self.global_height, 0, self.global_width

        return [gx1, gx2, gy1, gy2]

    def generate_traversability_mask_from_depth(self, rgb, depth, envs):
        """
        Generate traversability mask for current RGB image by checking ground pixels.
        
        Args:
            rgb: (H, W, 3) RGB image
            depth: (H, W) depth map in meters
            envs: environment for is_navigable check
        
        Returns:
            traversable_mask: (H, W) uint8 mask on RGB image (255 for traversable, 0 otherwise)
        """
        # 1. Get 3D point cloud from current depth image
        camera_pcd, _ = get_pointcloud_from_depth(rgb, depth, self.intrinsic_matrix)
        
        if len(camera_pcd) == 0:
            return np.zeros((self.screen_h, self.screen_w), dtype=np.uint8)
        
        # 2. Transform to world coordinates
        world_pcd = translate_to_world(camera_pcd, 
            self.current_camera_world_position, self.current_camera_world_rotation)
        
        # 3. Filter points that are near ground level (floor_height ± 0.3m)
        height_tolerance = 0.3
        ground_mask = np.abs(world_pcd[:, 2] - self.floor_height) < height_tolerance
        
        if not np.any(ground_mask):
            return np.zeros((self.screen_h, self.screen_w), dtype=np.uint8)
        
        # 4. Get pixel coordinates for ground points
        valid_row, valid_col = np.where(depth > 0)
        ground_pixels_row = valid_row[ground_mask]
        ground_pixels_col = valid_col[ground_mask]
        ground_world_points = world_pcd[ground_mask]
        
        # 5. Check navigability for each ground point
        traversable_mask = np.zeros((self.screen_h, self.screen_w), dtype=np.uint8)
        
        for i, (row, col, point) in enumerate(zip(ground_pixels_row, ground_pixels_col, ground_world_points)):
            # Convert to habitat coordinate system and check navigability
            point_sim = point + self.world_origin
            point_sim = np.array([point_sim[0], point_sim[2], point_sim[1]], dtype=np.float64)
            
            if envs.habitat_env.sim.is_navigable(point_sim):
                traversable_mask[row, col] = 255
        
        # 6. Dilate to make mask more continuous
        kernel = np.ones((3, 3), np.uint8)
        traversable_mask = cv2.dilate(traversable_mask, kernel, iterations=1)
        
        return traversable_mask

    def init_map_and_pose(self, infos):
        self.full_map.fill(0)
        self.full_pose.fill(0)
        self.full_pose[:2] = self.map_size / 2.0

        self.planner_pose_inputs[:3] = self.full_pose
        r, c = self.full_pose[0], self.full_pose[1]
        loc_r, loc_c = [int(r / self.args.map_resolution),
                        int(c / self.args.map_resolution)]

        self.full_map[2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.local_map_boundary = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[3:] = self.local_map_boundary
        self.origins = np.array([
            self.local_map_boundary[0] * self.args.map_resolution,
            self.local_map_boundary[2] * self.args.map_resolution, 0.])

        self.local_map = self.full_map[self.local_map_boundary[0]:self.local_map_boundary[1],
                                       self.local_map_boundary[2]:self.local_map_boundary[3], :]
        self.local_pose = self.full_pose - self.origins
        # NOTE: 由于在边界处 local_map 的boundary需要截断，agent并非总是在 local map 的中心

        self.world_origin = habitat_translation(infos['agent_world_position'])
        self.world_rotation = habitat_rotation(infos['agent_world_rotation'])

        # Point cloud
        self.whole_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        # self.traversable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.traversable_pcd_all = o3d.t.geometry.PointCloud(self.pcd_device)

    def mapping(self, obs, infos, envs, debug_save_ply=False, ply_path=None, save_idx=None):
        '''
        obs: (H, W, C) numpy array
        infos: dict
        '''
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3]
        semantic_features = obs[:, :, 4:]

        # Save previous trajectory trail before mapping overwrites it
        prev_trail = self.full_map[:, :, 3:4].copy()

        self.current_agent_world_position = habitat_translation(infos['agent_world_position']) - self.world_origin
        self.current_agent_world_rotation = habitat_rotation(infos['agent_world_rotation'])
        self.current_camera_world_position = habitat_translation(infos['camera_world_position']) - self.world_origin
        self.current_camera_world_rotation = habitat_rotation(infos['camera_world_rotation'])

        self.planner_pose_inputs[:2] = self.current_agent_world_position[:2]
        self.planner_pose_inputs[2] = calculate_yaw(
            self.current_agent_world_rotation, self.world_rotation)

        current_x = ((self.current_agent_world_position[0] + self.map_center) / self.resolution).astype(int)
        current_y = ((self.current_agent_world_position[1] + self.map_center) / self.resolution).astype(int)

        # Extract absolute yaw from current rotation matrix (not relative to start)
        R = self.current_agent_world_rotation
        agent_yaw_rad = np.arctan2(R[1, 0], R[0, 0])
        agent_yaw_deg = np.rad2deg(agent_yaw_rad)

        self.agent_2d_pos = np.array([current_y, current_x])
        self.agent_2d_yaw = agent_yaw_deg

        # Pass (row, col) = (y, x) to match full_map indexing convention
        self.local_map_boundary = self.get_local_map_boundaries((current_y, current_x))
        self.planner_pose_inputs[3:] = self.local_map_boundary
        self.origins = [self.local_map_boundary[0] * self.resolution,
                        self.local_map_boundary[2] * self.resolution, 0.]

        standing_position = np.array([self.current_agent_world_position[0],
            self.current_agent_world_position[1], self.floor_height])

        depth = ori_preprocess_depth(depth)

        # vis_depth = depth.copy()
        # vis_depth[vis_depth > 5.0] = 5.0
        # vis_depth = vis_depth / 5.0 * 255.0
        # vis_depth = vis_depth.astype(np.uint8)
        # cv2.imwrite(ply_path + f"/depth_{save_idx:03d}.png", vis_depth)

        camera_pcd, camera_rgb = get_pointcloud_from_depth(
            rgb, depth, self.intrinsic_matrix)
        world_pcd = translate_to_world(camera_pcd,
            self.current_camera_world_position, self.current_camera_world_rotation)
        current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if len(world_pcd) > 0:
            # cope with semantic features
            valid_x, valid_y = np.where(depth > 0)
            valid_semantic_features = semantic_features[valid_x, valid_y]  # (N, num_sem_categories)
            assert valid_semantic_features.shape[0] == len(world_pcd)

            # 将语义特征填到 full_map 上，取 max（使用 PyTorch scatter_reduce 向量化）
            map_x = world_pcd[:, 0] + self.map_center
            map_y = world_pcd[:, 1] + self.map_center
            cols = (map_x / self.resolution).astype(int)
            rows = (map_y / self.resolution).astype(int)
            valid_mask = (rows >= 0) & (rows < self.global_height) & \
                        (cols >= 0) & (cols < self.global_width)
            valid_rows = rows[valid_mask]
            valid_cols = cols[valid_mask]
            valid_features = valid_semantic_features[valid_mask]  # (M, num_sem_categories)

            if len(valid_rows) > 0:
                num_channels = valid_features.shape[1]
                num_cells = self.global_height * self.global_width
                device = self.device  # 使用 GPU

                # 转换为 PyTorch tensor 并移到 GPU
                linear_idx = valid_rows * self.global_width + valid_cols
                idx_t = torch.from_numpy(linear_idx).long().to(device).unsqueeze(1).expand(-1, num_channels)
                features_t = torch.from_numpy(valid_features).float().to(device)

                # 获取当前语义地图，reshape 成 (num_cells, num_channels)，移到 GPU
                semantic_flat = torch.from_numpy(
                    self.full_map[:, :, 4:4+num_channels].reshape(num_cells, num_channels).copy()
                ).float().to(device)

                # 一次性对所有 channels 取 max（GPU 上执行）
                semantic_flat.scatter_reduce_(0, idx_t, features_t, reduce='amax', include_self=True)

                # 写回 full_map（从 GPU 移回 CPU）
                self.full_map[:, :, 4:4+num_channels] = semantic_flat.cpu().reshape(
                    self.global_height, self.global_width, num_channels).numpy()

                # Explicitly free GPU memory
                del idx_t, features_t, semantic_flat
                torch.cuda.empty_cache()

            current_pcd = gpu_pointcloud_from_array(world_pcd, camera_rgb, self.pcd_device)
            current_navigable_point = safe_select_by_index(current_pcd,
                (current_pcd.point.positions[:, 2] < self.floor_height + 0.2).nonzero()[0])
            current_navigable_position = current_navigable_point.point.positions.cpu().numpy()

            self.whole_pcd = gpu_merge_pointcloud(self.whole_pcd, current_pcd, merge_color=False)
            self.whole_pcd = self.whole_pcd.voxel_down_sample(self.pcd_resolution)

            self.whole_pcd = safe_select_by_index(self.whole_pcd,
                (self.whole_pcd.point.positions[:, 2] > self.floor_height - 0.2).nonzero()[0])
            self.useful_pcd = safe_select_by_index(self.whole_pcd,
                (self.whole_pcd.point.positions[:, 2] < self.obstacle_height_max).nonzero()[0])
            self.obstacle_pcd = safe_select_by_index(self.useful_pcd,
                (self.useful_pcd.point.positions[:, 2] > self.floor_height + 0.2).nonzero()[0])

            current_obstacle_points = self.obstacle_pcd.point.positions.cpu().numpy()
            current_obstacle_points[:, 2] = \
                np.ones_like(current_obstacle_points[:, 2]) * self.floor_height
            if len(current_obstacle_points) > 0:
                current_obstacle_floor_pcd = gpu_pointcloud_from_array(
                    current_obstacle_points,
                    np.ones_like(current_obstacle_points) * 100,
                    self.pcd_device).voxel_down_sample(self.pcd_resolution)
                current_obstacle_floor_points = current_obstacle_floor_pcd.point.positions.cpu().numpy()
            else:
                current_obstacle_floor_points = np.empty((0, 3))

            current_pcd_points = current_pcd.point.positions.cpu().numpy()
            current_pcd_points[:, 2] = np.ones_like(current_pcd_points[:, 2]) * self.floor_height
            if len(current_pcd_points) > 0:
                current_floor_pcd = gpu_pointcloud_from_array(
                    current_pcd_points,
                    np.ones_like(current_pcd_points) * 100,
                    self.pcd_device).voxel_down_sample(self.pcd_resolution)
                current_floor_points = current_floor_pcd.point.positions.cpu().numpy()
            else:
                current_floor_points = np.empty((0, 3))

            closest_distances, closest_points = get_closest_disances_and_points(
                current_obstacle_floor_points, current_floor_points, standing_position, self.blind_area_max_distance)

            if closest_points.shape[0] != 0:
                # interpolate the nearby blind points
                interpolate_points = np.linspace(
                    np.ones_like(closest_points) * standing_position, closest_points,
                    60).reshape(-1, 3)
                interpolate_points = interpolate_points[
                    (interpolate_points[:, 2] > self.floor_height - 0.2) &
                    (interpolate_points[:, 2] < self.floor_height + 0.2)]
                interpolate_points = np.concatenate((current_navigable_position, interpolate_points),
                                                    axis=0)

                interpolate_points[:, 2] = np.ones_like(interpolate_points[:, 2]) * self.floor_height
                interpolate_colors = np.ones_like(interpolate_points) * 100
                current_navigable_pcd = gpu_pointcloud_from_array(interpolate_points,
                                                                interpolate_colors,
                                                                self.pcd_device)
                self.navigable_pcd = gpu_merge_pointcloud(
                    self.navigable_pcd, current_navigable_pcd)

                points_to_check = interpolate_points
            else:
                current_navigable_pcd = safe_select_by_index(current_pcd,
                    (current_pcd.point.positions[:, 2] < self.floor_height + 0.2).nonzero()[0])
                current_navigable_points = current_navigable_pcd.point.positions.cpu().numpy()
                current_navigable_points[:, 2] = np.ones_like(
                    current_navigable_points[:, 2]) * self.floor_height
                self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,
                                                        current_navigable_pcd)

                points_to_check = current_navigable_points

            if not self.navigable_pcd.is_empty():
                self.navigable_pcd = self.navigable_pcd.voxel_down_sample(self.pcd_resolution)

            traversable_points = []
            for point in points_to_check:
                point_sim = point + self.world_origin
                point_sim = np.array([point_sim[0], point_sim[2], point_sim[1]], dtype=np.float64)
                if envs.habitat_env.sim.is_navigable(point_sim):
                    traversable_points.append(point)
            traversable_points = np.array(traversable_points)
            traversable_colors = np.ones_like(traversable_points) * 100
            current_traversable_pcd = gpu_pointcloud_from_array(traversable_points,
                                                                traversable_colors,
                                                                self.pcd_device)
            self.traversable_pcd_all = gpu_merge_pointcloud(self.traversable_pcd_all,
                                                        current_traversable_pcd)
            if not self.traversable_pcd_all.is_empty():
                self.traversable_pcd_all = self.traversable_pcd_all.voxel_down_sample(self.pcd_resolution)
            # self.traversable_pcd = self.traversable_pcd_all.clone()
            # self.traversable_pcd = keep_the_max_connected_component(self.traversable_pcd)

            # explored = navigable_pcd ∪ obstacle_floor_pcd
            # occupied = obstacle_floor_pcd ∪ (navigable_pcd - traversable_pcd_all)
            # 0: occupied, 1: explored

        # save point clouds
        if debug_save_ply and ply_path is not None:
            logger.info(f"save point cloud to ply path: {ply_path}")
            write_gpu_point_cloud(current_pcd, ply_path + f"/current_{save_idx:03d}.ply")
            write_gpu_point_cloud(self.whole_pcd, ply_path + f"/whole_{save_idx:03d}.ply")
            write_gpu_point_cloud(self.useful_pcd, ply_path + f"/useful_{save_idx:03d}.ply")
            write_gpu_point_cloud(self.navigable_pcd, ply_path + f"/navigable_{save_idx:03d}.ply")
            write_gpu_point_cloud(self.obstacle_pcd, ply_path + f"/obstacle_{save_idx:03d}.ply")
            # write_gpu_point_cloud(self.traversable_pcd, ply_path + f"/traversable_{save_idx:03d}.ply")
            write_gpu_point_cloud(self.traversable_pcd_all, ply_path + f"/traversable_all_{save_idx:03d}.ply")

        if not self.navigable_pcd.is_empty():
            nav_points = self.navigable_pcd.point.positions.cpu().numpy()
            map_x = nav_points[:, 0] + self.map_center
            map_y = nav_points[:, 1] + self.map_center
            cols = (map_x / self.resolution).astype(int)
            rows = (map_y / self.resolution).astype(int)
            valid = (rows >= 0) & (rows < self.global_height) & \
                    (cols >= 0) & (cols < self.global_width)
            rows, cols = rows[valid], cols[valid]
            self.full_map[rows, cols, 1] = 1.0
            # self.full_map[rows, cols, 0] = 1.0

        if not self.obstacle_pcd.is_empty():
            obstacle_points = self.obstacle_pcd.point.positions.cpu().numpy()
            map_x = obstacle_points[:, 0] + self.map_center
            map_y = obstacle_points[:, 1] + self.map_center
            cols = (map_x / self.resolution).astype(int)
            rows = (map_y / self.resolution).astype(int)
            valid = (rows >= 0) & (rows < self.global_height) & \
                    (cols >= 0) & (cols < self.global_width)
            rows, cols = rows[valid], cols[valid]
            self.full_map[rows, cols, 1] = 1.0
            self.full_map[rows, cols, 0] = 1.0

        if not self.traversable_pcd_all.is_empty():
            trav_points = self.traversable_pcd_all.point.positions.cpu().numpy()
            map_x = trav_points[:, 0] + self.map_center
            map_y = trav_points[:, 1] + self.map_center

            cols = (map_x / self.resolution).astype(int)
            rows = (map_y / self.resolution).astype(int)
            valid = (rows >= 0) & (rows < self.global_height) & \
                    (cols >= 0) & (cols < self.global_width)
            rows, cols = rows[valid], cols[valid]
            self.full_map[rows, cols, 0] = 0.0

        self.full_map = post_process_map(self.full_map)

        self.full_map[current_y - 2:current_y + 3, current_x - 2:current_x + 3, 2] = 1.0
        self.full_map[:, :, 3:4] = np.maximum(prev_trail, self.full_map[:, :, 3:4])
        self.local_map = self.full_map[self.local_map_boundary[0]:self.local_map_boundary[1],
                                       self.local_map_boundary[2]:self.local_map_boundary[3], :]


        # agent pixel position in local map
        local_y = current_y - self.local_map_boundary[0]  # row offset
        local_x = current_x - self.local_map_boundary[2]  # column offset

        # # Generate traversability mask from current depth image (independent of debug_save_ply)
        # if ply_path is not None and ply_path != "":
        #     traversable_mask = self.generate_traversability_mask_from_depth(rgb, depth, envs)
            
        #     # Save traversability mask to separate folder (under ply_path to preserve scene_id/episode_id structure)
        #     traversability_mask_dir = os.path.join(ply_path, "traversability_masks")
        #     os.makedirs(traversability_mask_dir, exist_ok=True)
        #     mask_save_path = os.path.join(traversability_mask_dir, f"{save_idx:03d}.png")
        #     cv2.imwrite(mask_save_path, traversable_mask)
            
        #     # Optional: save RGB with traversable overlay for visualization
        #     rgb_with_traversable = rgb.copy()
        #     overlay = rgb_with_traversable.copy()
        #     overlay[traversable_mask > 0] = [0, 255, 0]  # Green for traversable
        #     rgb_with_traversable = cv2.addWeighted(rgb_with_traversable, 0.7, overlay, 0.3, 0)
            
        #     rgb_vis_path = os.path.join(traversability_mask_dir, f"{save_idx:03d}_rgb_overlay.png")
        #     cv2.imwrite(rgb_vis_path, cv2.cvtColor(rgb_with_traversable, cv2.COLOR_RGB2BGR))

        return current_y, current_x, agent_yaw_deg, local_y, local_x

    def frontiers_extraction(self,
                             envs,
                             closest_distance=1.6,
                             frontier_path=None,
                             save_idx=None):
        if self.navigable_pcd.is_empty():
            return None, None, None

        exp = (self.full_map[..., 1] == 1).astype(np.uint8)
        exp_border, _ = cv2.findContours(exp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        exp_border_mask = np.zeros_like(exp)
        cv2.drawContours(exp_border_mask, exp_border, -1, (1, 1, 1), 1)

        occ = (self.full_map[..., 0] == 1).astype(np.uint8)
        occ = cv2.dilate(occ, np.ones((3,3)))
        frontier_map = (exp_border_mask > 0) & (occ == 0)
        grid_map_x, grid_map_y = np.where(frontier_map > 0)

        if grid_map_x.shape[0] == 0:
            return None, None, None

        frontiers_3d = grid_coords_to_world(grid_map_x, grid_map_y, self.resolution, self.map_center, self.floor_height)

        valid_frontiers_3d = np.zeros(frontiers_3d.shape[0], dtype=bool)

        for i in range(frontiers_3d.shape[0]):
            current_frontier_point = frontiers_3d[i] + self.world_origin
            current_frontier_point = np.array([current_frontier_point[0], current_frontier_point[2], current_frontier_point[1]], dtype=np.float64)
            if envs.habitat_env.sim.is_navigable(current_frontier_point):
                valid_frontiers_3d[i] = True
            else:
                valid_frontiers_3d[i] = False

        if valid_frontiers_3d.sum() == 0:
            return None, None, None

        frontiers_3d = frontiers_3d[valid_frontiers_3d]
        grid_map_x = grid_map_x[valid_frontiers_3d]
        grid_map_y = grid_map_y[valid_frontiers_3d]
        frontier_map = np.zeros_like(frontier_map)
        frontier_map[grid_map_x, grid_map_y] = True

        # if frontier_path is not None:
        #     frontier_map_vis = frontier_map.reshape(self.voxel_dimension[0], self.voxel_dimension[1], 1)
        #     frontier_map_vis = np.concatenate((frontier_map_vis, frontier_map_vis, frontier_map_vis), axis=2)
        #     frontier_map_vis = frontier_map_vis.astype(np.uint8) * 255
        #     cv2.imwrite(frontier_path + f"/frontier_map_{save_idx:03d}.png", frontier_map_vis)

        #     frontiers_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        #     frontiers_pcd.point.positions = o3d.core.Tensor(frontiers_3d, dtype=o3d.core.Dtype.Float32, device=self.pcd_device)
        #     frontiers_pcd.point.colors = o3d.core.Tensor(np.ones_like(frontiers_3d) * 255, dtype=o3d.core.Dtype.UInt8, device=self.pcd_device)
        #     write_gpu_point_cloud(frontiers_pcd, frontier_path + f"/frontiers_{save_idx:03d}.ply")

        # current_position = self.current_agent_world_position[:2]
        # distance_frontiers = np.linalg.norm(frontiers[:, :2] - np.array(current_position), axis=1)
        # frontiers = frontiers[distance_frontiers > closest_distance]

        frontiers_pcd = o3d.t.geometry.PointCloud()
        frontiers_pcd.point.positions = o3d.core.Tensor(frontiers_3d, device=frontiers_pcd.device)

        # cluster the frontiers
        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = frontiers_pcd.cluster_dbscan(eps=0.3, min_points=3, print_progress=False).cpu().numpy()

        if labels.max() == -1:
            labels = frontiers_pcd.cluster_dbscan(eps=0.3, min_points=1, print_progress=False).cpu().numpy()
            if labels.max() == -1:
                return None, None, None

        max_label = labels.max()
        # extract each cluster and rule out the small ones caluclate the center of each cluster
        frontier_clusters_3d = []
        frontier_centers_3d = []
        for i in range(max_label + 1):
            mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0],
                                              o3d.core.Dtype.Int64,
                                              device=frontiers_pcd.device)
            frontier_cluster = safe_select_by_index(frontiers_pcd, mask_idx_tensor)
            frontier_clusters_3d.append(frontier_cluster.point.positions.cpu().numpy())
            frontier_center = np.mean(frontier_cluster.point.positions.cpu().numpy(), axis=0)
            frontier_centers_3d.append(frontier_center)
        frontier_centers_3d = np.array(frontier_centers_3d)

        # if frontier_path is not None:
        #     # visualize frontier_clusters, frontier_centers
        #     frontier_clusters_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        #     for i in range(len(frontier_clusters_3d)):
        #         frontier_cluster = frontier_clusters_3d[i]
        #         frontier_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        #         frontier_pcd.point.positions = o3d.core.Tensor(frontier_cluster, dtype=o3d.core.Dtype.Float32, device=self.pcd_device)
        #         frontier_pcd.point.colors = o3d.core.Tensor(np.ones_like(frontier_cluster) * COLORS[i % len(COLORS)], dtype=o3d.core.Dtype.UInt8, device=self.pcd_device)
        #         frontier_clusters_pcd = gpu_merge_pointcloud(frontier_clusters_pcd, frontier_pcd, merge_color=False)
        #     write_gpu_point_cloud(frontier_clusters_pcd, frontier_path + f"/frontier_clusters_{save_idx:03d}.ply")

        #     frontier_centers_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        #     frontier_centers_pcd.point.positions = o3d.core.Tensor(frontier_centers_3d, dtype=o3d.core.Dtype.Float32, device=self.pcd_device)
        #     frontier_centers_pcd.point.colors = o3d.core.Tensor(np.ones_like(frontier_centers_3d) * COLORS[len(frontier_clusters_3d) % len(COLORS)], dtype=o3d.core.Dtype.UInt8, device=self.pcd_device)
        #     write_gpu_point_cloud(frontier_centers_pcd, frontier_path + f"/frontier_centers_{save_idx:03d}.ply")

        # logger.info(f'Total number of clusters: {len(frontier_clusters_3d)}')

        rows, cols = world_coords_to_grid(frontier_centers_3d, self.resolution, self.map_center, self.global_width, self.global_height)
        frontier_centers_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]), dtype=bool)
        frontier_centers_map[rows, cols] = True
        frontier_centers_2d = np.concatenate((np.array([rows]).reshape(-1, 1), np.array([cols]).reshape(-1, 1)), axis=1)

        if len(frontier_centers_2d) == 0:
            return frontier_centers_3d, None, None

        # Convert frontier centers to local map coordinates
        frontier_centers_2d_local = frontier_centers_2d - np.array([self.local_map_boundary[0], self.local_map_boundary[2]])

        # Filter frontiers that are within local map bounds
        local_h = self.local_map_boundary[1] - self.local_map_boundary[0]
        local_w = self.local_map_boundary[3] - self.local_map_boundary[2]
        valid_mask = (frontier_centers_2d_local[:, 0] >= 0) & (frontier_centers_2d_local[:, 0] < local_h) & \
                     (frontier_centers_2d_local[:, 1] >= 0) & (frontier_centers_2d_local[:, 1] < local_w)

        # If no frontiers are within local map bounds, return None for valid_mask
        if not valid_mask.any():
            valid_mask = None

        # return:
        # 3d frontier centers in world coordinates,
        # 2d frontier centers in full map coordinates
        # valid mask: which frontiers are within local map bounds from 2d frontier centers (None if none are valid)
        return frontier_centers_3d, frontier_centers_2d, valid_mask


    def frontier_postprocess(self, frontier_centers_2d_full, frontier_centers_3d, valid_mask,
                            reference_replay, current_idx, habitat_env):
        """
        Post-process frontiers: select best frontier and compute local/global indices.

        Args:
            frontier_centers_2d_full: (N, 2) array of frontier centers in full map coordinates
            frontier_centers_3d: (N, 3) array of frontier centers in 3D world coordinates
            valid_mask: (N,) boolean mask indicating which frontiers are in local map bounds
            reference_replay: ground truth trajectory
            current_idx: current step index
            habitat_env: habitat environment for pathfinding

        Returns:
            frontier_centers_2d_local: (M, 2) array of valid frontiers in local map coordinates (M = sum(valid_mask))
            selected_local_indices: list of [local_index_pose, local_index_future] or [None, None]
            selected_global_indices: list of [global_index_pose, global_index_future] or [None, None]
        """
        if frontier_centers_2d_full is None or valid_mask is None:
            return None, [None, None], [None, None]

        # Get valid indices (global positions in full frontier array)
        valid_indices = np.where(valid_mask)[0]

        # Compute local map coordinates for valid frontiers
        frontier_centers_2d_local = frontier_centers_2d_full[valid_mask] - np.array([[
            self.local_map_boundary[0], self.local_map_boundary[2]
        ]])

        # Sanity check
        assert len(frontier_centers_2d_local) == len(valid_indices), \
            f"Local frontier count mismatch: {len(frontier_centers_2d_local)} vs {len(valid_indices)}"

        # Select best frontier by current pose (returns LOCAL index into filtered array)
        selected_local_index_pose = self.frontier_gt(
            frontier_centers_2d_full[valid_mask], heading_weight=1.0
        )

        # Select best frontier by future trajectory (returns LOCAL index into filtered array)
        selected_local_index_future = self.frontier_gt_future(
            frontier_centers_3d[valid_mask], current_idx, habitat_env, reference_replay
        )

        # Convert local indices to global indices for full map visualization
        selected_global_index_pose = None
        selected_global_index_future = None
        if selected_local_index_pose is not None:
            selected_global_index_pose = valid_indices[selected_local_index_pose]
        if selected_local_index_future is not None:
            selected_global_index_future = valid_indices[selected_local_index_future]

        return (frontier_centers_2d_local,
                [selected_local_index_pose, selected_local_index_future],
                [selected_global_index_pose, selected_global_index_future])

    def frontier_gt(self, frontier_centers_2d, heading_weight=0.9):
        """
        Find the frontier that the agent is most likely heading towards using 2D pixel coordinates.

        Args:
            heading_weight: Weight for heading angle (default 0.9), distance weight is (1 - heading_weight)

        Returns:
            best_frontier_idx: Index of the selected frontier, or None if no frontiers
        """
        if frontier_centers_2d is None:
            return None

        distance_weight = 1 - heading_weight

        # Agent position in 2D pixel coordinates (row, col)
        agent_pos = self.agent_2d_pos  # [row, col] = [y, x]
        agent_yaw = self.agent_2d_yaw  # in degrees

        # Convert yaw to heading vector in pixel space [row, col]
        yaw_rad = np.deg2rad(agent_yaw - 90)
        agent_heading_vec = np.array([np.sin(yaw_rad), np.cos(yaw_rad)])  # [row_dir, col_dir]
        agent_heading_vec = agent_heading_vec / (np.linalg.norm(agent_heading_vec) + 1e-8)

        best_frontier_idx = None
        best_score = float('inf')

        for i, frontier_2d in enumerate(frontier_centers_2d):
            # frontier_2d is [row, col]
            # Calculate distance in pixel space
            distance = np.linalg.norm(frontier_2d - agent_pos)

            # Calculate vector from agent to frontier in pixel coordinates
            to_frontier = frontier_2d - agent_pos
            to_frontier_normalized = to_frontier / (np.linalg.norm(to_frontier) + 1e-8)

            # Calculate angle between agent's heading and direction to frontier
            cos_angle = np.dot(agent_heading_vec, to_frontier_normalized)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # in radians [0, pi]

            # Normalize scores to [0, 1]
            # For angle: 0 radians (facing directly) -> 0, pi radians (opposite) -> 1
            angle_score = angle / np.pi

            # For distance: normalize by reference distance in pixels
            max_distance_pixels = 100.0  # reference distance in pixels
            distance_score = min(distance / max_distance_pixels, 1.0)

            # Combined score (lower is better)
            score = heading_weight * angle_score + distance_weight * distance_score

            if score < best_score:
                best_score = score
                best_frontier_idx = i




        return best_frontier_idx

    def calculate_chamfer_distance(self, pth1, pth2):
        """
        计算 pth2 中每个点到 pth1 的最短距离，并求和（只比较第 0 和第 2 维，忽略高度）

        Args:
            pth1: (N, 3) numpy array, 参考轨迹点
            pth2: (M, 3) numpy array, 待比较轨迹点

        Returns:
            float: 所有最短距离的和
        """
        # 只取第 0 和第 2 维（去掉高度维度 1）
        pth1_2d = pth1[:, [0, 2]]  # (N, 2)
        pth2_2d = pth2[:, [0, 2]]  # (M, 2)
        # pth2_2d[:, None, :] -> (M, 1, 2), pth1_2d[None, :, :] -> (1, N, 2)
        # 广播后得到 (M, N, 2) 的差值矩阵
        diff = pth2_2d[:, np.newaxis, :] - pth1_2d[np.newaxis, :, :]
        # 计算欧氏距离，得到 (M, N) 的距离矩阵
        distances = np.linalg.norm(diff, axis=2)
        # 对于 pth2 中的每个点，找到到 pth1 最近的距离
        min_distances = np.min(distances, axis=1)
        # 求和
        return np.mean(min_distances)

    def frontier_gt_future(self, frontier_centers_3d, current_idx, habitat_env, reference_replay, horizon=-1):

        if frontier_centers_3d is None:
            return None

        frontiers_world_3d = frontier_centers_3d + self.world_origin
        gt_trajectory = []
        for i in range(current_idx, len(reference_replay)):
            rp = reference_replay[i]
            gt_trajectory.append(rp['agent_state']['position'])
        gt_trajectory = np.array(gt_trajectory)
        if horizon > 0:
            horizon = min(horizon, len(gt_trajectory))
            gt_trajectory = gt_trajectory[:horizon]

        current_position = self.current_agent_world_position + self.world_origin
        current_position = np.array([current_position[0], current_position[2], current_position[1]], dtype=np.float64)
        scores = []
        for i, frontier in enumerate(frontiers_world_3d):
            frontier_habitat = np.array([frontier[0], frontier[2], frontier[1]], dtype=np.float64)
            path = habitat_sim.ShortestPath()
            path.requested_start = current_position
            path.requested_end = frontier_habitat
            found = habitat_env.sim.pathfinder.find_path(path)
            if not found:
                scores.append(np.inf)
                logger.info(f"!!!!!!!!!!!!!!!!!!!!fuckfuckfuck Path not found for frontier {i}, position: {current_position}, frontier: {frontier_habitat}")
            else:
                current_path = np.array(path.points)
                scores.append(self.calculate_chamfer_distance(gt_trajectory, current_path))
        return np.argmin(scores)

    def objects_extraction(self, full_map_pred):
        semantic_map = full_map_pred[4:]

        dst = np.zeros(semantic_map[0, :, :].shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        Object_list = {}
        """
        For each channel (semantic_map[i]):
            Converts tensor → NumPy.
            Threshold at 0.1 → binary mask.
            Apply morphological closing with a 7×7 kernel (removes holes, smooths blobs).
            Extract contours (cv2.findContours).
            Approximate polygon (cv2.approxPolyDP) if contour length > 30.
            Draw polyline into dst (for visualization).
            Store polygons into Single_object_list.
        """
        for i in range(len(semantic_map)):
            if semantic_map[i, :, :].sum() != 0:
                Single_object_list = []
                se_object_map = semantic_map[i, :, :].cpu().numpy()
                se_object_map[se_object_map > 0.1] = 1
                se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
                contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map, 0.1, 1), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_NONE)
                for cnt in contours:
                    if len(cnt) > 30:
                        epsilon = 0.05 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        Single_object_list.append(approx)
                        cv2.polylines(dst, [approx], True, 1)
                if len(Single_object_list) > 0:
                    Object_list[mp3d_category[i]] = Single_object_list

        return Object_list