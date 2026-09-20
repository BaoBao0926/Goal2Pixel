from collections import deque
import torch
import random
import sys
import os
import cv2
import hydra
import numpy as np
from PIL import Image, ImageDraw
import time

from collections import Counter, defaultdict
import json
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# habitat
from habitat import logger
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

# modules
from task_patch.env_register.env import AnnotatedGymHabitatEnv
from run_utils.config_patch import patch_exp_name, update_and_save_hydra_config
from scripts.run_utils.env_construct import construct_envs, modify_config
from task_patch.utils.get_config import register_plugins
from scripts.agent.if_map_generate_data import IF_Agent_pixel as IF_Agent
from il_patch.register.config import HabitatBaselinesConfigPlugin_BC

from run_utils.mapping.mapping import BEV_Map
from scripts.run_utils.mapping.mapping_utils import world_to_pixel, get_camera_matrix, world_to_agent, pixel_to_agent_wp, agent_to_world, agent_to_pixel
from scripts.run_utils.mapping.visualization_refined import write_all_images

from scripts.scripts_for_generating_data.useless_util.find_scene_name import SCENE_LIST, FIRST_THREE

SAVE_MASKRCNN_SEMANTIC      = True
SAVE_GROUNDINGSAM2_SEMANTIC = True
DEBUG_SAVE_PLY = False  # save point clouds to PLY for debugging (set to False for normal runs)
DEBUG_PLY_STEPS = [0, 5, 10, 45]  # which steps to save PLY files for
MAX_STEPS_PER_WP = 400

CAMERA_HEIGHT_OFFSET = 0.08

WRITE_VISUALIZE             = False     # visualize the mapping process step by step
WRITE_PIXEL_VISUALIZATION   = True      # visualize the pixel selection process step by step
SUCCESS_DISTANCE=0.4
NUMBER_OF_EPISODES  = -1         # whether running first NUMBER_OF_EPISODE episode. -1 means all episodes
IF_RESUME           = True        # whether resume rollout
IF_SCENE_FILTER     = True        # whether only run episodes in SCENE_LIST
SCENE_LIST          = SCENE_LIST  # which scene list to use if IF_SCENE_FILTER is True


IF_vis_1 = False
IF_vis_2 = False
IF_vis_34 = True
MIN_DISTANCE_v3=0.6     # distance between visible and invisible threshold
MAX_INVISIBLE_GAP = 10  # if 0-N is visible, N+1-M is invisible, M+1-Z is visible again. If (M-N-1) > MAX_INVISIBLE_GAP, it will discard the second visible path

Padding = 8  # pixels on each side


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg: "DictConfig"):
    # insert the experiment name into the config
    cfg = patch_exp_name(cfg)

    # Modifies a configuration by inferring some missing keys and makes sure some keys are present and compatible with each other.
    cfg = patch_config(cfg)

    # update the new cfg to hydra logging
    update_and_save_hydra_config(cfg)

    # start the exp
    execute_exp(cfg)



def check_exist(output_dir, scene_id, ep_id):
    json_path = os.path.join(output_dir, "annotations", scene_id, f"{ep_id}.json")
    return os.path.exists(json_path)


def record_finished_episode(output_dir, scene_id, ep_id, time_taken):
    record_file = os.path.join(output_dir, "finished_episodes.json")
    entry = {
        "scene_id": scene_id,
        "episode_id": ep_id,
        "time": time_taken
    }
    with open(record_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def project_and_visualize_waypoint_1(point, wp_i, agent, config, camera_matrix, PADDING = Padding):
    """
    Project world waypoint to pixel coordinates and visualize it on RGB image with padding.
    Adds 24-pixel padding on left and right sides to show out-of-view waypoints.
    Out-of-view waypoints are positioned proportionally based on their horizontal angle.
    
    Args:
        point: World coordinates of the waypoint [x, y, z]
        wp_i: Waypoint index
        agent: IF_Agent instance
        config: Configuration object
        camera_matrix: Camera intrinsics matrix
        
    Returns:
        tuple: (rgb_with_waypoint, waypoint_pixel)
            - rgb_with_waypoint: RGB image with waypoint marker and padding
            - waypoint_pixel: Pixel coordinates [u, v] or None if not visible
    """

    
    # Get current agent state
    agent_state = agent.envs.habitat_env.sim.get_agent_state()
    
    # Get agent frame coordinates (needed to determine left/right for out-of-view waypoints)
    wp_agent = world_to_agent(point, agent_state.position, agent_state.rotation)
    right, fwd, up = float(wp_agent[0]), float(wp_agent[1]), float(wp_agent[2])
    
    # Project waypoint to pixel coordinates
    waypoint_pixel, wp_agent = world_to_pixel(
        world_wp=point,
        world_pos=agent_state.position,
        world_rot=agent_state.rotation,
        camera_matrix=camera_matrix,
        sensor_height=config.mapping.camera_height,
        camera_elevation_degree=0,
        image_height=config.mapping.env_frame_height,
        image_width=config.mapping.env_frame_width,
    )
    
    # Add padding to RGB image (gray color for padding areas)
    rgb_base = agent.rgb_vis.copy()
    height, width = rgb_base.shape[:2]
    
    # Create padded image with gray padding
    rgb_padded = np.ones((height, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
    rgb_padded[:, PADDING:PADDING + width, :] = rgb_base
    
    # Calculate horizontal angle in agent frame
    horizontal_angle = np.arctan2(right, fwd)  # angle in radians, -pi to pi
    hfov_rad = np.deg2rad(config.mapping.hfov)
    half_fov = hfov_rad / 2.0
    
    # Determine waypoint visualization position
    if waypoint_pixel is not None:
        # Waypoint is visible in camera view
        u, v = int(waypoint_pixel[0]), int(waypoint_pixel[1])
        # Adjust u coordinate for padding
        u_padded = u + PADDING
        color = (0, 0, 255)  # Red in BGR
        marker_type = cv2.MARKER_CROSS
        # logger.info(f"Waypoint {wp_i} visible at pixel ({u}, {v})")
    else:
        # Waypoint is out of view - map angle to padding position proportionally
        # FOV range: [-half_fov, half_fov] maps to image [0, width]
        # Left padding: angles from [-pi, -half_fov] map to [0, PADDING]
        # Right padding: angles from [half_fov, pi] map to [width+PADDING, width+2*PADDING]
        
        if horizontal_angle < -half_fov:
            # Left side out of view
            # Map angle from [-pi, -half_fov] to padding position [0, PADDING]
            # When angle = -pi (directly behind), position = 0 (far left)
            # When angle = -half_fov (edge of FOV), position = PADDING (near image)
            angle_range = -half_fov - (-np.pi)  # range of angles on left
            angle_offset = horizontal_angle - (-np.pi)  # offset from -pi
            ratio = angle_offset / angle_range  # 0 (at -pi) to 1 (at -half_fov)
            u_padded = int(ratio * PADDING)
        elif horizontal_angle > half_fov:
            # Right side out of view
            # Map angle from [half_fov, pi] to padding position [width+PADDING, width+2*PADDING]
            # When angle = half_fov (edge of FOV), position = width+PADDING (near image)
            # When angle = pi (directly behind), position = width+2*PADDING (far right)
            angle_range = np.pi - half_fov  # range of angles on right
            angle_offset = horizontal_angle - half_fov  # offset from half_fov
            ratio = angle_offset / angle_range  # 0 (at half_fov) to 1 (at pi)
            u_padded = int(width + PADDING + ratio * PADDING)
        else:
            # Within FOV but failed projection (e.g., behind camera with small angle)
            # This shouldn't happen often, but handle gracefully
            if right >= 0:
                u_padded = width + PADDING + PADDING // 2
            else:
                u_padded = PADDING // 2
        
        # Vertical position: use middle of image
        v = height // 2
        color = (0, 165, 255)  # Orange in BGR for out-of-view
        marker_type = cv2.MARKER_TRIANGLE_UP
        
        # angle_deg = np.rad2deg(horizontal_angle)
        # logger.info(f"Waypoint {wp_i} out of view (angle={angle_deg:.1f}°, right={right:.2f}, fwd={fwd:.2f}) - shown at u={u_padded}")
    
    # Draw waypoint marker
    cv2.drawMarker(
        rgb_padded, 
        (u_padded, v), 
        color=color,
        markerType=marker_type,
        markerSize=20,
        thickness=3
    )
    
    # Add text label
    font_scale = 0.5
    text = f"WP{wp_i}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_x = u_padded - text_width // 2
    text_y = v - 20
    cv2.putText(
        rgb_padded,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2
    )
    
    return rgb_padded, [u_padded, v]


def find_closest_visible_pixel(reference_waypoint, agent, config, camera_matrix):
    """
    Find the closest visible pixel to a reference waypoint by projecting all pixels to 3D.
    
    This function:
    1. Projects all pixels in the depth image to 3D waypoints in world frame
    2. Calculates distances from each projected waypoint to the reference waypoint
    3. Returns the pixel with the minimum distance
    
    Args:
        reference_waypoint: (3,) array-like [x, y, z] in world frame (meters)
        agent: IF_Agent instance with depth image
        config: Configuration object
        camera_matrix: Camera intrinsics matrix
        
    Returns:
        tuple: (best_pixel, best_waypoint, min_distance)
            - best_pixel: [u, v] pixel coordinates of closest point
            - best_waypoint: [x, y, z] world coordinates of closest point
            - min_distance: distance in meters to reference waypoint
    """
    # Get current agent state
    agent_state = agent.envs.habitat_env.sim.get_agent_state()
    world_pos = agent_state.position
    world_rot = agent_state.rotation
    
    # Extract raw depth from observation (OrderedDict with 'depth' key)
    # Depth is stored as (H, W, 1) in meters
    depth_image = agent.observation['depth'][:, :, 0]  # Shape: (H, W) in meters
    
    H, W = depth_image.shape[:2]
    
    # Create a grid of all pixel coordinates
    u_coords = np.arange(W)
    v_coords = np.arange(H)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    
    # Flatten to get all pixel coordinates as (N, 2)
    all_pixels = np.stack([u_grid.flatten(), v_grid.flatten()], axis=1)  # (H*W, 2)
    
    # Project all pixels to agent frame waypoints
    # pixel_to_agent_wp expects pixels in (N, 2) format and depth in (H, W)
    all_waypoints_agent = pixel_to_agent_wp(
        pixel=all_pixels,
        depth_image=depth_image,
        camera_matrix=camera_matrix,
        sensor_height=config.mapping.camera_height,
        camera_elevation_degree=0,
        device='cpu',
        depth_unit='m',
        direction='front'
    )  # Returns (N, 3) in agent frame
    
    # Convert all agent frame waypoints to world frame
    all_waypoints_world = []
    for wp_agent in all_waypoints_agent:
        wp_world = agent_to_world(wp_agent, world_pos, world_rot)
        all_waypoints_world.append(wp_world)
    all_waypoints_world = np.array(all_waypoints_world)  # (N, 3)
    
    # Calculate distances to reference waypoint
    reference_waypoint = np.asarray(reference_waypoint, dtype=np.float32)
    distances = np.linalg.norm(all_waypoints_world - reference_waypoint, axis=1)  # (N,)
    
    # Find the pixel with minimum distance
    min_idx = np.argmin(distances)
    best_pixel = all_pixels[min_idx]  # [u, v]
    best_waypoint = all_waypoints_world[min_idx]  # [x, y, z]
    min_distance = distances[min_idx]
    
    return best_pixel, best_waypoint, min_distance
 

def project_and_visualize_waypoint_2(point, wp_i, agent, config, camera_matrix, PADDING = Padding):
    """
    Project world waypoint to pixel coordinates and visualize it on RGB image with padding.
    If the waypoint is in view, it uses the closest visible pixel selection method.
    If out of view, it places a marker in the padding based on the angle.
    
    Args:
        point: World coordinates of the waypoint [x, y, z]
        wp_i: Waypoint index
        agent: IF_Agent instance
        config: Configuration object
        camera_matrix: Camera intrinsics matrix
        
    Returns:
        tuple: (rgb_with_waypoint, waypoint_pixel, selected_waypoint_world)
            - rgb_with_waypoint: RGB image with waypoint marker and padding
            - waypoint_pixel: Pixel coordinates [u, v] of the marker
            - selected_waypoint_world: World coordinates of the selected waypoint (if in view)
    """
    
    # Get current agent state
    agent_state = agent.envs.habitat_env.sim.get_agent_state()

    # Check if the original waypoint is directly visible
    waypoint_pixel_direct = world_to_pixel(
        world_wp=point,
        world_pos=agent_state.position,
        world_rot=agent_state.rotation,
        camera_matrix=camera_matrix,
        sensor_height=config.mapping.camera_height,
        camera_elevation_degree=0,
        image_height=config.mapping.env_frame_height,
        image_width=config.mapping.env_frame_width,
    )

    # Add padding to RGB image
    rgb_base = agent.rgb_vis.copy()
    height, width = rgb_base.shape[:2]
    rgb_padded = np.ones((height, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
    rgb_padded[:, PADDING:PADDING + width, :] = rgb_base

    best_pixel, best_waypoint, min_distance = None, None, None

    if waypoint_pixel_direct is not None:
        # Waypoint is in view: find the closest visible pixel
        best_pixel, best_waypoint, min_distance = find_closest_visible_pixel(
            reference_waypoint=point,
            agent=agent,
            config=config,
            camera_matrix=camera_matrix
        )
        
        u, v = int(best_pixel[0]), int(best_pixel[1])
        u_padded = u + PADDING
        color = (0, 0, 255)  # Red in BGR
        marker_type = cv2.MARKER_CROSS
        # logger.info(f"Waypoint {wp_i} in view. Closest visible at pixel ({u}, {v}), distance={min_distance:.3f}m")

    else:
        # Waypoint is out of view: use angular position for padding
        wp_agent = world_to_agent(point, agent_state.position, agent_state.rotation)
        right, fwd, up = float(wp_agent[0]), float(wp_agent[1]), float(wp_agent[2])
        
        horizontal_angle = np.arctan2(right, fwd)
        hfov_rad = np.deg2rad(config.mapping.hfov)
        half_fov = hfov_rad / 2.0
        
        if horizontal_angle < -half_fov:
            angle_range = -half_fov - (-np.pi)
            angle_offset = horizontal_angle - (-np.pi)
            ratio = angle_offset / angle_range
            u_padded = int(ratio * PADDING)
        elif horizontal_angle > half_fov:
            angle_range = np.pi - half_fov
            angle_offset = horizontal_angle - half_fov
            ratio = angle_offset / angle_range
            u_padded = int(width + PADDING + ratio * PADDING)
        else:
            u_padded = PADDING // 2 if right < 0 else width + PADDING + PADDING // 2

        v = height // 2
        color = (0, 165, 255)  # Orange in BGR for out-of-view
        marker_type = cv2.MARKER_TRIANGLE_UP
        
        # angle_deg = np.rad2deg(horizontal_angle)
        # logger.info(f"Waypoint {wp_i} out of view (angle={angle_deg:.1f}°) - shown at u={u_padded}")
        
        # For out-of-view points, the "best_pixel" is the padded coordinate
        best_pixel = [u_padded, v]

    # Draw waypoint marker
    cv2.drawMarker(
        rgb_padded, 
        (int(best_pixel[0] if best_pixel is not None and waypoint_pixel_direct is None else u_padded), int(best_pixel[1] if best_pixel is not None and waypoint_pixel_direct is None else v)),
        color=color,
        markerType=marker_type,
        markerSize=20,
        thickness=3
    )
    
    # Add text label
    label_text = f"WP{wp_i}"
    if min_distance is not None:
        label_text += f" ({min_distance:.2f}m)"
    
    font_scale = 0.5
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    
    # Determine the base x for the marker
    marker_x = int(best_pixel[0] if best_pixel is not None and waypoint_pixel_direct is None else u_padded)
    marker_y = int(best_pixel[1] if best_pixel is not None and waypoint_pixel_direct is None else v)

    text_x = marker_x - text_width // 2
    text_y = marker_y - 20

    cv2.putText(
        rgb_padded,
        label_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2
    )
    
    # Return the selected pixel and world coordinates
    return rgb_padded, best_pixel, best_waypoint


def check_pixel_on_floor(pixel, floor_mask):
    """
    Check if a pixel location is on the floor using a floor segmentation mask.
    
    Args:
        pixel: [u, v] pixel coordinates
        floor_mask: (H, W) binary mask where 1=floor, 0=other
        
    Returns:
        bool: True if pixel is on floor, False otherwise
    """
    u, v = int(pixel[0]), int(pixel[1])
    H, W = floor_mask.shape[:2]
    
    # Check bounds
    if not (0 <= u < W and 0 <= v < H):
        return False
    
    # Check if the pixel is on floor
    return floor_mask[v, u] == 1


def check_waypoint_visibility(wp_world, agent_state, config, camera_matrix, depth_image, MIN_DISTANCE_=0.4):
    """
    Check if a world waypoint is visible in the current camera view and on the floor.
    
    Args:
        wp_world: World coordinates of the waypoint [x, y, z]
        agent_state: Agent state object (position, rotation)
        config: Configuration object
        camera_matrix: Camera intrinsics matrix
        depth_image: Depth image (H, W) in meters
        floor_mask: (H, W) binary mask where 1=floor, 0=other (optional)
        MIN_DISTANCE_: Distance threshold for visibility check
        
    Returns:
        tuple: (is_visible, pixel, min_dist)
            - is_visible: Boolean indicating visibility AND floor presence
            - pixel: [u, v] pixel coordinates if visible and on floor, else [-1, -1]
            - min_dist: Distance between reprojected point and target point
    """
    # 1. Project 3D waypoint to 2D pixel
    wp_world_copy = wp_world.copy()
    wp_world_copy[1] -= CAMERA_HEIGHT_OFFSET  # Adjust for camera height offset
    
    pixel, wp_agent = world_to_pixel(
        world_wp=wp_world_copy,
        world_pos=agent_state.position,
        world_rot=agent_state.rotation,
        camera_matrix=camera_matrix,
        sensor_height=config.mapping.camera_height,
        camera_elevation_degree=0,
        image_height=config.mapping.env_frame_height,
        image_width=config.mapping.env_frame_width,
    )
    
    is_visible = False
    u_vis, v_vis = -1, -1
    min_dist = float('inf')
    
    if pixel is not None:
        u_center, v_center = int(pixel[0]), int(pixel[1])
        
        # Check if pixel is within image bounds
        if 0 <= u_center < config.mapping.env_frame_width and 0 <= v_center < config.mapping.env_frame_height:
            # 2. Check the specific projected pixel
            pixel_to_check = np.array([[u_center, v_center]])

            # Reproject pixel to 3D in robot coordinate
            wp_agent_reproj = pixel_to_agent_wp(
                pixel=pixel_to_check,
                depth_image=depth_image,
                camera_matrix=camera_matrix,
                sensor_height=config.mapping.camera_height,
                camera_elevation_degree=0, device='cpu', depth_unit='m', direction='front'
            )
            
            # Convert target waypoint to agent frame
            wp_target_agent = world_to_agent(
                wp_world_copy, world_pos=agent_state.position, world_rot=agent_state.rotation,
            )
            
            # Calculate distance
            min_dist = np.linalg.norm(wp_agent_reproj - wp_target_agent)
            
            # Threshold for visibility (relaxed to 0.5m)
            if min_dist < MIN_DISTANCE_:
                is_visible = True
                u_vis, v_vis = u_center, v_center
                
    return is_visible, [u_vis, v_vis], min_dist


def safe_bev_mapping(bev_map, rgbd, infos, envs, **kwargs):
    """Run BEV mapping but swallow sporadic depth preprocessing errors."""
    try:
        return bev_map.mapping(rgbd, infos, envs, **kwargs)
    except (TypeError, ValueError) as err:
        save_idx = kwargs.get("save_idx", "unknown")
        logger.warning(
            f"Skipping BEV map update at step {save_idx} because depth preprocessing failed: {err}"
        )
        return None


# 1) first consider turn_left/right 2) first consider farthest
def project_and_visualize_waypoint_v3v4(agent, config, camera_matrix, current_step_idx, target_waypoint, reference_path_for_v3, rgbd, current_action_id=None, reference_actions=None, PADDING = Padding):
    """
    Project all visible and next waypoints to the image based on GT trajectory.
    
    Args:
        agent: IF_Agent instance
        config: Configuration object
        camera_matrix: Camera intrinsics matrix
        current_step_idx: Current step index in the episode
        target_waypoint: The current target waypoint from the high-level path
        current_action_id: The action just taken (used for immediate left/right cues)
        reference_actions: List of future action ids aligned with reference path
        
    Returns:
        tuple: (rgb_vis, visible_waypoints_info)
            - rgb_vis: RGB image with visible waypoints marked
            - visible_waypoints_info: List of dicts with info about visible waypoints
    """
    
    rgb_padded_v3, visible_waypoints_info_3 = None, None

    # Pad image for visualization
    rgb_vis = agent.rgb_vis.copy()
    height, width = rgb_vis.shape[:2]
    rgb_padded = np.ones((height, width + 2 * PADDING, 3), dtype=rgb_vis.dtype) * 128
    rgb_padded[:, PADDING:PADDING + width, :] = rgb_vis


    # Immediate left/right cue based on current action for verson 3
    if current_action_id in (2, 3):
        rgb_padded_v3 = rgb_padded.copy()
        u_final = 0 if current_action_id == 2 else width + PADDING * 2
        v_final = (height + 2 * PADDING) // 2
        cv2.circle(rgb_padded_v3, (u_final, v_final), 2, (0, 165, 255), -1)
        visible_waypoints_info_3 = [{
            "future_step_idx": current_step_idx,
            "pixel": [u_final, v_final],
            "world_pos": target_waypoint.tolist() if hasattr(target_waypoint, 'tolist') else target_waypoint,
            "distance": -1.0,
            "source": "current_action"
        }]

    visible_waypoints_info = []
    
    # We assume current_step_idx aligns with the steps in the JSON
    future_waypoints = reference_path_for_v3[current_step_idx + 1:]
    agent_state = agent.envs.habitat_env.sim.get_agent_state()
    depth_image = rgbd[:, :, 3] # Depth is stored as (H, W, 1) in meters

    visible_candidates = []
    consecutive_invisible_count = 0

    # count the visible waypoints
    for i, wp_world in enumerate(future_waypoints):
        is_visible, pixel_vis, min_dist = check_waypoint_visibility(wp_world, agent_state, config, camera_matrix, depth_image, MIN_DISTANCE_=MIN_DISTANCE_v3)

        if is_visible:
            agent_dist = float(np.linalg.norm(np.asarray(wp_world) - agent_state.position))
            visible_candidates.append({
                "future_step_idx": current_step_idx + 1 + i,
                "pixel": pixel_vis,
                "world_pos": wp_world.tolist(),
                "distance": float(min_dist),
                "agent_distance": agent_dist
            })
            consecutive_invisible_count = 0
        else:
            if visible_candidates:
                consecutive_invisible_count += 1
                if consecutive_invisible_count >= MAX_INVISIBLE_GAP:
                    break

    if visible_candidates:  # for visiable waypoints
        # get the index of the farthest visible waypoint
        farthest_idx = max(
            range(len(visible_candidates)),
            key=lambda idx: visible_candidates[idx]["agent_distance"]
        )
        if farthest_idx != 1:
            second_farthest_idx = farthest_idx - 1
        else:
            second_farthest_idx = farthest_idx

        # just for visualization, mark all visible candidates and highlight the farthest one
        for idx, cand in enumerate(visible_candidates): # visualzie each pixel on the padded image
            u_vis, v_vis = cand["pixel"]
            u_final = u_vis + PADDING
            v_final = v_vis
            color = (0, 0, 255) if idx == farthest_idx else (0, 255, 0)
            cv2.circle(rgb_padded, (u_final, v_final), 2, color, -1)
            cand["pixel"] = [u_final, v_final]

        selected_cand = visible_candidates[second_farthest_idx]
        selected_cand["source"] = "visible_farthest"
        visible_waypoints_info.append(selected_cand)
        if visible_waypoints_info_3 is not None:
            return rgb_padded_v3, visible_waypoints_info_3, rgb_padded, visible_waypoints_info
        else:
            return rgb_padded, visible_waypoints_info, rgb_padded, visible_waypoints_info

    else:   # for steps with no visible waypoint
        # First, consider current acion is turn_left or turn right
        next_turn_idx = None
        next_turn_action = None
        if reference_actions is not None and len(reference_actions) > current_step_idx + 1:
            for future_idx in range(current_step_idx + 1, len(reference_actions)):
                act = reference_actions[future_idx]
                if act in (2, 3):
                    next_turn_idx = future_idx
                    next_turn_action = act
                    break
        mean_point_world = np.asarray(target_waypoint)
        if next_turn_action in (2, 3):
            u_final = 0 if next_turn_action == 2 else width + PADDING * 2
            v_final = (height + 2 * PADDING) // 2
            cv2.circle(rgb_padded, (u_final, v_final), 2, (0, 165, 255), -1)
            visible_waypoints_info.append({
                "future_step_idx": next_turn_idx if next_turn_idx is not None else -1,
                "pixel": [u_final, v_final],
                "world_pos": mean_point_world.tolist() if hasattr(mean_point_world, 'tolist') else mean_point_world,
                "distance": -1.0,
                "source": "future_action_padding"
            })
            if rgb_padded_v3 is not None:
                return rgb_padded_v3, visible_waypoints_info_3, rgb_padded, visible_waypoints_info
            else:
                return rgb_padded, visible_waypoints_info, rgb_padded, visible_waypoints_info

        # Second: use geometry to decide left/right, consider next five waypoint, and determine whether it is the right or left
        distinct_points = []
        last_kept = None
        for wp_world in future_waypoints:
            wp_arr = np.asarray(wp_world)
            if last_kept is None or np.linalg.norm(wp_arr - last_kept) >= 0.05:
                distinct_points.append(wp_arr)
                last_kept = wp_arr
            if len(distinct_points) >= 5:
                break
        if len(distinct_points) > 0:
            mean_point_world = np.mean(distinct_points, axis=0)

        wp_agent_check = world_to_agent(mean_point_world, agent_state.position, agent_state.rotation)
        right = float(wp_agent_check[0])

        v_final = (height + 2 * PADDING) // 2
        if right > 0:
            u_final = width + PADDING * 2
        else:
            u_final = 0

        cv2.circle(rgb_padded, (u_final, v_final), 2, (0, 165, 255), -1)

        visible_waypoints_info.append({
            "future_step_idx": -1,
            "pixel": [u_final, v_final],
            "world_pos": mean_point_world.tolist() if hasattr(mean_point_world, 'tolist') else mean_point_world,
            "distance": -1.0,
            "source": "fallback_geometry_padding"
        })

        if rgb_padded_v3 is not None:
            return rgb_padded_v3, visible_waypoints_info_3, rgb_padded, visible_waypoints_info
        else:
            return rgb_padded, visible_waypoints_info, rgb_padded, visible_waypoints_info
            


def execute_exp(config: "DictConfig") -> None:
    # Set seeds for reproducibility
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)

    config = modify_config(config)

    logger.info("Final Config Passed to Env and Agent:\n" + OmegaConf.to_yaml(config, resolve=True))

    # Output organization
    if config.habitat.dataset.type == "R2RVLN-v2":
        output_dir = (f"../training_data/data_mp3d_r2r/"
                    f"{config.habitat.dataset.data_path.split('/')[-3]}/"
                    f"{config.habitat.dataset.split}_pixel")
    elif config.habitat.dataset.type == "RxRVLN-v2":
        output_dir = (f"../training_data/data_mp3d_rxr/"
                f"{config.habitat.dataset.data_path.split('/')[-3]}_{config.habitat.dataset.ROLES[0]}_{config.habitat.dataset.LANGUAGES[0]}/"
                f"{config.habitat.dataset.split}_pixel")
    else:
        raise ValueError(f"Unsupported dataset type: {config.habitat.dataset.type}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Writing images and json to {output_dir} for VLM training.")


    # construct the environment
    envs = construct_envs(config)

    # construct the agent
    agent = IF_Agent(config, envs)

    possible_action_env = envs.habitat_env.task.actions.keys()
    # success_dist = config.habitat.task.measurements.success.success_distance
    follower = ShortestPathFollower(
        envs.habitat_env.sim,
        goal_radius=max(1e-3, SUCCESS_DISTANCE - 1e-3),
        # goal_radius=max(1e-3, 0.2 - 1e-3),
        return_one_hot=False
    )

    # Get camera intrinsics for pixel projection
    camera_matrix = get_camera_matrix(
        config.mapping.frame_width,
        config.mapping.env_frame_height,
        config.mapping.hfov
    )

    start_time = time.time()

    # Setup for history RGBD accumulation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, episode in enumerate(envs.habitat_env.episodes):

        episode_start_time = time.time()
        if i == NUMBER_OF_EPISODES:
            break
        
        # reset the environment
        obs, rgbd, infos = agent.reset(output_dir)

        if IF_RESUME:
            if check_exist(output_dir, agent.scene_id, agent.ep_id):
                logger.info(f"Skipping existing episode: Episode {i} / {envs.number_of_episodes} - {agent.scene_id}/{agent.ep_id}")
                continue

        if IF_SCENE_FILTER:
            if agent.scene_id not in SCENE_LIST:
                logger.info(f"Skipping episode not in SCENE_LIST: Episode {i} / {envs.number_of_episodes} - {agent.scene_id}")
                continue


        current_episode = envs.habitat_env.current_episode
        # Initialize history buffer for this episode
        
        path = current_episode.reference_path + [current_episode.goals[0].position]

        save_idx = 0


        
        # Load GT trajectory  
        if config.habitat.dataset.type == "R2RVLN-v2":
            json_path = f"../training_data/data_mp3d_r2r/{config.habitat.dataset.data_path.split('/')[-3]}/train_trajectory/adjusted_annotations/{agent.scene_id}/{agent.ep_id}.json"
        elif config.habitat.dataset.type == "RxRVLN-v2":
            json_path = (f"../training_data/data_mp3d_rxr/"
                  f"{config.habitat.dataset.data_path.split('/')[-3]}_{config.habitat.dataset.ROLES[0]}_{config.habitat.dataset.LANGUAGES[0]}/"
                  f"{config.habitat.dataset.split}_trajectory/adjusted_annotations/{agent.scene_id}/{agent.ep_id}.json")
        else:
            raise ValueError(f"Unsupported dataset type: {config.habitat.dataset.type}")
        
        
        if not os.path.exists(json_path):
            logger.warning(f"Annotation file not found: {json_path}")
            # raise FileNotFoundError(f"Annotation file not found: {json_path}")
            continue
        try:
            with open(json_path, 'r') as f:
                annot_data = json.load(f)
        except Exception as e:
            raise e
        
        reference_path_for_v3 = None
        if IF_vis_34:
            # Get reference path (list of agent positions) and align action names to ids for later use
            reference_path_for_v3 = [np.array(step['agent_position']) for step in annot_data['steps']]

            action_names = list(possible_action_env)
            action_name_to_id = {name: idx for idx, name in enumerate(action_names)}
            reference_actions = []
            for step in annot_data.get("steps", []):
                name = step.get("action_name")
                if name in action_name_to_id:
                    reference_actions.append(action_name_to_id[name])
                elif step.get("action_id") is not None:
                    reference_actions.append(int(step["action_id"]))

        # keep episode_data always defined
        episode_data = {
            "episode_id": agent.ep_id,
            "scene_id": agent.scene_id,
            "instruction": current_episode.instruction.instruction_text,
            "steps": []
        }

        # INIT step logging
        agent_state = envs.habitat_env.sim.get_agent_state()
        step_data = {
            "step_idx": save_idx,
            "agent_position": agent_state.position.tolist(),
            "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z],
            "action_id": None,
            "action_name": "INIT",
            "instruction": current_episode.instruction.instruction_text
        }

        episode_data["steps"].append(step_data)
        save_idx += 1
        episode_done = False
        last_infos = {}
        episode_steps = 0  # counts env steps (including STOP)

        for wp_i, point in enumerate(path):
            if episode_done or envs.habitat_env.episode_over:
                break

            is_final_wp = (wp_i == len(path) - 1)
            wp_reached = False
            steps_this_wp = 0
            
            # get camera intrinsics
            camera_matrix = get_camera_matrix(
                config.mapping.frame_width,
                config.mapping.env_frame_height,
                config.mapping.hfov
            )

            while not wp_reached and not episode_done:
                if envs.habitat_env.episode_over:
                    episode_done = True
                    break

                best_action = follower.get_next_action(point)

                # Reached this waypoint (or follower says STOP) Last action
                if best_action is None or best_action == HabitatSimActions.stop:
                    if is_final_wp:
                        # Step STOP to terminate the episode within success radius
                        obs, rgbd, done, infos = agent.step(HabitatSimActions.stop)
                        last_infos = infos
                        episode_done = bool(done)
                        episode_steps += 1

                        # Stop step logging
                        agent_state = envs.habitat_env.sim.get_agent_state()
                        step_data = {
                            "step_idx": save_idx,
                            "agent_position": agent_state.position.tolist(),
                            "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z],
                            "action_id": int(HabitatSimActions.stop),
                            "action_name": "STOP",
                            "waypoint_id": wp_i,
                            "reference_waypoint": point.tolist() if hasattr(point, 'tolist') else list(point),
                            "instruction": current_episode.instruction.instruction_text
                        }


                        episode_data["steps"].append(step_data)
                        save_idx += 1

                        info_for_log = last_infos or {}
                        logger.info(
                            f"Episode {i} / {envs.number_of_episodes}-{agent.scene_id}-{agent.ep_id}: " 
                            f"TL: {info_for_log.get('path_length')}, "
                            f"NE: {info_for_log.get('oracle_navigation_error')}, "
                            f"SR: {info_for_log.get('success')}, "
                            f"OS: {info_for_log.get('oracle_success')}, "
                            f"SPL: {info_for_log.get('spl')}, "
                            f"Instruction: {current_episode.instruction.instruction_text}"
                        )
                    wp_reached = True
                    break

                # Normal step (take the follower's suggested action)
                action_id = int(best_action)
                action_name = list(possible_action_env)[action_id]
                obs, rgbd, done, infos = agent.step(action_id)
                last_infos = infos
                episode_done = bool(done)
                episode_steps += 1
                

                # Perform pixel visualization based on selected method
                if IF_vis_1:
                    rgb_with_waypoint_1, waypoint_pixel_1 = project_and_visualize_waypoint_1(
                        point, wp_i, agent, config, camera_matrix
                    )
                if IF_vis_2:
                    rgb_with_waypoint_2, waypoint_pixel_2, selected_waypoint_world_2 = project_and_visualize_waypoint_2(
                        point, wp_i, agent, config, camera_matrix
                    )
                if IF_vis_34:
                    rgb_with_waypoint_3, visible_waypoints_info_3, rgb_with_waypoint_4, visible_waypoints_info_4 = project_and_visualize_waypoint_v3v4(
                        agent,config,camera_matrix,save_idx,point,reference_path_for_v3,rgbd=rgbd,current_action_id=reference_actions[save_idx],
                        reference_actions=reference_actions
                    )


                if not episode_done:
                    
                    # Main loop logging
                    agent_state = envs.habitat_env.sim.get_agent_state()
                    step_data = {
                        "step_idx": save_idx,
                        "agent_position": agent_state.position.tolist(),
                        "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z],
                        "action_id": action_id,
                        "action_name": action_name,
                        "waypoint_id": wp_i,
                        "reference_waypoint": point.tolist() if hasattr(point, 'tolist') else list(point),
                        "instruction": current_episode.instruction.instruction_text
                    }

                    # Save pixel visualization data
                    if WRITE_PIXEL_VISUALIZATION:
                        if IF_vis_1:
                            rgb_wp_path_1 = f"{agent.paths['generated_pixel_1']}/{save_idx:03d}.jpg"
                            cv2.imwrite(rgb_wp_path_1, rgb_with_waypoint_1)
                            
                            json_path_1 = f"{agent.paths['generated_pixel_1_json']}/{save_idx:03d}.json"
                            save_dict_1 = {
                                "x": int(waypoint_pixel_1[0]),
                                "y": int(waypoint_pixel_1[1]),
                                "instruction": current_episode.instruction.instruction_text,
                                "waypoint_world": point.tolist() if hasattr(point, 'tolist') else list(point),
                                "agent_position": agent_state.position.tolist(),
                                "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z]
                            }
                            with open(json_path_1, 'w') as f:
                                json.dump(save_dict_1, f, indent=4)
                            
                            step_data["rgb_with_waypoint_path_1"] = rgb_wp_path_1
                            step_data["waypoint_pixel_1"] = {"x": int(waypoint_pixel_1[0]), "y": int(waypoint_pixel_1[1])}
                            step_data["waypoint_world_1"] = point.tolist() if hasattr(point, 'tolist') else list(point)

                        if IF_vis_2:
                            rgb_wp_path_2 = f"{agent.paths['generated_pixel_2']}/{save_idx:03d}.jpg"
                            cv2.imwrite(rgb_wp_path_2, rgb_with_waypoint_2)
                            
                            json_path_2 = f"{agent.paths['generated_pixel_2_json']}/{save_idx:03d}.json"
                            save_dict_2 = {
                                "x": int(waypoint_pixel_2[0]),
                                "y": int(waypoint_pixel_2[1]),
                                "instruction": current_episode.instruction.instruction_text,
                                "waypoint_world": selected_waypoint_world_2.tolist() if hasattr(selected_waypoint_world_2, 'tolist') else list(selected_waypoint_world_2) if selected_waypoint_world_2 is not None else None,
                                "agent_position": agent_state.position.tolist(),
                                "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z]
                            }
                            with open(json_path_2, 'w') as f:
                                json.dump(save_dict_2, f, indent=4)
                            
                            step_data["rgb_with_waypoint_path_2"] = rgb_wp_path_2
                            step_data["waypoint_pixel_2"] = {"x": int(waypoint_pixel_2[0]), "y": int(waypoint_pixel_2[1])} if waypoint_pixel_2 is not None else None
                            step_data["waypoint_world_2"] = selected_waypoint_world_2.tolist() if hasattr(selected_waypoint_world_2, 'tolist') else list(selected_waypoint_world_2) if selected_waypoint_world_2 is not None else None

                        if IF_vis_34:
                            rgb_wp_path_3 = f"{agent.paths['generated_pixel_3']}/{save_idx:03d}.jpg"
                            cv2.imwrite(rgb_wp_path_3, rgb_with_waypoint_3)
                            
                            selected_wp_3 = visible_waypoints_info_3[-1]
                            
                            step_data["rgb_with_waypoint_path_3"] = rgb_wp_path_3
                            step_data["visible_waypoints_3"] = visible_waypoints_info_3
                            if selected_wp_3:
                                step_data["waypoint_pixel_3"] = {
                                    "x": int(selected_wp_3["pixel"][0]),
                                    "y": int(selected_wp_3["pixel"][1])
                                }
                                step_data["waypoint_world_3"] = selected_wp_3["world_pos"]
                            else:
                                raise ValueError(f"No visible waypoint found for v3 at step {save_idx} in episode {agent.ep_id} of scene {agent.scene_id}, but IF_vis_34 is True.")

                        if IF_vis_34:
                            rgb_wp_path_4 = f"{agent.paths['generated_pixel_4']}/{save_idx:03d}.jpg"
                            cv2.imwrite(rgb_wp_path_4, rgb_with_waypoint_4)
                            
                            selected_wp_4 = visible_waypoints_info_4[-1]
                            
                            step_data["rgb_with_waypoint_path_4"] = rgb_wp_path_4
                            step_data["visible_waypoints_4"] = visible_waypoints_info_4
                            if selected_wp_4:
                                step_data["waypoint_pixel_4"] = {
                                    "x": int(selected_wp_4["pixel"][0]),
                                    "y": int(selected_wp_4["pixel"][1])
                                }
                                step_data["waypoint_world_4"] = selected_wp_4["world_pos"]
                            else:
                                raise ValueError(f"No visible waypoint found for v4 at step {save_idx} in episode {agent.ep_id} of scene {agent.scene_id}, but IF_vis_34 is True.")

                    episode_data["steps"].append(step_data)
                    save_idx += 1

                steps_this_wp += 1
                if steps_this_wp >= MAX_STEPS_PER_WP:
                    logger.info(
                        f"Episode {i} / {envs.number_of_episodes}: "
                        f"Waypoint safety cap hit ({MAX_STEPS_PER_WP} steps) at step {save_idx}."
                    )
                    # Fail the waypoint gracefully; move on or end episode per your policy
                    wp_reached = True
                    # Optionally: episode_done = True
                    break

            if episode_done:
                # Already logged after STOP; optional guard log:
                info_for_log = last_infos or {}
                break

        episode_data["episode_length"] = int(episode_steps)  # includes STOP if issued
        episode_data["final_metrics"] = {
            "distance_to_goal": last_infos.get("distance_to_goal"),
            "soft_spl": last_infos.get("soft_spl"),
            "path_length": last_infos.get("path_length"),
            "oracle_navigation_error": last_infos.get("oracle_navigation_error"),
            "success": last_infos.get("success"),
            "oracle_success": last_infos.get("oracle_success"),
            "spl": last_infos.get("spl"),
        }

        out_file = f"{output_dir}/annotations/{agent.scene_id}/{agent.ep_id}.json"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(episode_data, f, indent=4)
        
        record_finished_episode(output_dir, agent.scene_id, agent.ep_id, time.time() - episode_start_time)


    end_time = time.time()
    elapsed_time = end_time - start_time
    time_file = os.path.join(output_dir, "execution_time.txt")
    with open(time_file, "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds saved to {time_file}")

if __name__ == "__main__":
    # register custom plugins
    register_plugins()
    # register the config plugin for habitat-baselines MP3D-HDT
    register_hydra_plugin(HabitatBaselinesConfigPlugin_BC)

    if "--exp-config" in sys.argv or "--run-type" in sys.argv:
        raise ValueError(
            "The API of run.py has changed to be compatible with hydra.\n"
            "--exp-config is now --config-name and is a config path inside habitat-baselines/habitat_baselines/config/. \n"
            "--run-type train is replaced with habitat_baselines.evaluate=False (default) and --run-type eval is replaced with habitat_baselines.evaluate=True.\n"
            "instead of calling:\n\n"
            "python -u -m habitat_baselines.run --exp-config habitat-baselines/habitat_baselines/config/<path-to-config> --run-type train/eval\n\n"
            "You now need to do:\n\n"
            "python -u -m habitat_baselines.run --config-name=<path-to-config> habitat_baselines.evaluate=False/True\n"
        )
    main()
