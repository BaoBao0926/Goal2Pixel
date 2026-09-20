import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch
from decord import VideoReader, cpu
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Sequence, Tuple, List, Union, Dict

from habitat_sim.utils.common import d3_40_colors_rgb
from habitat.utils.visualizations.utils import append_text_underneath_image
from habitat.utils.visualizations import maps as habitat_vis_maps
from habitat import logger

from scripts.run_utils.constant import mp3d_category
from scripts.run_utils.tools import _ensure_dir, _to_hwc_uint8, _to_numpy
from scripts.run_utils.mapping.visualization import (plot_local_occupancy_with_pose_pil,
                                                     plot_local_occupency_explore_with_pose_pil,
                                                     overlay_frontier_centroids_on_rotated_map,
                                                     select_and_visualize_frontier,
                                                     )
from scripts.run_utils.mapping.mapping_utils import accumulate_history_rgbd_to_panoramic
from task_patch.utils.vln_ce_utils import observations_to_image
from task_patch.mp3d_register import maps as vln_maps

def write_top_down_images(save_idx,
                          top_down_map_path,
                          observations,
                          infos,
                          goal_pixel=None,
                          goal_wp=None,
                          ):

    frame = observations_to_image(observations, infos)
    if goal_wp is not None:
        frame = _overlay_waypoint_on_topdown(frame, observations, infos, goal_wp)
    if goal_pixel is not None:
        frame = _overlay_waypoint_on_rgb_depth(frame, goal_pixel)
    frame = append_text_underneath_image(
        frame, str(observations['instruction']['text']),
    )
    frame_pil = Image.fromarray(frame)
    frame_pil.save(os.path.join(top_down_map_path, f"{save_idx:03d}.png"))
    return frame_pil


def _overlay_waypoint_on_topdown(frame: np.ndarray,
                                 observations: Dict[str, np.ndarray],
                                 infos: Dict[str, Dict[str, np.ndarray]],
                                 goal_wp: Union[List[float], Tuple[float, ...], np.ndarray]) -> np.ndarray:
    map_info = _extract_topdown_map_info(infos)
    if map_info is None:
        return frame
    rendered = _render_topdown_with_goal(frame, observations, map_info, goal_wp)
    if rendered is None:
        return frame

    updated = frame.copy()
    if _is_panoramic_observation(observations):
        map_height = rendered.shape[0]
        updated[-map_height:, :, :] = rendered
    else:
        map_width = rendered.shape[1]
        updated[:, -map_width:, :] = rendered
    return updated


def _overlay_waypoint_on_rgb_depth(frame: np.ndarray,
                                   goal_pixel: Union[List[int], Tuple[int, int], np.ndarray]
                                   ) -> np.ndarray:
    """
    frame: (H, 512, 3) where left  half = RGB image  (256,256)
                                   right half = depth vis (256,256)
    goal_pixel: (u, v) pixel of waypoint in LEFT image.
    """
    vis = frame.copy()
    H, total_W, _ = frame.shape
    W = frame.shape[0]

    # Extract coordinates
    u, v = int(goal_pixel[0]), int(goal_pixel[1])

    # Clamp pixel inside one-half region
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    # Pixel on depth image is shifted by W
    u_depth = u + W

    # Draw small green dot on both RGB and depth images
    color = (0, 255, 0)
    radius = 4
    thickness = -1

    cv2.circle(vis, (u, v), radius, color, thickness)       # RGB
    cv2.circle(vis, (u_depth, v), radius, color, thickness) # depth colormap

    return vis

def _extract_topdown_map_info(infos: Dict[str, Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if "top_down_map_vlnce" in infos:
        return infos["top_down_map_vlnce"]
    if "top_down_map" in infos:
        return infos["top_down_map"]
    return None


def _is_panoramic_observation(observations: Dict[str, np.ndarray]) -> bool:
    if "rgb" in observations and len(observations["rgb"].shape) == 4:
        return True
    if "depth" in observations and len(observations["depth"].shape) == 4:
        return True
    return False


def _render_topdown_with_goal(frame: np.ndarray,
                              observations: Dict[str, np.ndarray],
                              map_info: Dict[str, np.ndarray],
                              goal_wp: Union[List[float], Tuple[float, ...], np.ndarray]) -> Optional[np.ndarray]:
    raw_map = map_info.get("map")
    if raw_map is None:
        return None

    td_map = np.array(raw_map, copy=True)
    goal_mask = _build_goal_mask(td_map.shape, map_info, goal_wp, scale_factor=1.0)

    td_map = vln_maps.colorize_topdown_map(
        td_map,
        map_info.get("fog_of_war_mask"),
        fog_of_war_desat_amount=0.75,
    )
    td_map = habitat_vis_maps.draw_agent(
        image=td_map,
        agent_center_coord=map_info.get("agent_map_coord"),
        agent_rotation=map_info.get("agent_angle"),
        agent_radius_px=max(1, min(td_map.shape[0:2]) // 24),
    )

    td_map, goal_mask = _rotate_topdown_if_needed(td_map, goal_mask)
    td_map, goal_mask = _resize_and_pad_topdown(td_map, goal_mask, frame.shape, _is_panoramic_observation(observations))

    if goal_mask is not None and np.any(goal_mask):
        dot_mask = cv2.dilate(goal_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        td_map = td_map.copy()
        td_map[dot_mask > 0] = np.array([255, 64, 64], dtype=np.uint8)
    return td_map


def _build_goal_mask(map_shape: Tuple[int, int],
                     map_info: Dict[str, np.ndarray],
                     goal_wp: Union[List[float], Tuple[float, ...], np.ndarray],
                     scale_factor: float = 1.0) -> Optional[np.ndarray]:
    """
    Builds a goal mask with a dynamically adjustable circle size.

    Args:
        map_shape: Shape of the map (height, width).
        map_info: Dictionary containing map metadata.
        goal_wp: Goal waypoint in world coordinates.
        scale_factor: Multiplier to increase the circle radius.

    Returns:
        A binary mask with the goal waypoint as a circle.
    """
    bounds = map_info.get("bounds")
    if bounds is None or goal_wp is None:
        return None

    wp = np.asarray(goal_wp).flatten()
    if wp.size < 2:
        return None

    world_x = float(wp[0])
    world_z = float(wp[2]) if wp.size >= 3 else float(wp[1])
    try:
        grid_x, grid_y = vln_maps.static_to_grid(world_z, world_x, map_shape, bounds)
    except Exception:
        return None

    if not (0 <= grid_x < map_shape[0] and 0 <= grid_y < map_shape[1]):
        return None

    mask = np.zeros(map_shape, dtype=np.uint8)
    meters_per_px = map_info.get("meters_per_px", 0.05) or 0.05
    base_radius = 0.25  # Base radius in meters
    radius_cells = max(1, int((base_radius * scale_factor) / meters_per_px))
    cv2.circle(mask, (grid_y, grid_x), radius_cells, 255, -1)
    return mask


def _rotate_topdown_if_needed(td_map: np.ndarray,
                              mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if td_map.shape[1] < td_map.shape[0]:
        td_map = np.rot90(td_map, 1)
        if mask is not None:
            mask = np.rot90(mask, 1)
    if td_map.shape[0] > td_map.shape[1]:
        td_map = np.rot90(td_map, 1)
        if mask is not None:
            mask = np.rot90(mask, 1)
    return td_map, mask


def _resize_and_pad_topdown(td_map: np.ndarray,
                            mask: Optional[np.ndarray],
                            frame_shape: Tuple[int, int, int],
                            is_pano: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    old_h, old_w = td_map.shape[:2]
    if is_pano:
        frame_width = frame_shape[1]
        top_down_width = max(1, frame_width // 3)
        top_down_height = max(1, int(top_down_width / old_w * old_h))
        td_map = cv2.resize(td_map, (top_down_width, top_down_height), interpolation=cv2.INTER_CUBIC)
        if mask is not None:
            mask = cv2.resize(mask, (top_down_width, top_down_height), interpolation=cv2.INTER_NEAREST)
        pad_width = frame_width - top_down_width
        if pad_width > 0:
            white = np.ones((top_down_height, pad_width, 3), dtype=np.uint8) * 255
            td_map = np.concatenate((white, td_map), axis=1)
            if mask is not None:
                pad = np.zeros((top_down_height, pad_width), dtype=np.uint8)
                mask = np.concatenate((pad, mask), axis=1)
        return td_map, mask

    target_h = frame_shape[0]
    target_w = max(1, int(float(target_h) / old_h * old_w))
    td_map = cv2.resize(td_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    if mask is not None:
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return td_map, mask


def get_r2r_episode_info_display(episode_data):
    """
    Extract episode info for R2R VLN visualization.
    Includes: instruction, action, navigation metrics (SR, SPL, OSR, etc.)
    """
    save_idx = episode_data.get('save_idx', 0)
    return {
        'save_idx': save_idx,
        'instruction': episode_data.get('instruction', 'N/A'),
        'action': episode_data.get('action_name', 'N/A'),
        'SR': episode_data.get('infos', {}).get('success', 'N/A'),
        'SPL': episode_data.get('infos', {}).get('spl', 'N/A'),
        'distance_to_goal': episode_data.get('infos', {}).get('distance_to_goal', 'N/A'),
        'OSR': episode_data.get('infos', {}).get('oracle_success', 'N/A'),
        'OSPL': episode_data.get('infos', {}).get('oracle_spl', 'N/A'),
        'NE': episode_data.get('infos', {}).get('oracle_navigation_error', 'N/A')
    }


def get_hdt_episode_info_display(episode_data):
    """
    Extract episode info for HDT object navigation visualization.
    Includes: object_category, action, navigation metrics (success, SPL, distance_to_goal)
    """
    save_idx = episode_data.get('save_idx', 0)
    return {
        'save_idx': save_idx,
        'object_category': episode_data.get('object_category', 'N/A'),
        'action': episode_data.get('action_name', 'N/A'),
        'success': episode_data.get('infos', {}).get('success', 'N/A'),
        'SPL': episode_data.get('infos', {}).get('spl', 'N/A'),
        'distance_to_goal': episode_data.get('infos', {}).get('distance_to_goal', 'N/A'),
    }


def write_all_images(episode_data, episode_info_display=None):
    """
    Write all visualization images for a single step.
    
    Args:
        episode_data: dict containing all visualization data
        episode_info_display: optional pre-formatted dict for episode info panel.
                            If None, will use get_r2r_episode_info_display as default.
    """
    
    save_idx = episode_data['save_idx']
    
    # frontier
    centroids = episode_data['frontier'][0]
    frontier_u8 = episode_data['frontier'][1]

    # mapping
    local_map = episode_data['mapping']['map'][0]
    planner_pose_inputs = episode_data['mapping']['map'][1]
    vis_config = episode_data['mapping']['config']
    map_resolution = episode_data['mapping']['resolution']

    # egocentric
    rgb_vis = episode_data['rgb_vis']
    depth_vis = episode_data['depth_vis']
    object_segmentation = episode_data['object_segmentation']

    paths = episode_data['save_paths']
    

    # rgb/s
    rgb_pil, depth_pil, semantic_pil = write_rgbds_images(save_idx,
                                                          paths['rgb'], paths['depth'], 
                                                          paths['semantic'], paths['semantic_room'],
                                                          rgb_vis, depth_vis,
                                                          object_segmentation,
                                                          # agent.seg_idx_region.squeeze(-1), 
                                                          # agent.seg_name_region.squeeze(-1),
                                                          )

    # map
    occ_pil, occ_exp_pil, occ_exp_ft_pil,  selected_frontier_pil, best_label, fov_pil = write_map_images(save_idx,
                                                            local_map, planner_pose_inputs,
                                                            paths['occupancy'], paths['occupancy_explored'],
                                                            paths['occupancy_frontier'], paths['occupancy_frontier_gt'],
                                                            episode_data['mapping']['resolution'],
                                                            centroids, frontier_u8,
                                                            vis_config=vis_config)


    # map + semantic (object+region)
    # obj_pil = write_semantic_bev_object(
    #     BEV_map.local_map, BEV_map.planner_pose_inputs[0],
    #     # agent.seg_idx_obj.squeeze(-1),
    #     # agent.seg_name_obj.squeeze(-1),
    #     out_path=os.path.join(agent.bev_path_object, f"{save_idx:03d}.png"),
    #     # env_idx=0,
    #     # ch_start_obj=4,           # set these two to your object head range
    #     # ch_count_obj=40,  # e.g., 80 if you have 80 object channels
    # )
    
    # top_down
    # top_down_pil = write_top_down_images(
    #     save_idx,
    #     paths['top_down_map'],
    #     agent.observation,
    #     agent.info,
    #     episode_data['goal_pixel'],
    #     episode_data['goal_wp']
    # )
    top_down_pil = save_top_down_map(
        save_idx,
        paths['top_down_map'],
        episode_data['infos']
    )
    
 
    combined_pil = create_combined_visualization(
        paths['combined_debug'],
        save_idx,
        rgb_pil, depth_pil, semantic_pil,
        occ_pil, occ_exp_pil, selected_frontier_pil, 
        top_down_pil=top_down_pil,
        episode_info=episode_info_display,
        fov_pil=fov_pil,
    )
    
    # Generate and save history-aggregated RGB if history data is provided
    history_data = episode_data.get('history_data')
    if history_data is not None:
        history_buffer = history_data['buffer']
        save_interval = history_data['save_interval']
        
        # Only save every n steps and if we have history accumulated
        if save_idx % save_interval == 0 and len(history_buffer) > 1:
            try:
                
                # logger.info(f"Generating history RGB at step {save_idx} with {len(history_buffer)} frames")
                
                history_rgb = accumulate_history_rgbd_to_panoramic(
                    history_buffer=history_buffer,
                    current_pose=history_data['current_pose'],
                    camera_matrix=history_data['camera_matrix'],
                    sensor_height=history_data['sensor_height'],
                    camera_elevation_degree=history_data['camera_elevation'],
                    device=history_data['device'],
                    output_size=history_data['output_size'],
                    window_size=history_data['window_size'],
                    sparse_factor_old=history_data.get('sparse_factor_old', 4)
                )
                
                # Save history RGB
                history_path = f"{paths['history_rgbs']}/{save_idx:03d}.png"
                cv2.imwrite(history_path, cv2.cvtColor(history_rgb, cv2.COLOR_RGB2BGR))
                # logger.info(f"Saved history RGB to {history_path}")
            except Exception as e:
                from habitat import logger
                import traceback
                logger.error(f"Failed to generate history RGB at step {save_idx}: {e}")
                logger.error(traceback.format_exc())
    
    return best_label


def save_top_down_map(save_idx, top_down_path, infos):
    """
    Save top-down map visualization from infos with full processing:
    colorization, fog of war, agent drawing, rotation, and resizing.
    
    Args:
        save_idx: int, step index for filename
        top_down_path: str, directory path to save the image
        infos: dict, contains 'top_down_map_vlnce' or 'top_down_map' key
    
    Returns:
        PIL.Image or None if no top-down map found
    """
    # Determine which map key exists
    map_k = None
    if "top_down_map_vlnce" in infos:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in infos:
        map_k = "top_down_map"
    
    if map_k is None:
        return None
    
    # Extract raw map
    td_map = infos[map_k]["map"]
    
    # Colorize with fog of war
    td_map = vln_maps.colorize_topdown_map(
        td_map,
        infos[map_k]["fog_of_war_mask"],
        fog_of_war_desat_amount=0.75,
    )
    
    # Draw agent on map
    # Ensure agent_angle is a scalar (extract from list/array if needed)
    agent_angle = infos[map_k]["agent_angle"]
    if isinstance(agent_angle, (list, tuple, np.ndarray)):
        agent_angle = float(agent_angle[0]) if len(agent_angle) > 0 else 0.0
    else:
        agent_angle = float(agent_angle)
    
    # Ensure agent_center_coord is a flat tuple of two ints
    agent_coord = infos[map_k]["agent_map_coord"]
    if isinstance(agent_coord, np.ndarray):
        agent_coord = agent_coord.flatten().tolist()
    # Recursively flatten nested structures
    while isinstance(agent_coord, (list, tuple)) and len(agent_coord) > 0 and isinstance(agent_coord[0], (list, tuple)):
        agent_coord = agent_coord[0]
    if isinstance(agent_coord, (list, tuple)) and len(agent_coord) >= 2:
        agent_coord = (int(agent_coord[0]), int(agent_coord[1]))
    else:
        # Fallback to center of map if coord is invalid
        agent_coord = (td_map.shape[0] // 2, td_map.shape[1] // 2)
    
    td_map = habitat_vis_maps.draw_agent(
        image=td_map,
        agent_center_coord=agent_coord,
        agent_rotation=agent_angle,
        agent_radius_px=max(1, min(td_map.shape[0:2]) // 24),
    )
    
    # Rotate if needed to make landscape orientation
    if td_map.shape[1] < td_map.shape[0]:
        td_map = np.rot90(td_map, 1)
    
    if td_map.shape[0] > td_map.shape[1]:
        td_map = np.rot90(td_map, 1)
    
    # Convert to PIL Image
    if td_map.dtype != np.uint8:
        td_map = np.clip(td_map, 0, 255).astype(np.uint8)
    
    top_down_pil = Image.fromarray(td_map)
    
    # Ensure RGB mode
    if top_down_pil.mode != 'RGB':
        top_down_pil = top_down_pil.convert('RGB')
    
    # Save to file
    output_file = os.path.join(top_down_path, f"{save_idx:03d}.png")
    top_down_pil.save(output_file)
    
    return top_down_pil


def write_rgbds_images(save_idx,
                       rgb_path, depth_path, semantic_path, semantic_room_path,
                       rgb_vis, depth_vis,
                       semantic_obj_idx,
                       semantic_room_idx=None, semantic_room_name=None):
    """
    Write RGB, depth, and semantic images to the specified paths.
    """

    cv2.imwrite(f"{rgb_path}/{save_idx:03d}.png", rgb_vis)
    cv2.imwrite(f"{depth_path}/{save_idx:03d}.png", depth_vis)

    rgb_pil = Image.fromarray(cv2.cvtColor(rgb_vis.astype(np.uint8), cv2.COLOR_BGR2RGB))
    depth_pil = Image.fromarray(depth_vis)

    # save_rgb(rgb_vis, os.path.join(rgb_path, f"{save_idx:03d}"), also_npy=True)
    # save_depth(depth_vis, os.path.join(depth_path, f"{save_idx:03d}"))

    semantic_pil = save_semantic_annotated_from_masks(f"{semantic_path}/{save_idx:03d}.png",
                                                      semantic_obj_idx)

    # save_semantic(semantic_vis, os.path.join(semantic_path, f"{save_idx:03d}"))

    # semantic_room_pil = save_semantic_annotated_from_masks_room(f"{semantic_room_path}/{save_idx:03d}.png",
    #                                                             semantic_room_idx,
    #                                                             semantic_room_name)

    return rgb_pil, depth_pil, semantic_pil


def write_semantic_images(save_idx,):
    pass


def write_map_images(save_idx,
                     local_map, planner_pose_inputs,
                     occ_path, occ_exp_path, occ_frontier_path, occ_frontier_gt_path,
                     map_resolution, centroids, frontier_u8, vis_config=None):
    """
    Generate independent BEV visualizations for each stage.
    Each visualization is created from scratch to avoid accumulation.
    
    Args:
        vis_config: Optional dict with visualization settings from config.mapping.visualization
    """
    # Extract visualization config parameters
    if vis_config is None:
        vis_config = {}
    
    crop_radius = vis_config.get('crop_radius', 150)
    output_size = vis_config.get('output_size', 480)
    arrow_color = tuple(vis_config.get('arrow_color', [255, 0, 0]))
    arrow_len_px = vis_config.get('arrow_len_px', 22)
    arrow_width = vis_config.get('arrow_width', 4)
    head_length = vis_config.get('arrow_head_length', 10)
    head_width = vis_config.get('arrow_head_width', 8)
    mark_radius = vis_config.get('mark_radius', 4)
    trail_color = tuple(vis_config.get('trail_color', [0, 0, 255]))
    trail_alpha = vis_config.get('trail_alpha', 200)
    trail_erode_ksize = vis_config.get('trail_erode_ksize', 3)
    frontier_dot_radius = vis_config.get('frontier_dot_radius', 5)
    frontier_color = tuple(vis_config.get('frontier_color', [0, 255, 0]))
    frontier_outline = tuple(vis_config.get('frontier_outline', [255, 255, 255]))
    frontier_width = vis_config.get('frontier_width', 2)
    
    # 1. Occupancy only: free + occupied + trajectory + agent pose
    occ_pil = plot_local_occupancy_with_pose_pil(
        local_map,
        planner_pose_inputs_row=planner_pose_inputs,
        map_resolution=map_resolution,
        crop_radius=crop_radius,
        output_size=output_size,
        arrow_color=arrow_color,
        arrow_len_px=arrow_len_px,
        arrow_width=arrow_width,
        head_length=head_length,
        head_width=head_width,
        mark_radius=mark_radius,
        trail_color=trail_color,
        trail_alpha=trail_alpha,
        trail_erode_ksize=trail_erode_ksize,
    )
    occ_pil.save(os.path.join(occ_path, f"{save_idx:03d}.png"))

    # 2. Occupancy + Exploration: free + occupied + explored + trajectory + agent pose
    occ_exp_pil = plot_local_occupency_explore_with_pose_pil(
        local_map,
        planner_pose_inputs_row=planner_pose_inputs,
        map_resolution=map_resolution,
        crop_radius=crop_radius,
        output_size=output_size,
        arrow_color=arrow_color,
        arrow_len_px=arrow_len_px,
        arrow_width=arrow_width,
        head_length=head_length,
        head_width=head_width,
        mark_radius=mark_radius,
        trail_color=trail_color,
        trail_alpha=trail_alpha,
        trail_erode_ksize=trail_erode_ksize,
    )
    occ_exp_pil.save(os.path.join(occ_exp_path, f"{save_idx:03d}.png"))

    # 3. Occupancy + Exploration + Frontier points
    # Create fresh base from occ_exp to avoid modifying original
    occ_exp_for_frontiers = plot_local_occupency_explore_with_pose_pil(
        local_map,
        planner_pose_inputs_row=planner_pose_inputs,
        map_resolution=map_resolution,
        crop_radius=crop_radius,
        output_size=output_size,
        arrow_color=arrow_color,
        arrow_len_px=arrow_len_px,
        arrow_width=arrow_width,
        head_length=head_length,
        head_width=head_width,
        mark_radius=mark_radius,
        trail_color=trail_color,
        trail_alpha=trail_alpha,
        trail_erode_ksize=trail_erode_ksize,
    )
    occ_exp_with_pts = overlay_frontier_centroids_on_rotated_map(
        occ_exp_for_frontiers,
        local_map,
        planner_pose_inputs,
        map_resolution,
        frontier_u8,
        centroids,
        flip_y_up=True,
        dot_radius=frontier_dot_radius,
        color=frontier_color,
        outline=frontier_outline,
        width=frontier_width,
        enumerate_points=True,
        rotate_to_heading=False,  # No rotation
        crop_radius=crop_radius
    )
    occ_exp_with_pts.save(os.path.join(occ_frontier_path, f"{save_idx:03d}.png"))

    # 4. Occupancy + Exploration + All frontiers + Selected frontier highlighted
    # Create fresh base from occ_exp with frontier points
    occ_exp_for_selection = plot_local_occupency_explore_with_pose_pil(
        local_map,
        planner_pose_inputs_row=planner_pose_inputs,
        map_resolution=map_resolution,
        crop_radius=crop_radius,
        output_size=output_size,
        arrow_color=arrow_color,
        arrow_len_px=arrow_len_px,
        arrow_width=arrow_width,
        head_length=head_length,
        head_width=head_width,
        mark_radius=mark_radius,
        trail_color=trail_color,
        trail_alpha=trail_alpha,
        trail_erode_ksize=trail_erode_ksize,
    )
    occ_exp_with_pts_fresh = overlay_frontier_centroids_on_rotated_map(
        occ_exp_for_selection,
        local_map,
        planner_pose_inputs,
        map_resolution,
        frontier_u8,
        centroids,
        flip_y_up=True,
        dot_radius=frontier_dot_radius,
        color=frontier_color,
        outline=frontier_outline,
        width=frontier_width,
        enumerate_points=True,
        rotate_to_heading=False,  # No rotation
        crop_radius=crop_radius
    )
    
    # Transform centroids to match rotated/cropped map for selection
    import cv2
    import torch
    lm = local_map.detach().cpu().numpy() if isinstance(local_map, torch.Tensor) else np.asarray(local_map)
    H_orig, W_orig = lm.shape[2], lm.shape[3]
    
    # Extract agent position and yaw
    start_x_m, start_y_m, start_o_deg, gx1, gx2, gy1, gy2 = planner_pose_inputs[:7]
    gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
    row_idx = start_y_m / map_resolution
    col_idx = start_x_m / map_resolution
    agent_row = int(row_idx - gx1)
    agent_col = int(col_idx - gy1)
    agent_yaw_deg = float(start_o_deg)
    
    # Calculate rotation and crop parameters (use config values)
    rotation_angle = 0.0  # No rotation since rotate_to_heading=False
    crop_r1 = max(0, agent_row - crop_radius)
    crop_r2 = min(H_orig, agent_row + crop_radius)
    crop_c1 = max(0, agent_col - crop_radius)
    crop_c2 = min(W_orig, agent_col + crop_radius)
    H_displayed = crop_r2 - crop_r1
    W_displayed = crop_c2 - crop_c1
    
    # Transform centroids (only crop, no rotation)
    transformed_centroids = []
    for r, c in centroids:
        # Apply crop offset
        r_final = r - crop_r1
        c_final = c - crop_c1
        # Only include if within bounds
        if 0 <= r_final < H_displayed and 0 <= c_final < W_displayed:
            transformed_centroids.append((r_final, c_final))
    
    # Select and highlight the chosen frontier
    selected_color = tuple(vis_config.get('selected_frontier_color', [255, 215, 0]))
    selected_line_width = vis_config.get('selected_line_width', 3)
    selected_ring_radius = vis_config.get('selected_ring_radius', 7)
    selected_show_index = vis_config.get('selected_show_index', False)
    selected_index_color = tuple(vis_config.get('selected_index_color', [255, 255, 255]))
    
    # Frontier selection parameters
    selection_mode = vis_config.get('frontier_selection_mode', 'hybrid')
    w_dist = vis_config.get('frontier_weight_distance', 0.3)
    w_theta = vis_config.get('frontier_weight_heading', 0.7)
    angle_cone = vis_config.get('frontier_angle_cone_deg', 180)
    
    if len(transformed_centroids) > 0:
        best_idx, best_label, selected_frontier_pil, info = select_and_visualize_frontier(
            base_panel=occ_exp_with_pts_fresh,
            centroids=transformed_centroids,  # Use transformed coordinates
            planner_pose_inputs_row=planner_pose_inputs,
            bev_hw_cells=(H_displayed, W_displayed),  # Use displayed size
            map_resolution=map_resolution,   # meters/cell
            flip_y_up=True,
            mode=selection_mode,
            label_start=1,
            w_theta=w_theta,
            w_dist=w_dist,
            angle_cone_deg=angle_cone,
            highlight_color=selected_color,
            line_color=selected_color,
            line_width=selected_line_width,
            ring_radius=selected_ring_radius,
            show_index=selected_show_index,
            index_fill=selected_index_color,
        )
        assert best_label == best_idx + 1
    else:
        # No valid frontiers in visible area, use first frontier as fallback
        best_idx = 0
        best_label = 1
        selected_frontier_pil = occ_exp_with_pts_fresh
    selected_frontier_pil.save(os.path.join(occ_frontier_gt_path, f"{save_idx:03d}.png"))

    # 5. Generate FOV visualization
    from .mapping.visualization import draw_fov_on_bev
    # Extract sensor parameters from vis_config if available, otherwise use defaults
    fov_hfov = vis_config.get('fov_hfov')  # Default HFOV from sensor config (degrees)
    fov_max_depth_m = vis_config.get('fov_max_depth_m')  # Default max_depth from sensor config (meters)
    # NOTE: map_resolution is in meters/cell (e.g., 0.05), NOT centimeters
    # With map_resolution=0.05 m/cell and max_depth=4.85m: fov_range_cells = 97 cells
    fov_range_cells = int(fov_max_depth_m / map_resolution)  # Convert meters to cells
    
    fov_pil = draw_fov_on_bev(
        local_map,
        planner_pose_inputs_row=planner_pose_inputs,
        map_resolution=map_resolution,
        crop_radius=crop_radius,
        output_size=output_size,
        fov_angle_deg=fov_hfov,
        fov_range_cells=fov_range_cells,
        arrow_color=arrow_color,
        arrow_len_px=arrow_len_px,
        arrow_width=arrow_width,
        head_length=head_length,
        head_width=head_width,
        mark_radius=mark_radius,
        trail_color=trail_color,
        trail_alpha=trail_alpha,
        trail_erode_ksize=trail_erode_ksize,
    )

    return occ_pil, occ_exp_pil, occ_exp_with_pts, selected_frontier_pil, best_label, fov_pil


def build_name_map(seg_idx: np.ndarray, seg_name: np.ndarray) -> Dict[int, str]:
    """
    Majority-vote name per id from an egocentric map.
    seg_idx: (H,W) ints
    seg_name: (H,W) strings (your case)
    Returns: {class_id:int -> name:str}, always includes {0: "unknown"}.
    """
    idx = np.asarray(seg_idx)
    nam = np.asarray(seg_name)

    # Be resilient: force unicode strings and drop Nones/empties
    nam = nam.astype("U", copy=False)
    nam[nam == None] = ""  # just in case
    id_to_name = {0: "unknown"}

    for cid in np.unique(idx):
        mask = (idx == cid)
        if not np.any(mask):
            continue
        vals, cts = np.unique(nam[mask], return_counts=True)
        # drop empty strings
        nonempty = np.array([len(v.strip()) > 0 for v in vals])
        vals, cts = vals[nonempty], cts[nonempty]
        if vals.size == 0:
            continue
        best = vals[int(np.argmax(cts))]
        id_to_name[int(cid)] = str(best)
    return id_to_name

def name_for(id_to_name: Union[dict, List[str]], cid: int) -> str:
    cid = int(cid)
    if isinstance(id_to_name, dict):
        return id_to_name.get(cid, "unknown")
    if isinstance(id_to_name, (list, tuple)):
        return id_to_name[cid] if 0 <= cid < len(id_to_name) else "unknown"
    return "unknown"


def _bev_ids_from_local_map(local_map, *, env_idx=0, ch_start=4, ch_count=None, conf_thr=0.0):
    """Argmax over selected BEV semantic channels → (H,W) int ids (0=unknown)."""
    lm = local_map.detach().cpu().numpy() if isinstance(local_map, torch.Tensor) else np.asarray(local_map)
    _, C, H, W = lm.shape
    if ch_count is None:
        sem = lm[env_idx, ch_start:, :, :]
    else:
        sem = lm[env_idx, ch_start:ch_start+int(ch_count), :, :]
    if sem.size == 0:
        return np.zeros((H, W), dtype=np.int32)
    arg = sem.argmax(axis=0)          # 0..K-1
    mx  = sem.max(axis=0)
    ids = (arg + 1).astype(np.int32)  # shift: 0 reserved for unknown
    ids[mx < conf_thr] = 0
    return ids


from typing import Optional, Tuple, Sequence, Dict
import math
import numpy as np
from PIL import Image, ImageDraw

def write_semantic_bev_object(
    local_map,
    planner_pose_inputs_row: Optional[Sequence[float]],
    *,
    env_idx: int = 0,
    # Occupancy/exploration thresholds
    occ_thr: float = 0.5,
    exp_thr: float = 0.1,
    # Grays
    gray_unknown: int = 127,
    gray_free: int = 255,
    # Semantic inputs
    sem_ids_bev: Optional[np.ndarray] = None,  # pass HxW BEV ids if you already have them
    sem_ch_start: int = 4,                     # used if sem_ids_bev is None
    sem_ch_count: Optional[int] = None,
    conf_thr: float = 0.0,
    skip_ids: Tuple[int, ...] = (0,),          # don't paint these ids
    # Trail
    trail_color: Tuple[int, int, int] = (30, 144, 255),
    trail_alpha: int = 200,
    trail_erode_ksize: int = 3,
    # Arrow
    arrow_color: Tuple[int, int, int] = (255, 0, 0),
    arrow_len_px: int = 22,
    arrow_width: int = 4,
    head_length: int = 10,
    head_width: int = 8,
    mark_radius: int = 4,
    # Coords / size
    flip_y_up: bool = True,
    target_size: Optional[Tuple[int, int]] = None,  # (W,H) resize before drawing arrow
    map_resolution: Optional[float] = None,         # cm/cell when using gx1..gy2 mapping
    # Palette
    palette_rgb: Optional[np.ndarray] = None,       # defaults to d3_40_colors_rgb if None
    # Output
    out_path: Optional[str] = None,
) -> Image.Image:
    """High-level wrapper: builds a BEV panel per the color scheme and saves if out_path is given."""
    img = _write_semantic_bev_single(
        local_map,
        planner_pose_inputs_row=planner_pose_inputs_row,
        env_idx=env_idx,
        occ_thr=occ_thr, exp_thr=exp_thr,
        gray_unknown=gray_unknown, gray_free=gray_free,
        sem_ids=sem_ids_bev,
        sem_ch_start=sem_ch_start, sem_ch_count=sem_ch_count, conf_thr=conf_thr,
        skip_ids=skip_ids,
        trail_color=trail_color, trail_alpha=trail_alpha, trail_erode_ksize=trail_erode_ksize,
        arrow_color=arrow_color, arrow_len_px=arrow_len_px, arrow_width=arrow_width,
        head_length=head_length, head_width=head_width, mark_radius=mark_radius,
        flip_y_up=flip_y_up, target_size=target_size, map_resolution=map_resolution,
        palette_rgb=palette_rgb
    )
    if out_path is not None:
        img.save(out_path)
    return img


def _write_semantic_bev_single(
    local_map,
    planner_pose_inputs_row: Optional[Sequence[float]] = None,
    *,
    env_idx: int = 0,
    # thresholds
    occ_thr: float = 0.5,
    exp_thr: float = 0.1,
    # grays
    gray_unknown: int = 127,
    gray_free: int = 255,
    # semantic
    sem_ids: Optional[np.ndarray] = None,   # HxW ids; if None -> derive from local_map semantic channels
    sem_ch_start: int = 4,
    sem_ch_count: Optional[int] = None,
    conf_thr: float = 0.0,
    skip_ids: Tuple[int, ...] = (0,),
    # trail
    trail_color: Tuple[int, int, int] = (30, 144, 255),
    trail_alpha: int = 200,
    trail_erode_ksize: int = 3,
    # arrow
    arrow_color: Tuple[int, int, int] = (255, 0, 0),
    arrow_len_px: int = 22,
    arrow_width: int = 4,
    head_length: int = 10,
    head_width: int = 8,
    mark_radius: int = 4,
    # coords / size
    flip_y_up: bool = True,
    target_size: Optional[Tuple[int, int]] = None,  # (W,H)
    map_resolution: Optional[float] = None,         # cm/cell (used with gx1..gy2)
    # palette
    palette_rgb: Optional[np.ndarray] = None        # defaults to d3_40_colors_rgb if None
) -> Image.Image:
    """
    BEV rendering with rule:
      Unexplored → gray_unknown
      Explored & free → white
      Explored & occupied → semantic color
    """

    # ---- helpers ----
    def _as_np(a):
        import torch
        return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)

    def _get_bev_channel(lmap, env_i, ch_i):
        m = _as_np(lmap)
        if m.ndim == 4:   # (N,C,H,W)
            return m[env_i, ch_i, :, :]
        elif m.ndim == 3: # (C,H,W)
            return m[ch_i, :, :]
        raise ValueError(f"Unexpected local_map shape: {m.shape}")

    def _resize_bool(mask, wh):
        Wt, Ht = wh
        return np.array(Image.fromarray(mask.astype(np.uint8)*255, "L").resize((Wt, Ht), Image.NEAREST)) > 0

    def _resize_int(arr, wh):
        Wt, Ht = wh
        return np.array(Image.fromarray(arr.astype(np.int32), "I").resize((Wt, Ht), Image.NEAREST))

    def _ensure_palette():
        return d3_40_colors_rgb if palette_rgb is None else palette_rgb

    def _colorize_ids(ids_hw: np.ndarray, wh: Optional[Tuple[int,int]] = None) -> Image.Image:
        pal = palette_flat_256(_ensure_palette())
        pal_img = colorize_by_ids(ids_hw, pal)  # 'P'
        if wh is not None:
            pal_img = pal_img.resize(wh, Image.NEAREST)
        return pal_img.convert("RGBA")

    # ---- extract maps ----
    lm = _as_np(local_map)
    assert lm.ndim == 4, f"local_map must be (N,C,H,W); got {lm.shape}"
    N, C, H, W = lm.shape
    assert 0 <= env_idx < N, f"env_idx out of range (N={N})"

    occ = np.clip(lm[env_idx, 0], 0.0, 1.0)                   # (H,W)
    exp = lm[env_idx, 1] if C >= 2 else np.ones_like(occ)     # (H,W)
    agent_mask = (lm[env_idx, 2] > 0.5) if C >= 3 else None   # (H,W)
    trail_mask = (lm[env_idx, 3] > 0.5) if C >= 4 else None   # (H,W)

    # semantic ids
    if sem_ids is None:
        sem_ids = _bev_ids_from_local_map(local_map, env_idx=env_idx,
                                          ch_start=sem_ch_start, ch_count=sem_ch_count,
                                          conf_thr=conf_thr)
    else:
        sem_ids = np.asarray(sem_ids)
        assert sem_ids.shape == (H, W), f"sem_ids must be (H,W); got {sem_ids.shape}"

    # classify
    explored = (exp >= exp_thr)
    occupied = (occ >= occ_thr) & explored
    free = explored & (~occupied)
    unknown = ~explored

    # flip display coords
    if flip_y_up:
        occupied = np.flipud(occupied).copy()
        free     = np.flipud(free).copy()
        unknown  = np.flipud(unknown).copy()
        if agent_mask is not None: agent_mask = np.flipud(agent_mask).copy()
        if trail_mask is not None: trail_mask = np.flipud(trail_mask).copy()
        sem_ids  = np.flipud(sem_ids).copy()

    # optional resize before vector drawing
    Wd, Hd = W, H
    if target_size is not None:
        Wd, Hd = target_size
        occupied = _resize_bool(occupied, (Wd, Hd))
        free     = _resize_bool(free,     (Wd, Hd))
        unknown  = _resize_bool(unknown,  (Wd, Hd))
        sem_ids  = _resize_int(sem_ids,   (Wd, Hd))
        if agent_mask is not None:
            agent_mask = _resize_bool(agent_mask, (Wd, Hd))
        if trail_mask is not None:
            trail_mask = _resize_bool(trail_mask, (Wd, Hd))

    # base = unknown gray
    base_L = np.full((Hd, Wd), np.uint8(np.clip(gray_unknown, 0, 255)), dtype=np.uint8)
    panel = Image.fromarray(base_L, "L").convert("RGBA")
    panel.putalpha(255)

    # overlay free = white
    if free.any():
        white = Image.new("RGBA", (Wd, Hd), (255, 255, 255, 255))
        free_L = Image.fromarray((free.astype(np.uint8) * 255), "L")
        panel = Image.composite(white, panel, free_L)

    # overlay occupied = semantic color
    if occupied.any():
        # mask out skip ids
        occ_mask = occupied
        if skip_ids:
            occ_mask = occ_mask & (~np.isin(sem_ids, np.array(skip_ids, dtype=sem_ids.dtype)))
        sem_rgba = _colorize_ids(sem_ids, (Wd, Hd))
        a = np.zeros((Hd, Wd), dtype=np.uint8); a[occ_mask] = 255
        sem_rgba.putalpha(Image.fromarray(a, "L"))
        panel = Image.alpha_composite(panel, sem_rgba)

    # trail overlay
    if trail_mask is not None and trail_mask.any():
        tm = trail_mask.copy()
        if trail_erode_ksize and trail_erode_ksize > 1:
            try:
                import cv2
                k = np.ones((trail_erode_ksize, trail_erode_ksize), np.uint8)
                tm_u8 = (tm.astype(np.uint8) * 255)
                tm_u8 = cv2.erode(tm_u8, k, iterations=1)
                tm = tm_u8 > 0
            except Exception:
                try:
                    from scipy.ndimage import minimum_filter
                    tm = minimum_filter(tm.astype(np.uint8), size=trail_erode_ksize) > 0
                except Exception:
                    pass
        overlay = Image.new("RGBA", (Wd, Hd), trail_color + (0,))
        a = np.zeros((Hd, Wd), dtype=np.uint8); a[tm] = np.uint8(np.clip(trail_alpha, 0, 255))
        overlay.putalpha(Image.fromarray(a, "L"))
        panel = Image.alpha_composite(panel, overlay)

    # arrow
    draw = ImageDraw.Draw(panel)
    x_pix = y_pix = None
    if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 7 and map_resolution is not None:
        start_x_m, start_y_m, start_o_deg, gx1, gx2, gy1, gy2 = planner_pose_inputs_row[:7]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        H_win, W_win = (gx2 - gx1), (gy2 - gy1)    # rows, cols in cells
        row_idx = start_y_m / map_resolution  # meters / (meters/cell) = cells
        col_idx = start_x_m / map_resolution
        row_rel = row_idx - gx1
        col_rel = col_idx - gy1
        x_pix = col_rel * (Wd / W_win)
        y_pix = (H_win - row_rel) * (Hd / H_win) if flip_y_up else row_rel * (Hd / H_win)
        yaw_deg = float(-start_o_deg) if flip_y_up else float(start_o_deg)

    if x_pix is None or y_pix is None:
        if agent_mask is not None and agent_mask.any():
            ys, xs = np.nonzero(agent_mask)
            x_pix = xs.mean()
            y_pix = ys.mean()
        else:
            x_pix, y_pix = Wd / 2.0, Hd / 2.0
        yaw_deg = float(planner_pose_inputs_row[2]) if (planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 3) else 0.0
        yaw_deg = -yaw_deg if flip_y_up else yaw_deg

    # draw big arrow
    theta = math.radians(yaw_deg)
    dx, dy = math.cos(theta) * arrow_len_px, math.sin(theta) * arrow_len_px
    x1, y1 = float(x_pix), float(y_pix)
    x2, y2 = x1 + dx, y1 + dy
    vlen = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / vlen, dy / vlen
    bx, by = x2 - ux * head_length, y2 - uy * head_length
    draw.line([(x1, y1), (bx, by)], fill=arrow_color, width=int(arrow_width))
    px, py = -uy, ux
    hx1, hy1 = bx + px * (head_width / 2.0), by + py * (head_width / 2.0)
    hx2, hy2 = bx - px * (head_width / 2.0), by - py * (head_width / 2.0)
    draw.polygon([(x2, y2), (hx1, hy1), (hx2, hy2)], fill=arrow_color)

    # center marker (optional)
    if agent_mask is not None and agent_mask.any():
        r = int(mark_radius)
        draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], outline=arrow_color, width=2)

    return panel.convert("RGB")


# def write_mask_rcnn_images(rgb_vis, pred_box, save_idx, maskrcnn_path):
#
#     # 确保 rgb_vis 是 numpy (H,W,3), uint8, BGR
#     if isinstance(rgb_vis, Image.Image):
#         rgb_vis = np.array(rgb_vis)[:, :, ::-1]
#
#     if rgb_vis.dtype != np.uint8:
#         rgb_vis = rgb_vis.astype(np.uint8)
#
#     rgb_vis = np.ascontiguousarray(rgb_vis)
#
#     # 画框 + 标签
#     for cls, score, box in pred_box:
#         x1, y1, x2, y2 = box
#         cv2.rectangle(rgb_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
#
#         class_name = id2name.get(cls, str(cls))  # 找名字，找不到就打印数字
#         label = f"{class_name} ({score:.2f})"
#
#         cv2.putText(rgb_vis, label, (int(x1), int(y1)-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#
#     cv2.imwrite(f"{maskrcnn_path}/{save_idx:03d}.png", rgb_vis)


def process_depth_heuristics(depth, min_depth, max_depth):
    '''
    input:
        min_depth: minimum depth in meters
        max_depth: maximum depth in meters

    output:
        processed depth, numpy array, 640 X 480, in meters
    '''
    assert depth.ndim == 3 and depth.shape[2] == 1

    depth = depth[:, :, 0] * 1

    for i in range(depth.shape[0]):
        depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

    mask2 = depth > 0.99
    depth[mask2] = 0.

    mask1 = depth == 0
    depth[mask1] = 100.0
    depth = min_depth + depth * max_depth  # Keep in meters
    return depth


def save_rgb(rgb, filename_no_ext, also_npy=False):
    """
    Save RGB image:
      - <filename>.png : BGR uint8 (for viewing)
      - <filename>.npy : optional, HxWx3 original numeric (for exact reuse)
    """
    _ensure_dir(os.path.dirname(filename_no_ext))
    img = _to_hwc_uint8(rgb)
    cv2.imwrite(filename_no_ext + ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if also_npy:
        np.save(filename_no_ext + ".npy", _to_numpy(rgb))

def save_depth(depth_cm, filename_no_ext, save_vis_png=True):
    """
    depth: numpy array, 单位 cm (float)
    filename: 不带扩展名的文件路径，比如 'outputs/depth/depth_step0001'
    """
    # -------- 保存 numpy 原始数据 (完全保真) --------
    _ensure_dir(os.path.dirname(filename_no_ext))
    depth = _to_numpy(depth_cm).astype(np.float32)
    np.save(filename_no_ext + ".npy", depth)

    if save_vis_png:
        # simple linear mapping for visualization
        vmax = float(np.percentile(depth[np.isfinite(depth)], 99)) if np.isfinite(depth).any() else float(depth.max() + 1e-6)
        vmax = vmax if vmax > 0 else 1.0
        # 将深度映射到 0-255 范围，便于肉眼观察
        vis = np.clip(depth / vmax * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(filename_no_ext + ".png", vis)

def palette_flat_256(pal40: np.ndarray) -> list:
    base = pal40.reshape(-1).tolist()
    return base + [0] * (256*3 - len(base))

def colorize_by_ids(mask_ids: np.ndarray, palette_flat: list, base: int = 40) -> Image.Image:
    h, w = mask_ids.shape
    pal_img = Image.new("P", (w, h))
    pal_img.putpalette(palette_flat)
    pal_img.putdata((mask_ids.flatten() % base).astype(np.uint8))
    return pal_img.convert("RGBA")


# --- legend building (compact, top-right overlay) ---
def make_legend(categories, id_to_name, palette_flat,
                swatch=(18,18), pad=6, label_px_max=140, bg=(255,255,255,210)) -> Image.Image:
    if not categories:
        return Image.new("RGBA", (1,1), (0,0,0,0))
    try:
        font = ImageFont.load_default()  # compact default font
    except Exception:
        font = None

    # Measure text widths to compute a compact legend width
    dummy = Image.new("RGBA", (1,1))
    meas = ImageDraw.Draw(dummy)
    def text_w(s):  # robust width measurement
        if hasattr(meas, "textbbox"):
            return meas.textbbox((0,0), s, font=font)[2]
        return int(meas.textlength(s, font=font))  # fallback

    # Optional: ellipsize text to fit within max px
    def ellipsize_to_px(s, max_px):
        if text_w(s) <= max_px:
            return s
        base = s
        # keep trimming until "…"-appended fits
        while base and text_w(base + "…") > max_px:
            base = base[:-1]
        return base + "…" if base else "…"

    labels = [f"{int(cid)}: {name_for(id_to_name, cid)}" for cid in categories]

    # First compute the natural max width, then clamp to label_px_max
    natural_max = max((text_w(lbl) for lbl in labels), default=0)
    label_w = min(natural_max, label_px_max)
    # Create ellipsized labels that fit the clamp
    labels = [ellipsize_to_px(lbl, label_w) for lbl in labels]

    sw_w, sw_h = swatch
    row_h = max(sw_h, 16) + pad
    width  = pad + sw_w + pad + label_w + pad
    height = pad + len(categories) * row_h
    legend = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(legend)

    y = pad
    for cid, lbl in zip(categories, labels):
        color = color_from_palette(palette_flat, cid)
        draw.rectangle([pad, y, pad+sw_w, y+sw_h], fill=color, outline=(0,0,0,255))
        draw.text((pad+sw_w+pad, y), lbl, fill=(0,0,0,255), font=font)
        y += row_h
    return legend

def color_from_palette(palette_flat: list, idx: int, base: int = 40) -> tuple:
    j = (int(idx) % base) * 3
    return tuple(palette_flat[j:j+3])  # (R,G,B)

def overlay_top_right(base_rgba: Image.Image, overlay_rgba: Image.Image, margin_px: int = 8) -> Image.Image:
    out = base_rgba.copy()
    x = out.width - overlay_rgba.width - margin_px
    y = margin_px
    # paste with alpha (overlay_rgba has an alpha channel)
    out.paste(overlay_rgba, (x, y), overlay_rgba)
    return out

def save_semantic_annotated_from_masks_room(
    save_path: str,
    seg_idx: np.ndarray,          # (H,W) int ids; 0=
    seg_name: str,                # room name
    render_void: bool = False,  # False => make void transparent; True => paint with void_color
    void_color=(200, 200, 200, 180),  # RGBA if render_void=True
    include_void_in_legend: bool = False,  # whether to show "void" in legend
    skip_ids: tuple = (),
    # additional ids to hide in legend (besides auto-handling of 0 if include_void_in_legend=False)
    legend_max_label_px: int = 140,  # tighter legend width cap
    legend_swatch=(18, 18),  # smaller swatches
    legend_pad: int = 6,  # tighter padding
    legend_bg=(255, 255, 255, 210),  # semi-opaque legend background
    margin: int = 8,  # margin from top-right corner

):
    seg_idx_int = seg_idx.astype(int).flatten()
    seg_name_flat = seg_name.flatten()

    # Build dictionary
    id_to_name = dict(zip(seg_idx_int, seg_name_flat))

    ids_present = np.unique(seg_idx)

    palette_flat = palette_flat_256(d3_40_colors_rgb)
    semantic_img = colorize_by_ids(seg_idx, palette_flat, base=40)  # returns a PIL.Image ("P" or "RGBA" per your impl)

    # Ensure RGBA for editing alpha/colors
    if semantic_img.mode != "RGBA":
        semantic_rgba = semantic_img.convert("RGBA")
    else:
        semantic_rgba = semantic_img

    # --- handle void pixels (id==0) ---
    # If render_void is False => make void transparent; else paint with void_color
    void_mask = (seg_idx == 0)
    if void_mask.any():
        arr = np.array(semantic_rgba, dtype=np.uint8)  # (H,W,4)
        if render_void:
            vr, vg, vb, va = void_color
            arr[void_mask] = np.array([vr, vg, vb, va], dtype=np.uint8)
        else:
            # transparent
            arr[void_mask, 3] = 0  # set alpha to 0
        semantic_rgba = Image.fromarray(arr, mode="RGBA")

    # --- legend (omit void unless include_void_in_legend=True) ---
    cats_for_legend = [int(c) for c in ids_present
                       if (c != 0 or include_void_in_legend)
                       and int(c) not in skip_ids]

    legend_img = make_legend(
        cats_for_legend,
        id_to_name,
        palette_flat,
        swatch=legend_swatch,
        pad=legend_pad,
        label_px_max=legend_max_label_px,
        bg=legend_bg,
        # If your make_legend supports custom color for id 0, it will use void_color automatically
        # when include_void_in_legend is True. If not, it will pull the color from palette index 0.
        # (You can modify make_legend to special-case id==0 to use void_color if desired.)
    )

    vis = overlay_top_right(semantic_rgba, legend_img, margin_px=margin)
    vis.save(save_path)
    return vis



def save_semantic_annotated_from_masks(
    save_path: str,
    seg_idx: np.ndarray,          # (H,W) int ids; 0=void, 1..40 map to mp3d_category
    render_void: bool = False,           # False => make void transparent; True => paint with void_color
    void_color=(200, 200, 200, 180),     # RGBA if render_void=True
    include_void_in_legend: bool = False,# whether to show "void" in legend
    skip_ids: tuple = (),                # additional ids to hide in legend (besides auto-handling of 0 if include_void_in_legend=False)
    legend_max_label_px: int = 140,  # tighter legend width cap
    legend_swatch=(18, 18),          # smaller swatches
    legend_pad: int = 6,             # tighter padding
    legend_bg=(255, 255, 255, 210),  # semi-opaque legend background
    margin: int = 8,                 # margin from top-right corner
):
    """
    Colorize seg_idx with D3-40 palette (IDs 1..40) and add legend from mp3d_category.
    ID 0 is treated as 'void' (transparent by default, or colored if render_void=True).
    """

    # --- sanity checks ---
    assert isinstance(mp3d_category, (list, tuple)) and len(mp3d_category) == 40, \
        "mp3d_category must be a list of length 40 mapping ids 1..40."

    # --- build id -> name mapping from the fixed list ---
    ids_present = np.unique(seg_idx)
    id_to_name = {}
    for cid in ids_present:
        cid = int(cid)
        if cid == 0:
            if include_void_in_legend:
                id_to_name[0] = "void"
            continue
        if cid in skip_ids:
            continue
        if 1 <= cid <= 40:
            id_to_name[cid] = mp3d_category[cid - 1]
        else:
            # Fallback in case of out-of-range ids
            id_to_name[cid] = f"class_{cid}"

    # --- render base semantic image using D3-40 palette ---
    # palette_flat_256(d3_40_colors_rgb) should produce a 256*3 flat list where
    # indices 0..39 correspond to the D3-40 colors. That means:
    #  - id=1 uses the 2nd color, id=40 uses the 40th color.
    #  - id=0 (void) gets the first color; we'll override handling below.
    palette_flat = palette_flat_256(d3_40_colors_rgb)
    semantic_img = colorize_by_ids(seg_idx, palette_flat, base=40)  # returns a PIL.Image ("P" or "RGBA" per your impl)

    # Ensure RGBA for editing alpha/colors
    if semantic_img.mode != "RGBA":
        semantic_rgba = semantic_img.convert("RGBA")
    else:
        semantic_rgba = semantic_img

    # --- handle void pixels (id==0) ---
    # If render_void is False => make void transparent; else paint with void_color
    void_mask = (seg_idx == 0)
    if void_mask.any():
        arr = np.array(semantic_rgba, dtype=np.uint8)  # (H,W,4)
        if render_void:
            vr, vg, vb, va = void_color
            arr[void_mask] = np.array([vr, vg, vb, va], dtype=np.uint8)
        else:
            # transparent
            arr[void_mask, 3] = 0  # set alpha to 0
        semantic_rgba = Image.fromarray(arr, mode="RGBA")

    # --- legend (omit void unless include_void_in_legend=True) ---
    cats_for_legend = [int(c) for c in ids_present
                       if (c != 0 or include_void_in_legend)
                       and int(c) not in skip_ids]

    legend_img = make_legend(
        cats_for_legend,
        id_to_name,
        palette_flat,
        swatch=legend_swatch,
        pad=legend_pad,
        label_px_max=legend_max_label_px,
        bg=legend_bg,
        # If your make_legend supports custom color for id 0, it will use void_color automatically
        # when include_void_in_legend is True. If not, it will pull the color from palette index 0.
        # (You can modify make_legend to special-case id==0 to use void_color if desired.)
    )

    vis = overlay_top_right(semantic_rgba, legend_img, margin_px=margin)
    vis.save(save_path)
    return vis


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    Standard image transform for ImageNet pretrained models.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    chooses a patch layout (i.e., how many columns × rows) that:
        1. Best preserves the original image's aspect ratio
        2. Uses as many patches as possible (without being too small)
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Split a PIL image into a grid of square patches (tiles),
    each of size image_size × image_size,
    and optionally append a resized thumbnail of the original image

    potential image distortion
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list 