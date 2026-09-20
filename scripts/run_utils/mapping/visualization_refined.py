import numpy as np
import cv2
from numpy.random import f
import open3d as o3d
import tempfile

from PIL import Image, ImageDraw
import os

from scripts.run_utils.mapping.vis_utils_semantic_object import save_semantic_annotated_from_masks

from habitat.utils.visualizations import maps as habitat_vis_maps
from task_patch.mp3d_register import maps as vln_maps


COLORS = [
    [220, 20, 60],     # crimson (red)
    [50, 205, 50],     # lime green
    [148, 0, 211],     # dark violet
    [255, 215, 0],    # gold
    [139, 69, 19],    # saddle brown
    [255, 105, 180],  # hot pink
    [34, 139, 34],    # forest green
    [255, 69, 0],     # orange red
    [186, 85, 211],   # medium orchid
    [154, 205, 50],   # yellow green
    [199, 21, 133],   # medium violet red
    [255, 140, 0],    # dark orange
    [107, 142, 35],   # olive drab
    [221, 160, 221],  # plum
    [160, 82, 45],    # sienna
    [255, 228, 181],  # moccasin
    [210, 105, 30],   # chocolate
    [238, 232, 170],  # pale goldenrod
    [255, 182, 193],  # light pink
]

def write_gpu_point_cloud(pcd, ply_path):

    assert pcd.point.positions.shape[0] > 0
    save_pcd = o3d.geometry.PointCloud()
    save_pcd.points = o3d.utility.Vector3dVector(
        pcd.point.positions.cpu().numpy())
    save_pcd.colors = o3d.utility.Vector3dVector(
        pcd.point.colors.cpu().numpy())
    o3d.io.write_point_cloud(ply_path, save_pcd)


def write_map_vanilla(map, png_path):
    """
        Write map to PNG file.
        No arrow for agent heading.
        No scaling.
    """
    save_img = np.ones((map.shape[0], map.shape[1], 3), dtype=np.uint8)
    save_img *= 128
    save_img[map[:, :, 1] == 1] = [255, 255, 255]
    save_img[map[:, :, 0] == 1] = [0, 0, 0]
    save_img[map[:, :, 3] == 1] = [0, 0, 255]
    save_img[map[:, :, 2] == 1] = [255, 0, 0]
    cv2.imwrite(png_path, save_img)


def write_map_with_fov(full_map, png_path,
                       agent_pos_row, agent_pos_col, agent_yaw_deg,
                       map_resolution,
                       output_size=None,
                       hfov=79.0, sensor_max_range=5.0,
                       fov_color=(100, 200, 255), fov_alpha=0.3,
                       frontier_centers_2d=None, selected_frontier_index=None,
                       arrow_color=(0, 0, 255), arrow_length=20, arrow_thickness=2,
                       dot_radius=5, dot_color=(0, 255, 0),
                       outline_color=(255, 255, 255), outline_width=2):
    """
    Write full map with FOV sector/arc visualization (fog of war).

    Args:
        full_map: (H, W, C) numpy array with map channels
        png_path: Path to save the image
        agent_pos_row: Agent's row position in the map (y coordinate)
        agent_pos_col: Agent's col position in the map (x coordinate)
        agent_yaw_deg: Agent's heading in degrees
        map_resolution: Map resolution in meters per pixel
        output_size: Optional int, assert output image height matches this size
        hfov: Horizontal field of view in degrees (default 79.0)
        sensor_max_range: Maximum sensor range in meters (default 5.0)
        fov_color: BGR color tuple for FOV area
        fov_alpha: Alpha/transparency for FOV overlay (0-1)
        frontier_centers_2d: Optional numpy array (N, 2) - frontier centroids in (row, col) format
        selected_frontier_index: Optional int - index of selected frontier to highlight
        arrow_color: BGR color tuple for the arrow
        arrow_length: Length of the arrow in pixels
        arrow_thickness: Thickness of the arrow lines
        dot_radius: radius of frontier centroid dots
        dot_color: BGR color for frontier dots
        outline_color: BGR color for dot outline
        outline_width: width of outline

    Returns:
        save_img: numpy array of the visualization
    """
    # Create base image
    save_img = np.ones((full_map.shape[0], full_map.shape[1], 3), dtype=np.uint8)
    save_img *= 128
    save_img[full_map[:, :, 1] == 1] = [255, 255, 255]  # explored - white
    save_img[full_map[:, :, 0] == 1] = [0, 0, 0]        # obstacles - black
    save_img[full_map[:, :, 3] == 1] = [0, 0, 255]      # trajectory - blue
    save_img[full_map[:, :, 2] == 1] = [255, 0, 0]      # current position - red

    # Calculate depth range in pixels from sensor range and map resolution
    # sensor_max_range defines depth (perpendicular distance), not radial distance
    depth_range = int(sensor_max_range / map_resolution)

    # Create FOV triangle overlay (isoceles right triangle)
    overlay = save_img.copy()

    # Agent position (triangle apex)
    center = (int(agent_pos_col), int(agent_pos_row))

    # Calculate heading direction in image coordinates
    heading_angle = agent_yaw_deg - 90  # Transform to match image coordinates
    yaw_rad = np.deg2rad(heading_angle)

    # For isoceles right triangle with depth as the perpendicular distance:
    # - The apex angle is 90 degrees (hfov = 90)
    # - Leg length = depth * sqrt(2)
    # - The two legs are at ±45° from the heading direction
    half_fov_rad = np.deg2rad(45.0)  # Fixed 45° for isoceles right triangle
    leg_length = depth_range * np.sqrt(2)

    # Left vertex of triangle
    left_angle = yaw_rad - half_fov_rad
    left_x = int(agent_pos_col + leg_length * np.cos(left_angle))
    left_y = int(agent_pos_row + leg_length * np.sin(left_angle))

    # Right vertex of triangle
    right_angle = yaw_rad + half_fov_rad
    right_x = int(agent_pos_col + leg_length * np.cos(right_angle))
    right_y = int(agent_pos_row + leg_length * np.sin(right_angle))

    # Define triangle vertices
    triangle_pts = np.array([center, (left_x, left_y), (right_x, right_y)], dtype=np.int32)

    # Draw filled triangle
    cv2.fillPoly(overlay, [triangle_pts], fov_color)

    # Blend overlay with base image
    save_img = cv2.addWeighted(overlay, fov_alpha, save_img, 1 - fov_alpha, 0)

    # Draw triangle outline
    cv2.polylines(save_img, [triangle_pts], isClosed=True, color=fov_color, thickness=2)

    # Draw frontier centroids as dots if provided
    if frontier_centers_2d is not None and len(frontier_centers_2d) > 0:
        for idx, (row, col) in enumerate(frontier_centers_2d):
            # Skip selected frontier for now (will draw it last with different color)
            if selected_frontier_index is not None and idx == selected_frontier_index:
                continue
            # Draw outline
            if outline_width > 0:
                cv2.circle(save_img, (col, row),
                          dot_radius + outline_width, outline_color, -1)
            # Draw dot
            cv2.circle(save_img, (col, row),
                      dot_radius, dot_color, -1)
        
        # Draw selected frontier with different color
        if selected_frontier_index is not None and selected_frontier_index < len(frontier_centers_2d):
            row, col = frontier_centers_2d[selected_frontier_index]
            # Draw outline (blue for selected)
            if outline_width > 0:
                cv2.circle(save_img, (col, row), dot_radius + outline_width, (255, 0, 0), -1)
            # Draw dot (red for selected)
            cv2.circle(save_img, (col, row), dot_radius, (0, 0, 255), -1)

    # Draw arrow for agent heading
    yaw_rad = np.deg2rad(agent_yaw_deg - 90)
    end_x = int(agent_pos_col + arrow_length * np.cos(yaw_rad))
    end_y = int(agent_pos_row + arrow_length * np.sin(yaw_rad))
    cv2.arrowedLine(save_img,
                    (int(agent_pos_col), int(agent_pos_row)),
                    (end_x, end_y),
                    arrow_color,
                    arrow_thickness,
                    tipLength=0.3)

    # if output_size is not None:
    #     assert save_img.shape[0] == output_size

    cv2.imwrite(png_path, save_img)

    return save_img


def write_map_with_arrow(full_map, png_path,
                              agent_pos_row, agent_pos_col, agent_yaw_deg,
                              output_size=None,
                              arrow_color=(0, 0, 255), arrow_length=20, arrow_thickness=2):
    """
    Write full map with an arrow showing agent's heading direction.

    Args:
        full_map: (H, W, C) numpy array with map channels
        png_path: Path to save the image
        agent_pos_row: Agent's row position in the map (y coordinate)
        agent_pos_col: Agent's col position in the map (x coordinate)
        agent_yaw_deg: Agent's heading in degrees (0=right, 90=up, 180=left, 270=down)
        output_size: Optional int, assert output image height matches this size
        arrow_color: BGR color tuple for the arrow (default red: (0, 0, 255))
        arrow_length: Length of the arrow in pixels
        arrow_thickness: Thickness of the arrow lines
    """
    # Create base image
    save_img = np.ones((full_map.shape[0], full_map.shape[1], 3), dtype=np.uint8)
    save_img *= 128
    save_img[full_map[:, :, 1] == 1] = [255, 255, 255]  # explored - white
    save_img[full_map[:, :, 0] == 1] = [0, 0, 0]        # obstacles - black
    save_img[full_map[:, :, 3] == 1] = [0, 0, 255]      # trajectory - blue
    save_img[full_map[:, :, 2] == 1] = [255, 0, 0]      # current position - red

    # Draw arrow for agent heading
    yaw_rad = np.deg2rad(agent_yaw_deg - 90)

    # Calculate arrow end point using standard trigonometry
    end_x = int(agent_pos_col + arrow_length * np.cos(yaw_rad))
    end_y = int(agent_pos_row + arrow_length * np.sin(yaw_rad))

    # Draw arrow
    cv2.arrowedLine(save_img,
                    (int(agent_pos_col), int(agent_pos_row)),  # start point (col, row) = (x, y)
                    (end_x, end_y),  # end point
                    arrow_color,
                    arrow_thickness,
                    tipLength=0.3)

    # assert save_img.shape[0] == output_size

    cv2.imwrite(png_path, save_img)

    return save_img


def write_map_with_arrow_and_frontier(full_map, png_path,
                              agent_pos_row, agent_pos_col, agent_yaw_deg,
                              frontier_centers_2d, selected_frontier_index=None,
                              output_size=None,
                              arrow_color=(0, 0, 255), arrow_length=20, arrow_thickness=2,
                              dot_radius=5, dot_color=(0, 255, 0),
                              outline_color=(255, 255, 255), outline_width=2):
    """
    Write full map with an arrow showing agent's heading direction and frontiers.

    Args:
        full_map: (H, W, C) numpy array with map channels
        png_path: Path to save the image
        agent_pos_row: Agent's row position in the map (y coordinate)
        agent_pos_col: Agent's col position in the map (x coordinate)
        agent_yaw_deg: Agent's heading in degrees (0=right, 90=up, 180=left, 270=down)
        frontier_centers_2d: numpy array (N, 2) - frontier centroids in (row, col) format
        arrow_length: Length of the arrow in pixels
        arrow_thickness: Thickness of the arrow lines
        centroids: Optional numpy array (N, 2) - frontier centroids in (row, col) format
        dot_radius: radius of frontier centroid dots
        dot_color: BGR color for frontier dots
        outline_color: BGR color for dot outline
        outline_width: width of outline
    """
    # Create base image
    save_img = np.ones((full_map.shape[0], full_map.shape[1], 3), dtype=np.uint8)
    save_img *= 128
    save_img[full_map[:, :, 1] == 1] = [255, 255, 255]  # explored - white
    save_img[full_map[:, :, 0] == 1] = [0, 0, 0]        # obstacles - black
    save_img[full_map[:, :, 3] == 1] = [0, 0, 255]      # trajectory - blue
    save_img[full_map[:, :, 2] == 1] = [255, 0, 0]      # current position - red

    # # draw frontier with 3*3
    # if frontier_centers_map is not None:
    #     frontier_dilated = cv2.dilate(frontier_centers_map.astype(np.uint8), np.ones((3,3)))
    #     save_img[frontier_dilated == 1] = [0, 255, 0]

    # Draw frontier centroids as dots if provided
    if frontier_centers_2d is not None:
        assert frontier_centers_2d.shape[0] > 0, "Frontier centers must be provided"
        for row, col in frontier_centers_2d:
            # Draw outline
            if outline_width > 0:
                cv2.circle(save_img, (col, row),
                          dot_radius + outline_width, outline_color, -1)
            # Draw dot
            cv2.circle(save_img, (col, row),
                      dot_radius, dot_color, -1)

    # Draw arrow for agent heading
    yaw_rad = np.deg2rad(agent_yaw_deg - 90)

    # Calculate arrow end point using standard trigonometry
    end_x = int(agent_pos_col + arrow_length * np.cos(yaw_rad))
    end_y = int(agent_pos_row + arrow_length * np.sin(yaw_rad))

    # Draw arrow
    cv2.arrowedLine(save_img,
                    (int(agent_pos_col), int(agent_pos_row)),  # start point (col, row) = (x, y)
                    (end_x, end_y),  # end point
                    arrow_color,
                    arrow_thickness,
                    tipLength=0.3)

    if selected_frontier_index is not None:
        row, col = frontier_centers_2d[selected_frontier_index][0], frontier_centers_2d[selected_frontier_index][1]
        # Draw outline
        if outline_width > 0:
            cv2.circle(save_img, (col, row), dot_radius + outline_width, (0,0,255), -1)
        # Draw dot
        cv2.circle(save_img, (col, row), dot_radius, (255,0,0), -1)

    # assert save_img.shape[0] == output_size

    cv2.imwrite(png_path, save_img)

    return save_img


def write_map_images(save_idx,
                     full_map, local_map,
                     current_y, current_x, agent_yaw_deg, local_y, local_x,
                     full_occupancy_explore_path,
                     local_occupancy_explore_path,
                     full_occupancy_explore_frontier_path,
                     full_occupancy_explore_frontier_gt_path,
                     local_occupancy_explore_frontier_path,
                     local_occupancy_explore_frontier_gt_path,
                     map_resolution,
                     frontier_centers_2d, selected_frontier_index,
                     frontier_centers_2d_local_valid,
                     vis_config=None, if_skip_frontier=False):
    """
    Generate independent BEV visualizations for each stage.
    1. Full map with occupancy and exploration
    2. Local map with occupancy and exploration
    3. Full map with frontiers
    4. Full map with ground truth frontiers

    Args:
        vis_config: Optional dict with visualization settings from config.mapping.visualization
    """
    # Extract visualization config parameters
    if vis_config is None:
        vis_config = {}

    crop_radius = vis_config.get('crop_radius', 150)
    output_size = vis_config.get('output_size')
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



    full_occupancy_explore_pil = write_map_with_arrow(full_map,
                                                      os.path.join(full_occupancy_explore_path, f"{save_idx:03d}.jpg"),
                                                      current_y, current_x, agent_yaw_deg,
                                                      output_size,
                                                      arrow_color, arrow_len_px, arrow_width)

    local_occupancy_explore_pil = write_map_with_arrow(local_map,
                                                       os.path.join(local_occupancy_explore_path, f"{save_idx:03d}.jpg"),
                                                       local_y, local_x, agent_yaw_deg,
                                                       output_size=output_size,
                                                       arrow_color=arrow_color, arrow_length=arrow_len_px, arrow_thickness=arrow_width
                                                    )

    assert local_occupancy_explore_pil.shape[0] == output_size

    if if_skip_frontier:
        return full_occupancy_explore_pil, local_occupancy_explore_pil, None, None, None, None
    full_occupancy_explore_frontiers_pil, full_occupancy_explore_selected_frontiers_pil, local_occupancy_explore_frontiers_pil, local_occupancy_explore_selected_frontiers_pil = None, None, None, None
    full_occupancy_explore_frontiers_pil = write_map_with_arrow_and_frontier(full_map,
                                                           os.path.join(full_occupancy_explore_frontier_path, f"{save_idx:03d}.jpg"),
                                                           current_y, current_x, agent_yaw_deg,
                                                           frontier_centers_2d,
                                                           selected_frontier_index=None,
                                                           output_size=output_size,
                                                           arrow_color=arrow_color, arrow_length=arrow_len_px, arrow_thickness=arrow_width,
                                                           dot_radius=frontier_dot_radius,dot_color=frontier_color,
                                                           outline_color=frontier_outline,outline_width=frontier_width,
                                                           )

    full_occupancy_explore_selected_frontiers_pil = write_map_with_arrow_and_frontier(full_map,
                                                        os.path.join(full_occupancy_explore_frontier_gt_path, f"{save_idx:03d}.jpg"),
                                                        current_y, current_x, agent_yaw_deg,
                                                        frontier_centers_2d,
                                                        selected_frontier_index=selected_frontier_index,
                                                        output_size=output_size,
                                                        arrow_color=arrow_color, arrow_length=arrow_len_px, arrow_thickness=arrow_width,
                                                        dot_radius=frontier_dot_radius,dot_color=frontier_color,
                                                        outline_color=frontier_outline, outline_width=frontier_width,
                                                        )
    
    local_occupancy_explore_frontiers_pil = write_map_with_arrow_and_frontier(local_map,
                                                        os.path.join(local_occupancy_explore_frontier_path, f"local_{save_idx:03d}.jpg"),
                                                        local_y, local_x, agent_yaw_deg,
                                                        frontier_centers_2d_local_valid,
                                                        selected_frontier_index=None,
                                                        output_size=output_size,
                                                        arrow_color=arrow_color, arrow_length=arrow_len_px, arrow_thickness=arrow_width,
                                                        dot_radius=frontier_dot_radius,dot_color=frontier_color,
                                                        outline_color=frontier_outline,outline_width=frontier_width,
                                                        )
    
    local_occupancy_explore_selected_frontiers_pil = write_map_with_arrow_and_frontier(local_map,
                                                        os.path.join(local_occupancy_explore_frontier_gt_path, f"local_{save_idx:03d}.jpg"),
                                                        local_y, local_x, agent_yaw_deg,
                                                        frontier_centers_2d_local_valid,
                                                        selected_frontier_index=selected_frontier_index,
                                                        output_size=output_size,
                                                        arrow_color=arrow_color, arrow_length=arrow_len_px, arrow_thickness=arrow_width,
                                                        dot_radius=frontier_dot_radius,dot_color=frontier_color,
                                                        outline_color=frontier_outline, outline_width=frontier_width,
                                                        )   

    return full_occupancy_explore_pil, local_occupancy_explore_pil, \
        full_occupancy_explore_frontiers_pil, full_occupancy_explore_selected_frontiers_pil, \
        local_occupancy_explore_frontiers_pil, local_occupancy_explore_selected_frontiers_pil





def write_rgbds_images(save_idx,
                       rgb_path, depth_path, semantic_path,
                       rgb_vis, depth_vis,
                       semantic_obj_idx,
                       semantic_room_idx=None, semantic_room_name=None):
    """
    Write RGB, depth, and semantic images to the specified paths.
    """

    cv2.imwrite(f"{rgb_path}/{save_idx:03d}.jpg", rgb_vis)
    cv2.imwrite(f"{depth_path}/{save_idx:03d}.jpg", depth_vis)

    rgb_pil = Image.fromarray(cv2.cvtColor(rgb_vis.astype(np.uint8), cv2.COLOR_BGR2RGB))
    depth_pil = Image.fromarray(depth_vis)

    semantic_pil = save_semantic_annotated_from_masks(f"{semantic_path}/{save_idx:03d}.png",
                                                      semantic_obj_idx)


    return rgb_pil, depth_pil, semantic_pil



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
    output_file = os.path.join(top_down_path, f"{save_idx:03d}.jpg")
    top_down_pil.save(output_file)

    return top_down_pil



def create_combined_visualization(save_path,
                                  save_idx,
                                  rgb_pil, depth_pil, semantic_pil,
                                  bev_1, bev_2, bev_3,
                                  bev_4, top_down_pil=None, episode_info=None):
    """
    Combine all visualizations into one large image.

    Layout (3 rows x 3 columns):
    Row 1: RGB | Depth | Semantic
    Row 2: Full Occupancy with Explored | Full Occupancy with Explored + Frontier Centroids | Full Occupancy with Explored + Selected GT Frontier
    Row 3: FOV Visualization | Top Down Map | Episode Info

    Args:
        save_path: Directory to save the combined image
        save_idx: int index for filename
        8 visualization PIL Images
        episode_info: Optional dict with episode info to display in info panel

    Returns:
        combined_pil: PIL Image with all visualizations
    """
    # Convert all to numpy arrays and ensure RGB
    def pil_to_rgb(img):
        if img is None:
            return None
        arr = np.array(img)
        if arr.ndim == 2:  # grayscale
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:  # RGBA
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr

    # Get all images as RGB arrays (or None)
    rgb_arr = pil_to_rgb(rgb_pil)
    assert rgb_arr is not None
    rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    depth_arr = pil_to_rgb(depth_pil)
    semantic_arr = pil_to_rgb(semantic_pil)
    bev_1_arr = pil_to_rgb(bev_1)
    bev_2_arr = pil_to_rgb(bev_2)
    bev_3_arr = pil_to_rgb(bev_3)

    # All rows: 480x640 each for uniform sizing
    row_h, row_w = rgb_arr.shape[0], rgb_arr.shape[1]  # Assume RGB size as standard (480, 640)

    # Resize helper with placeholder support
    def resize_to_target(arr, h, w):
        if arr is None:
            # Create blank gray placeholder
            return np.ones((h, w, 3), dtype=np.uint8) * 200
        if arr.shape[0] != h or arr.shape[1] != w:
            return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return arr

    def fill_to_target(arr, h, w):
        # resize with equal aspect ratio, and fill the rest with black
        if arr is None:
            return np.ones((h, w, 3), dtype=np.uint8) * 200

        src_h, src_w = arr.shape[0], arr.shape[1]
        if src_h == h and src_w == w:
            return arr

        # Calculate scale factor to fit within target while preserving aspect ratio
        scale = min(w / src_w, h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        # Resize image with preserved aspect ratio
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create black canvas and center the resized image
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    # Resize all images to 480x640
    rgb_arr = fill_to_target(rgb_arr, row_h, row_w)
    depth_arr = fill_to_target(depth_arr, row_h, row_w)
    semantic_arr = fill_to_target(semantic_arr, row_h, row_w)
    bev_1_arr = fill_to_target(bev_1_arr, row_h, row_w)
    bev_2_arr = fill_to_target(bev_2_arr, row_h, row_w)
    bev_3_arr = fill_to_target(bev_3_arr, row_h, row_w)

    # Add titles to each image
    def add_title(arr, title, color=(255, 255, 255), bg_color=(0, 0, 0)):
        h, w = arr.shape[:2]
        title_h = 40
        canvas = np.ones((h + title_h, w, 3), dtype=np.uint8) * 255
        canvas[:title_h] = bg_color
        canvas[title_h:] = arr

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = (title_h + text_h) // 2
        cv2.putText(canvas, title, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        return canvas

    rgb_titled = add_title(rgb_arr, "RGB" if rgb_pil is not None else "RGB (N/A)",
                          color=(255, 255, 255) if rgb_pil is not None else (128, 128, 128))
    depth_titled = add_title(depth_arr, "Depth" if depth_pil is not None else "Depth (N/A)",
                            color=(255, 255, 255) if depth_pil is not None else (128, 128, 128))
    semantic_titled = add_title(semantic_arr, "Semantic" if semantic_pil is not None else "Semantic (N/A)",
                               color=(255, 255, 255) if semantic_pil is not None else (128, 128, 128))
    bev_1_titled = add_title(bev_1_arr, "Local Map + Frontiers" if bev_1 is not None else "Local Map + Frontiers (N/A)",
                                   color=(255, 255, 255) if bev_1 is not None else (128, 128, 128))
    bev_2_titled = add_title(bev_2_arr, "Local Map + Selected Frontiers" if bev_2 is not None else "Local Map + Selected Frontiers (N/A)",
                                  color=(255, 255, 255) if bev_2 is not None else (128, 128, 128))
    bev_3_titled = add_title(bev_3_arr, "Local Map FOV" if bev_3 is not None else "Local Map FOV (N/A)",
                                           color=(255, 255, 255) if bev_3 is not None else (128, 128, 128))

    # Row 3: FOV visualization (col 1) + Top-Down Map (col 2) + Episode Info (col 3)

    bev_4_arr = pil_to_rgb(bev_4)
    bev_4_arr = fill_to_target(bev_4_arr, row_h, row_w)
    bev_4_titled = add_title(bev_4_arr, "Full Map")


    # Add top-down map (col 2)
    if top_down_pil is not None:
        topdown_arr = pil_to_rgb(top_down_pil)
        assert topdown_arr is not None
        topdown_arr = cv2.cvtColor(topdown_arr, cv2.COLOR_RGB2BGR)
        topdown_arr = fill_to_target(topdown_arr, row_h, row_w)
        topdown_titled = add_title(topdown_arr, "Top-Down Map")
    else:
        # Create placeholder for top-down map
        topdown_arr = np.ones((row_h, row_w, 3), dtype=np.uint8) * 200
        topdown_titled = add_title(topdown_arr, "Top-Down Map (N/A)", color=(128, 128, 128))

    # Create info panel for col 3
    info_width = row_w  # Single column width (640)
    info_height = row_h  # Same height as other cells (480)
    info_arr = np.ones((info_height + 40, info_width, 3), dtype=np.uint8) * 255

    # Add title to info panel
    title_h = 40
    info_arr[:title_h] = (0, 0, 0)  # Black title bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    title_text = "Episode Info"
    (text_w, text_h), _ = cv2.getTextSize(title_text, font, font_scale, thickness)
    text_x = (info_width - text_w) // 2
    text_y = (title_h + text_h) // 2
    cv2.putText(info_arr, title_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Add episode info text
    if episode_info:
        font_scale = 0.24  # smaller font to fit more rows
        thickness = 1
        y_offset = 50
        line_height = 22
        x_offset = 10
        column_spacing = 18
        max_line_width = info_width - 2 * x_offset

        current_line = 0
        x_cursor = x_offset

        def clip_text_to_width(text):
            text = text.strip().replace('\n', ' ')
            if not text:
                return text
            (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
            if text_w <= max_line_width:
                return text
            # Reduce characters proportionally and append ellipsis when needed
            approx_chars = max(5, int(len(text) * max_line_width / max(text_w, 1)))
            trimmed = text[:approx_chars - 3].rstrip()
            return f"{trimmed}..."

        for key, value in episode_info.items():
            # Special handling for instruction - wrap across multiple lines
            if key == 'instruction':
                if x_cursor != x_offset:
                    current_line += 1
                    x_cursor = x_offset
                y = y_offset + current_line * line_height
                if y > info_height + 20:
                    break
                cv2.putText(info_arr, f"{key}:", (x_offset, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                current_line += 1

                # Wrap instruction text across multiple lines
                instruction_text = str(value)
                chars_per_line = 55
                for i in range(0, len(instruction_text), chars_per_line):
                    y = y_offset + current_line * line_height
                    if y > info_height + 20:
                        break
                    line_text = instruction_text[i:i+chars_per_line]
                    cv2.putText(info_arr, f"  {line_text}", (x_offset, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                    current_line += 1
                x_cursor = x_offset
            else:
                text = f"{key}: {str(value)}"
                text = clip_text_to_width(text)
                (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)

                if x_cursor + text_w > max_line_width + x_offset:
                    current_line += 1
                    x_cursor = x_offset

                y = y_offset + current_line * line_height
                if y > info_height + 20:
                    break

                cv2.putText(info_arr, text, (x_cursor, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                x_cursor += text_w + column_spacing

    # Combine into grid (3 rows x 3 columns)
    # All rows are now 1920 pixels wide (3 × 640)
    row1 = np.hstack([rgb_titled, depth_titled, semantic_titled])
    row2 = np.hstack([bev_1_titled, bev_2_titled, bev_3_titled])
    row3 = np.hstack([bev_4_titled, topdown_titled, info_arr])  # FOV + Top-Down + Info

    combined = np.vstack([row1, row2, row3])

    # Save the combined image
    output_file = os.path.join(save_path, f"{save_idx:03d}.jpg")
    cv2.imwrite(output_file, combined)
    return Image.fromarray(combined)



def write_all_images(episode_data, episode_info_display=None, if_skip_frontier=True):
    """
    Write all visualization images for a single step.
    1. RGB
    2. Depth
    3. Semantic segmentation
    4. Full occupancy explore map
    5. Local occupancy explore map
    6. Full occupancy explore map with frontiers
    7. Full occupancy explore map with ground truth frontiers
    8. Top-down map from habitat info
    9. with FOV visualization
    10. combined visualization

    Args:
        episode_data: dict containing all visualization data
        episode_info_display: optional pre-formatted dict for episode info panel.
                            If None, will use get_r2r_episode_info_display as default.
    """

    save_idx = episode_data['save_idx']

    # frontier
    if not if_skip_frontier:
        frontier_centers_2d = episode_data['frontier'][0]
        selected_frontier_index = episode_data['frontier'][1]
        frontier_centers_2d_local_valid = episode_data['frontier'][2]
    else:
        frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid = None, None, None

    # mapping
    full_map = episode_data['mapping']['map'][0]
    local_map = episode_data['mapping']['map'][1]
    current_y, current_x, agent_yaw_deg, local_y, local_x = episode_data['mapping']['map'][2:]

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
        paths['semantic'],
        rgb_vis, depth_vis,
        object_segmentation,
        # agent.seg_idx_region.squeeze(-1),
        # agent.seg_name_region.squeeze(-1),
    )

    # map
    full_occupancy_explore_pil, local_occupancy_explore_pil, \
    full_occupancy_explore_frontiers_pil, full_occupancy_explore_selected_frontiers_pil, \
    local_occupancy_explore_frontiers_pil, local_occupancy_explore_selected_frontiers_pil = write_map_images(save_idx,
        full_map, local_map,
        current_y, current_x, agent_yaw_deg, local_y, local_x,
        paths['full_occupancy_explore'],
        paths['local_occupancy_explore'],
        paths['full_occupancy_explore_frontier'],
        paths['full_occupancy_explore_frontier_gt'],
        paths['local_occupancy_explore_frontier'],
        paths['local_occupancy_explore_frontier_gt'],
        episode_data['mapping']['resolution'],
        frontier_centers_2d, selected_frontier_index, 
        frontier_centers_2d_local_valid,
        vis_config=vis_config, if_skip_frontier=if_skip_frontier
    )

    # top-down map from habitat info
    top_down_pil = save_top_down_map(
        save_idx,
        paths['top_down_map'],
        episode_data['infos']
    )


    fov_pil = write_map_with_fov(local_map,
        os.path.join(paths['fov'], f"{save_idx:03d}.jpg"),
        local_y, local_x, agent_yaw_deg,
        map_resolution,
        output_size=vis_config.get('output_size'),
        hfov=vis_config['fov_hfov'],
        sensor_max_range=vis_config['fov_max_depth_m'],
        fov_color=tuple(vis_config['fov_color']),
        fov_alpha=vis_config['fov_alpha'],
        frontier_centers_2d=frontier_centers_2d_local_valid,
        selected_frontier_index=selected_frontier_index,
        arrow_color=vis_config['arrow_color'],
        arrow_length=vis_config['arrow_len_px'],
        arrow_thickness=vis_config['arrow_width'],
        dot_radius=vis_config['frontier_dot_radius'],
        dot_color=vis_config['frontier_color'],
        outline_color=vis_config['frontier_outline'],
        outline_width=vis_config['frontier_width']
    )

    combined_pil = create_combined_visualization(
        paths['combined'],
        save_idx,
        rgb_pil, depth_pil, semantic_pil,
        local_occupancy_explore_frontiers_pil,
        local_occupancy_explore_selected_frontiers_pil,
        fov_pil,
        full_occupancy_explore_selected_frontiers_pil,
        top_down_pil=top_down_pil,
        episode_info=episode_info_display,
    )
    
def write_eval_images(episode_data, episode_info_display="", save_each_figure=False):
    """
    Build and save an evaluation visualization with a 4x4-style layout.

    Layout:
    - Row 1 (4 cells): RGB, Depth, Top-down, Semantic
    - Row 2-3 (8 cells): history images in sequence order
    - Row 4 (1 full-width panel): episode_info_display text

    Args:
        episode_data: dict containing at least
            - save_idx
            - save_paths
            - rgb_vis
            - depth_vis
            - object_segmentation
            - infos
            - history_image (optional, list-like, up to 8 images)
        episode_info_display: string-like text to render in the last row.

    Returns:
        PIL.Image: Combined visualization.
    """
    save_idx = episode_data["save_idx"]
    paths = episode_data["save_paths"]
    selected_pixel = episode_data.get("selected_pixel")

    def draw_selected_pixel_on_rgb(rgb_img, pixel):
        if pixel is None or rgb_img is None:
            return rgb_img

        arr = np.asarray(rgb_img.convert("RGB")).copy()
        h, w = arr.shape[:2]

        try:
            px, py = int(round(pixel[0])), int(round(pixel[1]))
        except Exception:
            return rgb_img

        if px < 0 or px >= w or py < 0 or py >= h:
            return rgb_img

        # Draw a simple red 5x5 square centered at selected pixel.
        x0, x1 = max(0, px - 5), min(w - 1, px + 5)
        y0, y1 = max(0, py - 5), min(h - 1, py + 5)
        arr[y0:y1 + 1, x0:x1 + 1] = (255, 0, 0)

        return Image.fromarray(arr)

    if save_each_figure:
        rgb_pil, depth_pil, semantic_pil = write_rgbds_images(
            save_idx,
            paths["rgb"],
            paths["depth"],
            paths["semantic"],
            episode_data["rgb_vis"],
            episode_data["depth_vis"],
            episode_data["object_segmentation"],
        )
        top_down_pil = save_top_down_map(
            save_idx,
            paths["top_down_map"],
            episode_data["infos"],
        )
    else:
        rgb_pil = Image.fromarray(cv2.cvtColor(episode_data["rgb_vis"].astype(np.uint8), cv2.COLOR_BGR2RGB))
        depth_pil = Image.fromarray(episode_data["depth_vis"].astype(np.uint8))

        # Reuse the same semantic renderer as write_all_images for consistent colors/legend.
        sem_idx = np.asarray(episode_data["object_segmentation"])
        if sem_idx.ndim == 3 and sem_idx.shape[-1] == 1:
            sem_idx = sem_idx.squeeze(-1)
        sem_idx = sem_idx.astype(np.int32)
        with tempfile.NamedTemporaryFile(suffix="_semantic.png", delete=False) as tmp_file:
            tmp_semantic_path = tmp_file.name
        try:
            semantic_pil = save_semantic_annotated_from_masks(tmp_semantic_path, sem_idx)
        finally:
            if os.path.exists(tmp_semantic_path):
                os.remove(tmp_semantic_path)

        # Build top-down map in memory; do not save as an individual figure.
        top_down_pil = None
        map_k = None
        infos = episode_data.get("infos", {})
        if "top_down_map_vlnce" in infos:
            map_k = "top_down_map_vlnce"
        elif "top_down_map" in infos:
            map_k = "top_down_map"
        if map_k is not None:
            td_map = infos[map_k]["map"]
            td_map = vln_maps.colorize_topdown_map(
                td_map,
                infos[map_k]["fog_of_war_mask"],
                fog_of_war_desat_amount=0.75,
            )

            agent_angle = infos[map_k]["agent_angle"]
            if isinstance(agent_angle, (list, tuple, np.ndarray)):
                agent_angle = float(agent_angle[0]) if len(agent_angle) > 0 else 0.0
            else:
                agent_angle = float(agent_angle)

            agent_coord = infos[map_k]["agent_map_coord"]
            if isinstance(agent_coord, np.ndarray):
                agent_coord = agent_coord.flatten().tolist()
            while isinstance(agent_coord, (list, tuple)) and len(agent_coord) > 0 and isinstance(agent_coord[0], (list, tuple)):
                agent_coord = agent_coord[0]
            if isinstance(agent_coord, (list, tuple)) and len(agent_coord) >= 2:
                agent_coord = (int(agent_coord[0]), int(agent_coord[1]))
            else:
                agent_coord = (td_map.shape[0] // 2, td_map.shape[1] // 2)

            td_map = habitat_vis_maps.draw_agent(
                image=td_map,
                agent_center_coord=agent_coord,
                agent_rotation=agent_angle,
                agent_radius_px=max(1, min(td_map.shape[0:2]) // 24),
            )
            if td_map.shape[1] < td_map.shape[0]:
                td_map = np.rot90(td_map, 1)
            if td_map.shape[0] > td_map.shape[1]:
                td_map = np.rot90(td_map, 1)
            if td_map.dtype != np.uint8:
                td_map = np.clip(td_map, 0, 255).astype(np.uint8)
            top_down_pil = Image.fromarray(td_map).convert("RGB")

    rgb_pil = draw_selected_pixel_on_rgb(rgb_pil, selected_pixel)

    history_images = list(episode_data.get("history_images") or [])[:8]

    def pil_to_rgb_array(img):
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return np.array(img.convert("RGB"))
        arr = np.asarray(img)
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    # Set cell size based on RGB frame, fallback to a standard shape.
    rgb_arr = pil_to_rgb_array(rgb_pil)
    if rgb_arr is None:
        cell_h, cell_w = 480, 640
    else:
        cell_h, cell_w = rgb_arr.shape[0], rgb_arr.shape[1]

    def fit_cell(img):
        arr = pil_to_rgb_array(img)
        if arr is None:
            return np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 200

        src_h, src_w = arr.shape[:2]
        if src_h == cell_h and src_w == cell_w:
            return arr

        scale = min(cell_w / src_w, cell_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        y0 = (cell_h - new_h) // 2
        x0 = (cell_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def add_title(arr, title, title_h=36):
        canvas = np.ones((arr.shape[0] + title_h, arr.shape[1], 3), dtype=np.uint8) * 255
        canvas[:title_h] = (0, 0, 0)
        canvas[title_h:] = arr
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
        tx = max(8, (arr.shape[1] - tw) // 2)
        ty = (title_h + th) // 2
        cv2.putText(canvas, title, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return canvas

    # Row 1
    row1_cells = [
        add_title(fit_cell(rgb_pil), "RGB"),
        add_title(fit_cell(depth_pil), "Depth"),
        add_title(fit_cell(top_down_pil), "Top-Down"),
        add_title(fit_cell(semantic_pil), "Semantic"),
    ]
    row1 = np.hstack(row1_cells)

    # Rows 2 & 3: history sequence slots H1..H8
    history_slots = history_images + [None] * (8 - len(history_images))
    row2_cells = [add_title(fit_cell(history_slots[i]), f"History {i + 1}") for i in range(4)]
    row3_cells = [add_title(fit_cell(history_slots[i]), f"History {i + 1}") for i in range(4, 8)]
    row2 = np.hstack(row2_cells)
    row3 = np.hstack(row3_cells)

    # Row 4: full-width info panel
    info_h = row1.shape[0]
    info_w = row1.shape[1]
    info_panel = np.ones((info_h, info_w, 3), dtype=np.uint8) * 255
    info_title_h = 36
    info_panel[:info_title_h] = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(info_panel, "Episode Info", (20, 23), font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    info_text = "" if episode_info_display is None else str(episode_info_display)
    max_text_width = info_w - 40
    wrapped_lines = []

    def wrap_line_to_width(text):
        text = text.rstrip()
        if not text:
            return [""]

        words = text.split(" ")
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            (cand_w, _), _ = cv2.getTextSize(candidate, font, 0.48, 1)
            if cand_w <= max_text_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                    current = word
                else:
                    # No spaces or one long token: split by characters.
                    chunk = ""
                    for ch in word:
                        cand_chunk = chunk + ch
                        (chunk_w, _), _ = cv2.getTextSize(cand_chunk, font, 0.48, 1)
                        if chunk_w <= max_text_width:
                            chunk = cand_chunk
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch
                    current = chunk
        if current:
            lines.append(current)
        return lines

    for raw_line in info_text.split("\n"):
        wrapped_lines.extend(wrap_line_to_width(raw_line))

    y = info_title_h + 28
    for line in wrapped_lines:
        if y >= info_h - 10:
            break
        cv2.putText(info_panel, line, (20, y), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        y += 20

    combined = np.vstack([row1, row2, row3, info_panel])

    output_root = paths.get("combined_eval", paths.get("combined", "."))
    os.makedirs(output_root, exist_ok=True)
    out_file = os.path.join(output_root, f"{save_idx:03d}.jpg")
    cv2.imwrite(out_file, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if save_each_figure:
        each_root = paths.get("eval_each", os.path.join(output_root, "each"))
        os.makedirs(each_root, exist_ok=True)

        fit_cell(rgb_pil)
        rgb_save = os.path.join(each_root, f"{save_idx:03d}_rgb.jpg")
        cv2.imwrite(rgb_save, cv2.cvtColor(fit_cell(rgb_pil), cv2.COLOR_RGB2BGR))

        depth_save = os.path.join(each_root, f"{save_idx:03d}_depth.jpg")
        cv2.imwrite(depth_save, fit_cell(depth_pil))

        topdown_save = os.path.join(each_root, f"{save_idx:03d}_topdown.jpg")
        cv2.imwrite(topdown_save, cv2.cvtColor(fit_cell(top_down_pil), cv2.COLOR_RGB2BGR))

        semantic_save = os.path.join(each_root, f"{save_idx:03d}_semantic.jpg")
        cv2.imwrite(semantic_save, cv2.cvtColor(fit_cell(semantic_pil), cv2.COLOR_RGB2BGR))

        for i in range(8):
            hist_save = os.path.join(each_root, f"{save_idx:03d}_history_{i + 1}.jpg")
            cv2.imwrite(hist_save, cv2.cvtColor(fit_cell(history_slots[i]), cv2.COLOR_RGB2BGR))

    return Image.fromarray(combined)