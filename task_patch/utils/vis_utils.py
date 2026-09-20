
from habitat.utils.visualizations import maps
import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
import textwrap
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt


def convert_depth_to_rgb(depth, min_depth=0.0, max_depth=5.0):
    """Convert depth map to RGB for visualization."""
    depth_vis = depth.copy()
    depth_vis = (depth_vis - min_depth) / (max_depth - min_depth + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)  # Scale to 0-255 uint8
    depth_vis_rgb = np.repeat(depth_vis, 3,
                              axis=2)
    return depth_vis_rgb

def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    """
    display_sample displays the RGB, semantic, and depth observations

    """
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(24, 16))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02, hspace=0.02)
    plt.show(block=False)

def save_display_sample(rgb_obs, semantic_obs, depth_obs, timestep, save_dir="frames_output"):
    """
    Save RGB, colorized semantic, and depth images for the given timestep at original resolution.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save RGB image
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA" if rgb_obs.shape[2] == 4 else "RGB")
    rgb_img.save(os.path.join(save_dir, f"t{timestep:04d}_rgb.png"), quality=100)

    # Save colorized semantic image
    if semantic_obs is not None and semantic_obs.size != 0:
        from habitat_sim.utils.common import d3_40_colors_rgb

        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        semantic_img.save(os.path.join(save_dir, f"t{timestep:04d}_semantic.png"), quality=100)

    # Save depth image
    if depth_obs is not None and depth_obs.size != 0:
        depth_vis = (np.clip(depth_obs, 0, 10) / 10 * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_vis, mode="L")
        depth_img.save(os.path.join(save_dir, f"t{timestep:04d}_depth.png"), quality=100)

# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)

def print_scene_recur(scene, limit_output=10):
    """
    print_scene_recur prints the structure of the scene recursively.
    House
        levels,
        regions
        objects
    """
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center()} dims:{scene.aabb.size()}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center()},"
            f" dims:{level.aabb.size()}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center()}, dims:{region.aabb.size()}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center()}, dims:{obj.aabb.size()}"
                )
                count += 1
                if count >= limit_output:
                    return

# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown
        # important! in navmesh, x,y is horizontal plane, z is the slicing height
        # so we need to convert x,z to px,py
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 2.0
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 15
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final

def is_on_same_floor(height, ref_floor_height, ceiling_height=0.5):
    """
    The return statement in the function checks whether height is
        greater than or equal to the lower bound (ref_floor_height - ceiling_height) and
        less than the upper bound (ref_floor_height + ceiling_height).

    If height lies within this range, the function will return True,
    indicating that the object is on the same floor as the reference floor.
    Otherwise, it returns False.
    """
    return (
        (ref_floor_height - ceiling_height)
        <= height
        < (ref_floor_height + ceiling_height)
    )

def draw_point(sim, top_down_map, position, point_type, point_padding=2):
    t_x, t_y = maps.to_grid(
        position[2],
        position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=sim,
    )
    top_down_map[
        t_x - point_padding : t_x + point_padding + 1,
        t_y - point_padding : t_y + point_padding + 1,
    ] = point_type
    return top_down_map




def draw_bounding_box(
    sim, top_down_map, goal_object_id, ref_floor_height, line_thickness=4
):
    sem_scene = sim.semantic_annotations()
    object_id = goal_object_id

    sem_obj = None
    for object in sem_scene.objects:
        if object.id == object_id:
            sem_obj = object
            break

    center = sem_obj.aabb.center
    x_len, _, z_len = sem_obj.aabb.sizes / 2.0
    # Nodes to draw rectangle
    corners = [
        center + np.array([x, 0, z])
        for x, z in [
            (-x_len, -z_len),
            (-x_len, z_len),
            (x_len, z_len),
            (x_len, -z_len),
            (-x_len, -z_len),
        ]
        if is_on_same_floor(center[1], ref_floor_height=ref_floor_height)
    ]

    map_corners = [
        maps.to_grid(
            p[2],
            p[0],
            (
                top_down_map.shape[0],
                top_down_map.shape[1],
            ),
            sim=sim,
        )
        for p in corners
    ]

    maps.draw_path(
        top_down_map,
        map_corners,
        maps.MAP_TARGET_BOUNDING_BOX,
        line_thickness,
    )
    return top_down_map


def agent_state_to_map_coord(agent_state, top_down_map, env):

    agent_position = agent_state.position
    a_x, a_y = maps.to_grid(
        agent_position[2],
        agent_position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=env.habitat_env.sim,
    )

    map_positions = (a_x, a_y)

    # get the heading angle of the agent
    ref_rotation = agent_state.rotation
    heading_vector = quaternion_rotate_vector(
        ref_rotation.inverse(), np.array([0, 0, -1])
    )
    map_angles = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]

    return map_positions, map_angles