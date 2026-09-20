import habitat
# from habitat.core.environments import GymHabitatEnv
from habitat.datasets import make_dataset
from habitat.sims import make_sim
import habitat_sim
from habitat.config.read_write import read_write
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat.utils.visualizations import maps
from task_patch.utils.vis_utils import draw_point, draw_bounding_box
from task_patch.utils.dataset_utils import load_dataset
import os
import gzip
import json
import cv2
import numpy as np

MAP_RESOLUTION = 512


def get_config(dataset_name, scene, dataset, split):

    if dataset_name == "ovon" or dataset_name == "hm3d":
        SCENES_ROOT = "../data/scene_datasets/hm3d_v0.2"

        if dataset_name == "ovon":
            config = habitat.get_config(
                config_path="configs/ovon_config.yaml",
                overrides=[
                    "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map",
                ], )
        else: # hm3d
            config = habitat.get_config(
                config_path="configs/hm3d_config.yaml",
                overrides=[
                    "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map",
                ], )
        with read_write(config):
            # Configuring the dataset properties
            config.habitat.dataset.split = split
            # Configuring the simulator properties
            scene = scene.replace('//', '/')
            if scene.count('/') > 2:
                segments = scene.split('/')
            config.habitat.simulator.scene = '/'.join(segments[-3:])

            # add semantic sensor to the agent for better visualization
            agent_config = get_agent_config(config.habitat.simulator)
            agent_config.sim_sensors.update(
                {"semantic_sensor": HabitatSimSemanticSensorConfig()
                 }
            )
    elif dataset_name == "mp3d":
        SCENES_ROOT = "../data/scene_datasets/mp3d"

        SCENE_DATASET_CFG = os.path.join(
            SCENES_ROOT, "mp3d.scene_dataset_config.json"
        )

        config = habitat.get_config(
            config_path="benchmark/nav/objectnav/objectnav_mp3d.yaml",
            overrides=[
                "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map",
            ], )
        with read_write(config):
            # Configuring the dataset properties
            config.habitat.dataset.max_replay_steps = 5000
            config.habitat.dataset.split = split
            config.habitat.dataset.type = "ObjectNav-mp3d"
            config.habitat.dataset.data_path = f"../data/datasets/objectnav/objectnav_mp3d_70k/{split}/{split}.json.gz"
            config.habitat.dataset.scenes_dir = "../data/scene_datasets/"

            # Configuring the simulator properties
            config.habitat.simulator.scene = scene[5:].replace('//', '/')
            config.habitat.simulator.scene_dataset = SCENE_DATASET_CFG

            config.habitat.simulator.habitat_sim_v0.physics_config_file = "../data/default.physics_config.json"
            config.habitat.simulator.habitat_sim_v0.enable_physics = True
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1  # Headless mode

            # add semantic sensor to the agent for better visualization
            agent_config = get_agent_config(config.habitat.simulator)
            agent_config.sim_sensors.update(
                {"semantic_sensor": HabitatSimSemanticSensorConfig()
                 }
            )
        pass

    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return config


def get_sim(scene, config):

    sim = make_sim("Sim-v0", config=config.habitat.simulator)

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()

    # idk why but the agent radius and height are not read from the config
    # seems nothing change
    navmesh_settings.agent_radius = (
        config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        config.habitat.simulator.agents.main_agent.height
    )

    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    return sim

def visualize_episodes(sim, episodes, goals, object_category):
    top_down_maps = []

    """
    Grouping Goals by Floor Height
        
        - This part of the code groups the goal points based on their height (y-coordinate) 0.25m
        - determine which episodes have goals at approximately the same height as the agent’s start position, ensuring that the start and end points are effectively on the same floor
    """
    grouped_goal_heights = []
    grouped_goals = []

    for i in range(len(goals)):
        goal = goals[i]
        group_exists = False
        for idx, heights in enumerate(grouped_goal_heights):
            if abs(goal["position"][1] - heights[0]) <= 0.25:
                group_exists = True
                grouped_goal_heights[idx].append(goal["position"][1])
                grouped_goals[idx].append(goal)
                break
        if not group_exists:
            grouped_goal_heights.append([goal["position"][1]])
            grouped_goals.append([goal])

    print("Category: {}".format(object_category))
    print(
        "Grouped goals: {}, goals: {}".format(
            [len(g) for g in grouped_goals], len(goals)
        )
    )

    for grouped_goal in grouped_goals:
        ref_floor_height = grouped_goal[0]["position"][1]
        top_down_map = None
        goal_height = grouped_goal[0]["view_points"][0]["agent_state"]["position"][1]

        for goal in grouped_goal:
            if top_down_map is None:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder,
                    height=goal_height,
                    map_resolution=MAP_RESOLUTION,
                    draw_border=True,
                )

            top_down_map = draw_point(
                sim,
                top_down_map,
                goal["position"],
                maps.MAP_TARGET_POINT_INDICATOR,
                point_padding=6,
            )
            for view_point in goal["view_points"]:
                top_down_map = draw_point(
                    sim,
                    top_down_map,
                    view_point["agent_state"]["position"],
                    maps.MAP_VIEW_POINT_INDICATOR,
                )

            if "object_name" in goal:
                # hm3d
                draw_bounding_box(sim, top_down_map, goal["object_name"], ref_floor_height)
            else:
                # ovon
                draw_bounding_box(sim, top_down_map, goal["object_id"], ref_floor_height)

        """
        Filtering Episodes by Floor Height
        """
        for episode in episodes:
            if episode["object_category"] != object_category:
                continue

            on_same_floor = False
            for goal in grouped_goal:
                if abs(episode["start_position"][1] - goal_height) <= 0.25:
                    on_same_floor = True
                    break

            # If the starting point is not on the same floor as the goal point, the code proceeds to skip that episode:
            if not on_same_floor:
                print(f"Skipping episode {episode['episode_id']} due to floor mismatch.")
                continue

            if top_down_map is None:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder,
                    height=episode["start_position"][1],
                    map_resolution=MAP_RESOLUTION,
                    draw_border=True,
                )

            draw_point(
                sim,
                top_down_map,
                episode["start_position"],
                maps.MAP_SOURCE_POINT_INDICATOR,
            )

        if top_down_map is None:
            continue

        top_down_map = maps.colorize_topdown_map(top_down_map)
        top_down_maps.append(top_down_map)

    print(
        "Grouped goals: {}, top down maps: {}".format(
            len(grouped_goals), len(top_down_maps)
        )
    )

    return top_down_maps


def save_visual(img, path):
    cv2.imwrite(path, img)


def visualize(dataset_name, split, episodes_path, output_path):
    dataset = load_dataset(episodes_path)

    os.makedirs(output_path, exist_ok=True)

    # get the scene, start position, start rotation, and object category
    scene = dataset["episodes"][0]["scene_id"]
    # check if scene all the same for all episodes (random sample another scene)
    if dataset["episodes"][-1]["scene_id"] != scene:
        raise ValueError("All episodes should be in the same scene")

    start_position = dataset["episodes"][0]["start_position"]
    start_rotation = dataset["episodes"][0]["start_rotation"]
    object_category = dataset["episodes"][0]["object_category"]
    print(f"Scene: {scene}")
    print(f"Start Position: {start_position}")
    print(f"Start Rotation: {start_rotation}")
    print(f"Object Categories: {object_category}")

    # get and modify the config
    config = get_config(dataset_name, scene, dataset, split)

    # make simulators
    sim = get_sim(scene, config)

    categories = dataset["goals_by_category"].keys()
    for category in categories:
        top_down_maps = visualize_episodes(
            sim,
            dataset["episodes"],
            dataset["goals_by_category"][category],
            object_category=category.split("_")[1],
        )
        for i, top_down_map in enumerate(top_down_maps):
            object_output_path = os.path.join(
                output_path, "{}_{}.png".format(category, i)
            )
            print(object_output_path, category)
            save_visual(top_down_map, object_output_path)


if __name__ == "__main__":
    # dataset = "ovon"
    # split = "select"

    # # checked
    dataset = "hm3d"
    split = "select"

    # dataset = "mp3d"
    # split = "sample"

    if dataset == "hm3d":
        episode_path = f"../data/datasets/objectnav/hm3d/{split}/content/TEEsavR23oF.json.gz"
    elif dataset == "ovon":
        episode_path = f"../data/datasets/{dataset}/hm3d/{split}/content/TEEsavR23oF.json.gz"
    elif dataset == "mp3d":
        episode_path = f"../data/datasets/objectnav/objectnav_mp3d_70k/{split}/content/r1Q1Z4BcV1o.json.gz"
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

    output_path = f"./assets/images/{dataset}/"

    visualize(dataset, split, episode_path, output_path)