#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import numpy as np

from habitat import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from task_patch.utils.get_config import register_plugins
from task_patch.utils.env_wrapper import SimpleRLEnv


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def shortest_path_example(config_path, output_path, dataset_name):

    # if the output path does not exist, create it
    os.makedirs(output_path, exist_ok=True)
    # if the output path is not empty, delete all files in it
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


    # this is important to Register custom hydra plugin
    register_plugins()
    if dataset_name == "hm3d":
        config = get_config(config_path=config_path,
                            overrides=[
                                "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map",
                            ],
                            )
    elif dataset_name == "ovon":
        config = get_config(config_path=config_path)

    elif dataset_name == "mp3d":
        config = get_config(config_path=config_path,
                            overrides=[
                                "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map",
                            ],
                            )


    with SimpleRLEnv(config=config) as env:
        if len(env.episodes[0].goals) > 0:
            goal_radius = env.episodes[0].goals[0].radius
        else:
            goal_radius = None
        if goal_radius is None:
            goal_radius = config.habitat.simulator.forward_step_size

        dataset = env._env._dataset
        # get the scene, start position, start rotation, and object category
        scene = dataset.episodes[0].scene_id
        # check if scene all the same for all episodes (random sample another scene)
        if dataset.episodes[-1].scene_id.count('/') > 2:
            segments = dataset.episodes[-1].scene_id.split('/')
            segmented_scene = '/'.join(segments[-3:])
        else:
            segmented_scene = dataset.episodes[-1].scene_id
        # if segmented_scene != scene:
        #     raise ValueError("All episodes should be in the same scene")

        print(f"Scene: {scene}")
        print(f"Number of episodes: {len(dataset.episodes)}")

        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        print("Environment creation successful")

        for episode in range(len(env.habitat_env.episodes)):
            env.reset()
            dirname = os.path.join(
                output_path, env.habitat_env.current_episode.episode_id +
                             '_' +
                             env.habitat_env.current_episode.object_category +
                             '_' +
                             str(env.habitat_env.current_episode.start_position[1])
            )
            print('dirname: ', dirname)
            # overwrite the directory if it exists
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)

            print("Agent stepping around inside environment.")
            images = []
            step_count = 0
            while not env.habitat_env.episode_over:
                step_count += 1
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                im = observations["rgb"]
                if "top_down_map" in info:
                    top_down_map = draw_top_down_map(info, im.shape[0])
                else:
                    breakpoint()
                # pad the image to match the size of the top-down map
                if im.shape[0] < top_down_map.shape[0]:
                    im = np.pad(im, ((0, top_down_map.shape[0] - im.shape[0]), (0, 0), (0, 0)), mode='constant')
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)
            images_to_video(images, dirname, "trajectory")
            print("Episode finished")


if __name__ == "__main__":
    # checked
    # dataset = "hm3d"
    # split = "select"

    # dataset = "ovon"
    # split = "select"

    dataset = "mp3d"
    split = "sample"

    if dataset == "hm3d":
        episode_path = f"../data/datasets/objectnav/hm3d/{split}/content/TEEsavR23oF.json.gz"
        config_path = "configs/hm3d_config.yaml"
    elif dataset == "ovon":
        episode_path = f"../data/datasets/{dataset}/hm3d/{split}/content/TEEsavR23oF.json.gz"
        config_path = "configs/ovon_config.yaml"
    elif dataset == "mp3d":
        episode_path = f"../data/datasets/objectnav/objectnav_mp3d_70k/{split}/content/r1Q1Z4BcV1o.json.gz"
        config_path = "configs/mp3d_config.yaml"
    else:
        raise ValueError(f"Dataset {dataset} not implemented")


    output_path = f"./assets/videos/{dataset}/"

    shortest_path_example(config_path, output_path, dataset)
