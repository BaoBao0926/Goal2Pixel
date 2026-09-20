import logging
import os
import shutil
import numpy as np
import cv2
from task_patch.utils.get_config import register_plugins
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from task_patch.utils.vis_utils import append_text_to_image
from habitat.utils.visualizations.utils import observations_to_image, images_to_video
from habitat import get_config
from habitat.utils.visualizations import maps
from task_patch.utils.env_wrapper import FilteredEnv
from task_patch.all_shortest_path_without_clean import draw_top_down_map
from task_patch.utils.vis_utils import draw_point, draw_bounding_box, agent_state_to_map_coord

MAP_THICKNESS_SCALAR = 128
VIDEO_SAVE_NUM = 5

"""
HABITAT_ENV_DEBUG=0;GLOG_minloglevel=2;MAGNUM_LOG=quiet;HABITAT_SIM_LOG=quiet;
"""
def get_goal_radius(env, config):
    if len(env.episodes[0].goals) > 0:
        goal_radius = env.episodes[0].goals[0].radius
    else:
        goal_radius = None
    if goal_radius is None:
        goal_radius = config.habitat.simulator.forward_step_size

    return goal_radius

def shortest_path_cleaned(config_path, output_path, dataset_name):
    # Setup output directory efficiently: clear it if it already exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Register custom hydra plugin
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
    logging.info(f"dataset name: {dataset_name} | split: {config.habitat.dataset.split}")

    with FilteredEnv(config=config) as env:
        total_success = 0
        total_spl = 0
        total_timesteps = 0

        video_save_count = 0

        num_episodes = len(env.habitat_env.episodes)
        logging.info("Replaying {} episodes".format(num_episodes))

        goal_radius = get_goal_radius(env, config)

        # initialize agent
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        for i in range(num_episodes):
            env.reset()
            images = []
            step_index = 1
            total_reward_ep = 0.0

            current_episode = env.habitat_env.current_episode
            scene_id = current_episode.scene_id.split("/")[-1][:-4]
            ep_id = current_episode.episode_id
            object_goal = current_episode.object_category

            previous_xy_location = None

            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                # print("current step: ", step_index)
                # print("current step reward: ", reward)
                # print("current done: ", done)
                # print("distance to goal: ", info["distance_to_goal"], "success: ", info["success"], "spl: ", info["spl"])

                # Generate visualization frame
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                frame = append_text_to_image(frame, f"Find and go to {object_goal}")

                if "top_down_map" in info:
                    top_down_map = draw_top_down_map(info, frame.shape[0])

                else:
                    assert dataset_name == "ovon"
                    top_down_map, previous_xy_location= top_down_map_ovon(env, previous_xy_location, step_index)

                # pad the image to match the size of the top-down map
                if frame.shape[0] < top_down_map.shape[0]:
                    frame = np.pad(frame, ((0, top_down_map.shape[0] - frame.shape[0]), (0, 0), (0, 0)), mode='constant')
                else:
                    top_down_map = np.pad(top_down_map, ((0, frame.shape[0] - top_down_map.shape[0]), (0, 0), (0, 0)), mode='constant')

                output_im = np.concatenate((frame, top_down_map), axis=1)
                images.append(output_im)

                # # # save the images (for debugging)
                # dirname = os.path.join(output_path, f"{scene_id}_{object_goal}_{ep_id}")
                # os.makedirs(dirname, exist_ok=True)
                # cv2.imwrite(f"{dirname}/step_{step_index}.png", output_im)

                step_index += 1
                total_reward_ep += reward

            # Save video only if needed
            if video_save_count < VIDEO_SAVE_NUM:
                # Create output directory for the current episode if a video is being saved
                dirname = os.path.join(output_path, f"{scene_id}_{object_goal}_{ep_id}")
                os.makedirs(dirname, exist_ok=True)

                video_name = f"{int(info['success'])}_{info['spl']:.2f}"
                images_to_video(images, output_dir=dirname, video_name=video_name, fps=5)
                video_save_count += 1

            # Log episode statistics
            print(f"Episode {i} finished, step count: {step_index}")
            print(f"Total reward for trajectory: {total_reward_ep}")
            print(f"Success: {info['success']}, SPL: {info['spl']}")

            total_success += info["success"]
            total_spl += info["spl"]
            total_timesteps += info["elasp_step"]

        # Final summary statistics
        print("SPL: {}, {}, {}".format(total_spl / num_episodes, total_spl, num_episodes))
        print("Success: {}, {}, {}".format(total_success / num_episodes, total_success, num_episodes))


def top_down_map_ovon(env, previous_xy_location, step_count):

    top_down_map = None

    ref_floor_height = env.habitat_env.current_episode.goals[0].position[1]
    agent_state = env.habitat_env.sim.get_agent_state(0)
    line_thickness = int(
        np.round(256 * 2 / MAP_THICKNESS_SCALAR)
    )
    for goal in env.habitat_env.current_episode.goals:
        if top_down_map is None:
            top_down_map = maps.get_topdown_map(
                env.habitat_env.sim.pathfinder,
                height=agent_state.position[1],
                map_resolution=512,
                draw_border=True,
            )

        # Draw the goal point on the map
        top_down_map = draw_point(
            env.habitat_env.sim,
            top_down_map,
            goal.position,
            maps.MAP_TARGET_POINT_INDICATOR,
            point_padding=6,
        )

        # Draw all view points associated with this goal
        for view_point in goal.view_points:
            top_down_map = draw_point(
                env.habitat_env.sim,
                top_down_map,
                view_point.agent_state.position,
                maps.MAP_VIEW_POINT_INDICATOR,
            )

        # Draw the bounding box around the object related to the goal
        draw_bounding_box(
            env.habitat_env.sim,
            top_down_map,
            goal.object_id,
            ref_floor_height
        )

    # convert the agent's state to map coordinates
    map_agent_pos, map_agent_angle = agent_state_to_map_coord(agent_state, top_down_map, env)

    a_x, a_y = map_agent_pos

    # Don't draw over the source point
    if top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
        color = 10 + min(
            step_count * 245 // env.config.environment.max_episode_steps, 245
        )

        thickness = line_thickness
        if previous_xy_location is not None:
            cv2.line(
                top_down_map,
                previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )
    previous_xy_location = (a_y, a_x)

    # Convert the top-down map to RGB
    top_down_map = maps.colorize_topdown_map(top_down_map)

    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,  # image coordinates where to paste the agent
        agent_rotation=map_agent_angle,  # agent's rotation in radians
        agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    )
    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)


    return top_down_map, previous_xy_location


if __name__ == "__main__":
    # checked
    dataset = "hm3d"

    # dataset = "ovon"

    # dataset = "mp3d"

    if dataset == "hm3d":
        config_path = "configs/hm3d_config.yaml"
    elif dataset == "ovon":
        config_path = "configs/ovon_config.yaml"
    elif dataset == "mp3d":
        config_path = "configs/mp3d_config.yaml"
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

    output_path = f"./assets/videos/{dataset}_clean/"

    shortest_path_cleaned(config_path, output_path, dataset)
