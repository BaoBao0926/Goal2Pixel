
from collections import deque
import torch
import random
import sys
import os
import cv2
import hydra
import numpy as np

import json
from omegaconf import DictConfig, OmegaConf
import time
import warnings

# new start
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
from task_patch.utils.get_config import register_plugins
from scripts.agent.if_map_generate_data import IF_Agent_trajectory as IF_Agent
from scripts.run_utils.env_construct import construct_envs, modify_config
from il_patch.register.config import HabitatBaselinesConfigPlugin_BC
from scripts.run_utils.mapping.visualization_refined import write_all_images
from run_utils.mapping.mapping import BEV_Map

from scripts.scripts_for_generating_data.useless_util.find_scene_name import SCENE_LIST, FIRST_THREE

NUMBER_OF_EPISODES  = -1          # whether running first NUMBER_OF_EPISODE episode. -1 means all episodes
IF_RESUME           = True        # whether resume rollout

SAVE_MASKRCNN_SEMANTIC = True
SAVE_GROUNDINGSAM2_SEMANTIC = True
DEBUG_SAVE_PLY = False  # save point clouds to PLY for debugging (set to False for normal runs)
DEBUG_PLY_STEPS = [0, 5, 10, 45]  # which steps to save PLY files for
MAX_STEPS_PER_WP = 400


WRITE_VISUALIZE = True  # visualize the mapping process step by step
SUCCESS_DISTANCE=0.4
# for padding image
IF_SAVE_RGB_PADDED = True
IF_SAVE_RGB_PADDED_DOWN = True
PADDING = 8

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg: "DictConfig"):

    # insert the experiment name into the config
    cfg = patch_exp_name(cfg)

    # Modifies a configuration by inferring some missing keys
    # and makes sure some keys are present and compatible with each other.
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

# for debug whether finish all episodes
def record_scene_episode_to_jsonl(temp_dir, scene_id, ep_id):
    os.makedirs(temp_dir, exist_ok=True)
    record_file = os.path.join(temp_dir, "scene_episode_ids.jsonl")
    entry = {
        "scene_id": scene_id,
        "episode_id": ep_id,
    }
    with open(record_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def execute_exp(config: "DictConfig") -> None:
    # Set seeds for reproducibility
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)

    config = modify_config(config)

    logger.info("Final Config Passed to Env and Agent:\n" + OmegaConf.to_yaml(config, resolve=True))

    # Output 
    if config.habitat.dataset.type == "R2RVLN-v2":
        output_dir = (f"../training_data/data_mp3d_r2r/"
                    f"{config.habitat.dataset.data_path.split('/')[-3]}/"
                    f"{config.habitat.dataset.split}_trajectory")
    elif config.habitat.dataset.type == "RxRVLN-v2":
        output_dir = (f"../training_data/data_mp3d_rxr/"
                    f"{config.habitat.dataset.data_path.split('/')[-3]}_{config.habitat.dataset.ROLES[0]}_{config.habitat.dataset.LANGUAGES[0]}/"
                    f"{config.habitat.dataset.split}_trajectory")
    else:
        raise ValueError(f"Unsupported dataset type: {config.habitat.dataset.type}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Writing images and json to {output_dir} for VLM training.")

    # initialize mapping module
    BEV_map = BEV_Map(config.mapping)

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
    
    start_time = time.time()

    # Setup for history RGBD accumulation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # agent.envs.habitat_env.current_episode = agent.envs.habitat_env.episode_iterator.episodes[5060]
    for i, episode in enumerate(envs.habitat_env.episodes):

        episode_start_time = time.time()
        if i == NUMBER_OF_EPISODES:
            break

        # reset the environment
        obs, rgbd, infos = agent.reset(output_dir)


        record_scene_episode_to_jsonl(
            os.path.join(output_dir, "rollout_records"), agent.scene_id, agent.ep_id
        )
        
        if IF_RESUME:
            if check_exist(output_dir, agent.scene_id, agent.ep_id):
                logger.info(f"Skipping existing episode: Episode {i} / {envs.number_of_episodes} - {agent.scene_id}/{agent.ep_id}")
                continue


        current_episode = envs.habitat_env.current_episode
        # Initialize history buffer for this episode
        history_buffer = deque(maxlen=config.mapping.visualization.history_window_size)
        
        path = current_episode.reference_path + [current_episode.goals[0].position]

        BEV_map.init_map_and_pose(infos)

        save_idx = 0

        # Update map first to get correct initial pose
        current_y, current_x, agent_yaw_deg, local_y, local_x = BEV_map.mapping(rgbd, infos, envs, debug_save_ply=DEBUG_SAVE_PLY, ply_path="", save_idx=save_idx)
        # Add initial observation to history buffer AFTER mapping updates pose
        # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
        agent_x, agent_y, agent_theta_deg = BEV_map.planner_pose_inputs[:3]
        current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)

        history_buffer.append({
            'rgb': obs['rgb'],  # (H, W, 3)
            'depth': rgbd[3:4, :, :].transpose(1, 2, 0),  # (H, W, 1) in meters
            'pose': current_pose
        })


        vis_config = config.mapping.visualization

        # keep episode_data always defined for json logging
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

        if IF_SAVE_RGB_PADDED:
            rgb_base = agent.rgb_vis.copy()
            height, width = rgb_base.shape[:2]
            rgb_padded = np.ones((height, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
            rgb_padded[:, PADDING:PADDING + width, :] = rgb_base
            cv2.imwrite(f"{agent.paths['rgb_padded']}/{save_idx:03d}.jpg", rgb_padded)
            step_data.update({"rgb_padded_path": f"{agent.paths['rgb_padded']}/{save_idx:03d}.jpg"})

        if IF_SAVE_RGB_PADDED_DOWN:
            rgb_base = agent.rgb_vis.copy()
            height, width = rgb_base.shape[:2]
            rgb_padded_down = np.ones((height + 2 * PADDING, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
            rgb_padded_down[0:height, PADDING:PADDING + width, :] = rgb_base
            os.makedirs(agent.paths['rgb_padded_down'], exist_ok=True)
            cv2.imwrite(f"{agent.paths['rgb_padded_down']}/{save_idx:03d}.jpg", rgb_padded_down)
            step_data.update({"rgb_padded_down_path": f"{agent.paths['rgb_padded_down']}/{save_idx:03d}.jpg"})

        if WRITE_VISUALIZE:
            # keep visualization data for visualization logging
            vis_data = {
                "instruction": current_episode.instruction.instruction_text,
                "save_idx": save_idx,
                "mapping": {
                    "config": config.mapping.visualization,
                    "map": [BEV_map.full_map, BEV_map.local_map,
                            current_y, current_x, agent_yaw_deg, local_y, local_x
                                ],
                    "resolution": config.mapping.map_resolution
                },
                "rgb_vis": agent.rgb_vis,
                "depth_vis": agent.depth_vis,
                "top_down_map": infos['top_down_map_vlnce'] if 'top_down_map_vlnce' in infos else infos['top_down_map'],
                "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                "save_paths": agent.paths,
                "infos": infos,
                "action_id": None,
                "action_name": "INIT"
            }
            best_frontier = write_all_images(vis_data, if_skip_frontier=True)
            step_data.update({
                "rgb_path": f"{agent.paths['rgb']}/{save_idx:03d}.jpg",
                "depth_path": f"{agent.paths['depth']}/{save_idx:03d}.jpg",
                "semantic_path": f"{agent.paths['semantic']}/{save_idx:03d}.png",
                "semantic_room_path": f"{agent.paths['semantic_room']}/{save_idx:03d}.jpg",
                "full_occupancy_explore_path": f"{agent.paths['full_occupancy_explore']}/{save_idx:03d}.jpg",
                "local_occupancy_explore_path": f"{agent.paths['local_occupancy_explore']}/{save_idx:03d}.jpg",
                "full_occupancy_explore_frontier_path": f"{agent.paths['full_occupancy_explore_frontier']}/{save_idx:03d}.jpg",
                "full_occupancy_explore_frontier_gt_path": f"{agent.paths['full_occupancy_explore_frontier_gt']}/{save_idx:03d}.jpg",
                "top_down_map_path": f"{agent.paths['top_down_map']}/{save_idx:03d}.jpg",
                "combined_path": f"{agent.paths['combined']}/{save_idx:03d}.jpg",
                "frontier": best_frontier,
            })
        
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

            while not wp_reached and not episode_done:
                if envs.habitat_env.episode_over:
                    episode_done = True
                    break

                best_action = follower.get_next_action(point)

                # Reached this waypoint (or follower says STOP)
                if best_action is None or best_action == HabitatSimActions.stop:
                    if is_final_wp:
                        # Step STOP to terminate the episode within success radius
                        obs, rgbd, done, infos = agent.step(HabitatSimActions.stop)
                        last_infos = infos
                        episode_done = bool(done)
                        episode_steps += 1


                        # Update map first - this updates BEV_map.planner_pose_inputs with current pose
                        current_y, current_x, agent_yaw_deg, local_y, local_x = BEV_map.mapping(rgbd, infos, envs,
                            debug_save_ply=DEBUG_SAVE_PLY, ply_path="",
                            save_idx=save_idx)
                        # # BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
                        agent_x, agent_y, agent_theta_deg = BEV_map.planner_pose_inputs[:3]
                        current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)

                        # Add observation with updated pose to history buffer
                        history_buffer.append({
                            'rgb': obs['rgb'],  # (H, W, 3)
                            'depth': rgbd[3:4, :, :].transpose(1, 2, 0),  # (H, W, 1) in meters
                            'pose': current_pose
                        })

                        
                        #  Stop step logging
                        agent_state = envs.habitat_env.sim.get_agent_state()
                        step_data = {
                            "step_idx": save_idx,
                            "frontier": best_frontier,
                            "agent_position": agent_state.position.tolist(),
                            "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z],
                            "action_id": int(HabitatSimActions.stop),
                            "action_name": "STOP",
                            "waypoint_id": wp_i,
                            "reference_waypoint": point.tolist() if hasattr(point, 'tolist') else list(point),
                            "instruction": current_episode.instruction.instruction_text
                        }

                        if IF_SAVE_RGB_PADDED:
                            rgb_base = agent.rgb_vis.copy()
                            height, width = rgb_base.shape[:2]
                            rgb_padded = np.ones((height, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
                            rgb_padded[:, PADDING:PADDING + width, :] = rgb_base
                            cv2.imwrite(f"{agent.paths['rgb_padded']}/{save_idx:03d}.jpg", rgb_padded)
                            step_data.update({"rgb_padded_path": f"{agent.paths['rgb_padded']}/{save_idx:03d}.jpg"})

                        if IF_SAVE_RGB_PADDED_DOWN:
                            rgb_base = agent.rgb_vis.copy()
                            height, width = rgb_base.shape[:2]
                            rgb_padded_down = np.ones((height + 2 * PADDING, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
                            rgb_padded_down[0:height, PADDING:PADDING + width, :] = rgb_base
                            os.makedirs(agent.paths['rgb_padded_down'], exist_ok=True)
                            cv2.imwrite(f"{agent.paths['rgb_padded_down']}/{save_idx:03d}.jpg", rgb_padded_down)
                            step_data.update({"rgb_padded_down_path": f"{agent.paths['rgb_padded_down']}/{save_idx:03d}.jpg"})

                        if WRITE_VISUALIZE:
                            # snapshot after STOP to capture terminal state
                            vis_data = {
                                "instruction": current_episode.instruction.instruction_text,
                                "save_idx": save_idx,
                                # "frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
                                "mapping": {
                                    "config": config.mapping.visualization,
                                    "map": [BEV_map.full_map, BEV_map.local_map,
                                            current_y, current_x, agent_yaw_deg, local_y, local_x
                                            ],
                                    "resolution": config.mapping.map_resolution
                                },
                                "rgb_vis": agent.rgb_vis,
                                "depth_vis": agent.depth_vis,
                                "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                                "save_paths": agent.paths,
                                "infos": infos,
                                "action_id": action_id,
                                "action_name": action_name,
                            }
                            best_frontier = write_all_images(vis_data, if_skip_frontier=True)
                            step_data.update({
                                "rgb_path": f"{agent.paths['rgb']}/{save_idx:03d}.jpg",
                                "depth_path": f"{agent.paths['depth']}/{save_idx:03d}.jpg",
                                "semantic_path": f"{agent.paths['semantic']}/{save_idx:03d}.png",
                                "semantic_room_path": f"{agent.paths['semantic_room']}/{save_idx:03d}.jpg",
                                "full_occupancy_explore_path": f"{agent.paths['full_occupancy_explore']}/{save_idx:03d}.jpg",
                                "local_occupancy_explore_path": f"{agent.paths['local_occupancy_explore']}/{save_idx:03d}.jpg",
                                "full_occupancy_explore_frontier_path": f"{agent.paths['full_occupancy_explore_frontier']}/{save_idx:03d}.jpg",
                                "full_occupancy_explore_frontier_gt_path": f"{agent.paths['full_occupancy_explore_frontier_gt']}/{save_idx:03d}.jpg",
                                "top_down_map_path": f"{agent.paths['top_down_map']}/{save_idx:03d}.jpg",
                                "combined_path": f"{agent.paths['combined']}/{save_idx:03d}.jpg",
                                "frontier": best_frontier,
                            })
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

                if not episode_done:
                    current_y, current_x, agent_yaw_deg, local_y, local_x = BEV_map.mapping(rgbd, infos, envs,
                        debug_save_ply=DEBUG_SAVE_PLY, ply_path="",
                        save_idx=save_idx)
                    # Add initial observation to history buffer AFTER mapping updates pose
                    # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
                    agent_x, agent_y, agent_theta_deg = BEV_map.planner_pose_inputs[:3]
                    current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
                    
                    # main loop logging
                    agent_state = envs.habitat_env.sim.get_agent_state()
                    step_data = {
                        "step_idx": save_idx,
                        "frontier": best_frontier,
                        "agent_position": agent_state.position.tolist(),
                        "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z],
                        "action_id": action_id,
                        "action_name": action_name,
                        "waypoint_id": wp_i,
                        "reference_waypoint": point.tolist() if hasattr(point, 'tolist') else list(point),
                        "instruction": current_episode.instruction.instruction_text
                    }
 

                    if IF_SAVE_RGB_PADDED:
                        rgb_base = agent.rgb_vis.copy()
                        height, width = rgb_base.shape[:2]
                        rgb_padded = np.ones((height, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
                        rgb_padded[:, PADDING:PADDING + width, :] = rgb_base
                        cv2.imwrite(f"{agent.paths['rgb_padded']}/{save_idx:03d}.jpg", rgb_padded)
                        step_data.update({"rgb_padded_path": f"{agent.paths['rgb_padded']}/{save_idx:03d}.jpg"})

                    if IF_SAVE_RGB_PADDED_DOWN:
                        rgb_base = agent.rgb_vis.copy()
                        height, width = rgb_base.shape[:2]
                        rgb_padded_down = np.ones((height + 2 * PADDING, width + 2 * PADDING, 3), dtype=rgb_base.dtype) * 128
                        rgb_padded_down[0:height, PADDING:PADDING + width, :] = rgb_base
                        os.makedirs(agent.paths['rgb_padded_down'], exist_ok=True)
                        cv2.imwrite(f"{agent.paths['rgb_padded_down']}/{save_idx:03d}.jpg", rgb_padded_down)
                        step_data.update({"rgb_padded_down_path": f"{agent.paths['rgb_padded_down']}/{save_idx:03d}.jpg"})

                    if WRITE_VISUALIZE:
                        vis_data = {
                            "instruction": current_episode.instruction.instruction_text,
                            "save_idx": save_idx,
                            #"frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
                            "mapping": {
                                "config": config.mapping.visualization,
                                "map": [BEV_map.full_map, BEV_map.local_map,
                                        current_y, current_x, agent_yaw_deg, local_y, local_x
                                        ],
                                "resolution": config.mapping.map_resolution
                            },
                            "rgb_vis": agent.rgb_vis,
                            "depth_vis": agent.depth_vis,
                            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                            "save_paths": agent.paths,
                            "infos": infos,
                            "action_id": action_id,
                            "action_name": action_name,
                            "history_data": {
                                "buffer": history_buffer,
                                "current_pose": current_pose,
                                "camera_matrix": BEV_map.intrinsic_matrix,  # Use camera matrix from mapping
                                "sensor_height": BEV_map.agent_height,  # Camera height from mapping config
                                "camera_elevation": 0.0,  # Habitat default (pitch angle)
                                "device": device,
                                "output_size": (BEV_map.screen_h, BEV_map.screen_w),  # Use mapping screen size
                                "window_size": vis_config.history_window_size,  # From config
                                "save_interval": vis_config.history_save_interval,  # From config
                                "sparse_factor_old": vis_config.history_sparse_factor_old  # From config
                            }
                        }
                        best_frontier = write_all_images(vis_data, if_skip_frontier=True)
                        step_data.update({
                            "rgb_path": f"{agent.paths['rgb']}/{save_idx:03d}.jpg",
                            "depth_path": f"{agent.paths['depth']}/{save_idx:03d}.jpg",
                            "semantic_path": f"{agent.paths['semantic']}/{save_idx:03d}.png",
                            "semantic_room_path": f"{agent.paths['semantic_room']}/{save_idx:03d}.jpg",
                            "full_occupancy_explore_path": f"{agent.paths['full_occupancy_explore']}/{save_idx:03d}.jpg",
                            "local_occupancy_explore_path": f"{agent.paths['local_occupancy_explore']}/{save_idx:03d}.jpg",
                            "full_occupancy_explore_frontier_path": f"{agent.paths['full_occupancy_explore_frontier']}/{save_idx:03d}.jpg",
                            "full_occupancy_explore_frontier_gt_path": f"{agent.paths['full_occupancy_explore_frontier_gt']}/{save_idx:03d}.jpg",
                            "top_down_map_path": f"{agent.paths['top_down_map']}/{save_idx:03d}.jpg",
                            "combined_path": f"{agent.paths['combined']}/{save_idx:03d}.jpg",
                            "frontier": best_frontier,
                        })
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
