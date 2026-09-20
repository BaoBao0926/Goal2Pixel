from collections import deque
from argparse import ArgumentParser
import torch
import random
import sys
import os
import time
import cv2
import hydra
import numpy as np

import json
from omegaconf import DictConfig, OmegaConf
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
from task_patch.utils.get_config import register_plugins
from scripts.agent.if_map_generate_data import IF_Agent_history_SIFT as IF_Agent
from scripts.run_utils.env_construct import construct_envs, modify_config
from il_patch.register.config import HabitatBaselinesConfigPlugin_BC
from scripts.run_utils.mapping.visualization_refined import write_all_images
# from scripts.policy.Data_Generation.History.history import History
from scripts.run_utils.mapping.mapping_utils import get_camera_matrix


WRITE_VISUALIZE         = False         # visualize the mapping process step by step
WRITE_HISTORY_VISUALIZE = True          # This should be true
SUCCESS_DISTANCE=0.4


NUMBER_OF_EPISODES  = -1          # whether running first NUMBER_OF_EPISODE episode. -1 means all episodes
IF_RESUME           = True        # whether resume rollout

SAVE_MASKRCNN_SEMANTIC = True
SAVE_GROUNDINGSAM2_SEMANTIC = True
DEBUG_SAVE_PLY = False  # save point clouds to PLY for debugging (set to False for normal runs)
DEBUG_PLY_STEPS = [0, 5, 10, 45]  # which steps to save PLY files for
MAX_STEPS_PER_WP = 400


def _yaw_deg_from_world_rotation(world_rotation) -> float:
    """Compute yaw (degrees) from quaternion-like rotation with w/x/y/z fields."""
    if world_rotation is None:
        return 0.0

    w = float(getattr(world_rotation, "w", 1.0))
    x = float(getattr(world_rotation, "x", 0.0))
    y = float(getattr(world_rotation, "y", 0.0))
    z = float(getattr(world_rotation, "z", 0.0))

    # Standard yaw extraction from quaternion.
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))


def build_surrogate_planner_pose_inputs(world_pos, world_rotation):
    """Build planner-pose-like inputs without relying on BEV map internals.

    Returns a 7-element list compatible with existing history interfaces:
    [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
    """
    wp = np.asarray(world_pos, dtype=np.float32)
    if wp.shape[0] >= 3:
        x_m = float(wp[0])
        y_m = float(wp[2])
    elif wp.shape[0] == 2:
        x_m = float(wp[0])
        y_m = float(wp[1])
    else:
        x_m = 0.0
        y_m = 0.0

    theta_deg = _yaw_deg_from_world_rotation(world_rotation)
    return [x_m, y_m, theta_deg, 0, 0, 0, 0]


def _parse_custom_args():
    """Parse custom script args before Hydra consumes argv."""
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--hist-alg",
        choices=["cv", "og"],
        default="og",
        help="History algorithm mode.",
    )
    args, unknown = parser.parse_known_args()
    # Keep only args intended for Hydra.
    sys.argv = [sys.argv[0]] + unknown
    return args


_CUSTOM_ARGS = _parse_custom_args()


def get_hist_alg() -> str:
    return _CUSTOM_ARGS.hist_alg


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

def save_history_images(agent, fpv_imgs, save_idx, history_path=None, cv_straight_path = None, cv_turn_path = None, step_type=None):
    """
        Save FPV images list to agent paths.
        Returns history_saved_paths (ordered from most recent to oldest)
    """
    history_saved_paths = []
    if history_path is None:
        history_path = agent.his_trajectory_colored_path
    his_dir = os.path.join(history_path, f"{save_idx:03d}")
    os.makedirs(his_dir, exist_ok=True)
    for ii, im in enumerate(fpv_imgs):
        if im is None:
            continue
        try:
            arr = np.asarray(im)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            try:
                im_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                im_bgr = arr[..., ::-1].copy()
            p = os.path.join(his_dir, f"{ii:03d}.jpg")
            cv2.imwrite(p, im_bgr)
            history_saved_paths.append(p)
        except Exception:
            continue

    # Reverse the order (most recent to oldest)
    history_saved_paths = list(reversed(history_saved_paths))

    return history_saved_paths

def save_history_type_images(agent, fpv_imgs, save_idx, step_types : list, cv_straight_path=None, cv_turn_path=None ):
    linear_history_saved_paths = []
    turn_history_saved_paths = []
    if cv_straight_path is None:
        cv_straight_path = agent.his_trajectory_colored_path
    if cv_turn_path is None:
        cv_turn_path = agent.his_trajectory_colored_path   
    straight_dir = os.path.join(cv_straight_path, f"{save_idx:03d}")
    turn_dir = os.path.join(cv_turn_path, f"{save_idx:03d}")
    os.makedirs(straight_dir, exist_ok=True)
    os.makedirs(turn_dir, exist_ok=True)
    for ii, im in enumerate(fpv_imgs):
        if im is None:
            continue
        try:
            arr =  np.asarray(im)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            try:               
                im_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)   
            except Exception:
                im_bgr = arr[..., ::-1].copy()
            if step_types[ii] == "STRAIGHT":
                p = os.path.join(straight_dir, f"{ii:03d}.png")
                linear_history_saved_paths.append(p)
            elif step_types[ii] == "TURN":
                p = os.path.join(turn_dir, f"{ii:03d}.png")
                turn_history_saved_paths.append(p)
            else:
                # indeterminate step type, skip saving
                continue
            cv2.imwrite(p, im_bgr)
        except Exception:
            continue
    # Reverse the order (most recent to oldest)
    linear_history_saved_paths = list(reversed(linear_history_saved_paths))
    turn_history_saved_paths = list(reversed(turn_history_saved_paths))
    return linear_history_saved_paths, turn_history_saved_paths

def save_bev_image(agent, local_map_img, full_map_img, save_idx):

    def _prepare_img(img):
        if img is None:
            return None
        try:
            arr = np.asarray(img)
        except Exception:
            return None
        # Reject empty arrays
        if arr.size == 0:
            return None
        # Ensure 3-channel
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4):
            # Unsupported channel count
            return None
        # If single channel, expand to 3
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        # Normalize dtype to uint8 [0,255]
        if arr.dtype != np.uint8:
            if arr.dtype.kind in ('f', 'c'):
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        # Convert RGB->BGR for OpenCV expectations
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            try:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                # Fallback swap if cv2.cvtColor fails
                arr = arr[..., ::-1].copy()
        # Ensure contiguous memory
        return np.ascontiguousarray(arr)

    local_arr = _prepare_img(local_map_img)
    full_arr = _prepare_img(full_map_img)

    local_dir = agent.paths['local_occupancy_explored_colored'] if local_arr is not None else None
    full_dir = agent.paths['full_occupancy_explored_colored'] if full_arr is not None else None

    # Create directories only if they are valid (non-None)
    if local_dir is not None:
        os.makedirs(local_dir, exist_ok=True)
    if full_dir is not None:
        os.makedirs(full_dir, exist_ok=True)

    local_saved_path = None
    full_saved_path = None

    if local_arr is not None and local_dir is not None:
        local_saved_path = os.path.join(local_dir, f"{save_idx:03d}.png")
        try:
            if not cv2.imwrite(local_saved_path, local_arr):
                logger.warning(f"cv2.imwrite returned False for {local_saved_path}")
                local_saved_path = None
        except Exception as e:
            logger.warning(f"Failed to write local BEV image at {local_saved_path}: {e}")
            local_saved_path = None

    if full_arr is not None and full_dir is not None:
        full_saved_path = os.path.join(full_dir, f"{save_idx:03d}.png")
        try:
            if not cv2.imwrite(full_saved_path, full_arr):
                logger.warning(f"cv2.imwrite returned False for {full_saved_path}")
                full_saved_path = None
        except Exception as e:
            logger.warning(f"Failed to write full BEV image at {full_saved_path}: {e}")
            full_saved_path = None

    return local_saved_path, full_saved_path



def execute_exp(config: "DictConfig") -> None:
    # Set seeds for reproducibility
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)

    config = modify_config(config)

    logger.info("Final Config Passed to Env and Agent:\n" + OmegaConf.to_yaml(config, resolve=True))
    hist_alg = get_hist_alg()
    logger.info(f"Using history algorithm: {hist_alg}")
    # Output organization
    if config.habitat.dataset.type == "R2RVLN-v2":
        if hist_alg == "og": # original algorithm, based on visibility
            output_dir = (f"../training_data/data_mp3d_r2r/"
                        f"{config.habitat.dataset.data_path.split('/')[-3]}/"
                        f"{config.habitat.dataset.split}_history")
        elif hist_alg == "cv":
            output_dir = (f"../training_data/data_mp3d_r2r/"
                        f"{config.habitat.dataset.data_path.split('/')[-3]}/"
                        f"{config.habitat.dataset.split}_history_SIFT")
        else:
            raise ValueError(f"Unsupported hist_alg: {hist_alg}")
    elif config.habitat.dataset.type == "RxRVLN-v2":
        if hist_alg == "og": # original algorithm, based on visibility
            output_dir = (f"../training_data/data_mp3d_rxr/"
                    f"{config.habitat.dataset.data_path.split('/')[-3]}_{config.habitat.dataset.ROLES[0]}_{config.habitat.dataset.LANGUAGES[0]}/"
                    f"{config.habitat.dataset.split}_history")
        elif hist_alg == "cv": 
            output_dir = (f"../training_data/data_mp3d_rxr/"
                    f"{config.habitat.dataset.data_path.split('/')[-3]}_{config.habitat.dataset.ROLES[0]}_{config.habitat.dataset.LANGUAGES[0]}/"
                    f"{config.habitat.dataset.split}_history_SIFT")
        else:
            raise ValueError(f"Unsupported hist_alg: {hist_alg}")
    else:
        raise ValueError(f"Unsupported dataset type: {config.habitat.dataset.type}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Writing images and json to {output_dir} for VLM training.")


    # construct the environment
    envs = construct_envs(config)

    # construct the agent
    agent = IF_Agent(config, envs)

    # build camera matrix + History helper
    camera_matrix = get_camera_matrix(
        config.mapping.frame_width,
        config.mapping.frame_height,
        config.mapping.hfov,
        )
    # build history module
    # set HIST_ALG=cv or HIST_ALG=og from shell to switch mode
    if hist_alg == "og":
        from scripts.policy.Data_Generation.History.version2.history import History
        history = History(
            camera_matrix,
            config.mapping.camera_height,
            config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov,
            config.mapping.map_resolution,
            config.mapping.visualization,
            image_height=config.mapping.frame_height,
            image_width=config.mapping.frame_width,
            hist_alg = hist_alg
        )
    elif hist_alg == "cv":
        from scripts.policy.Data_Generation.History.version2.history import History
        history = History(
            camera_matrix,
            config.mapping.camera_height,
            config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov,
            config.mapping.map_resolution,
            config.mapping.visualization,
            image_height=config.mapping.frame_height,
            image_width=config.mapping.frame_width,
            hist_alg = hist_alg
        )
    else:
        raise ValueError(f"Unsupported hist_alg: {hist_alg}")
    
    possible_action_env = envs.habitat_env.task.actions.keys()

    follower = ShortestPathFollower(
        envs.habitat_env.sim,
        goal_radius=max(1e-3, SUCCESS_DISTANCE - 1e-3),
        # goal_radius=max(1e-3, 0.2 - 1e-3),
        return_one_hot=False
    )

    start_time = time.time()

    for i, episode in enumerate(envs.habitat_env.episodes):
        episode_start_time = time.time()
        # debug code to determien how many episodes to run
        if i == NUMBER_OF_EPISODES:
            break

        # reset the environment
        obs, rgbd, infos = agent.reset(output_dir)

        if IF_RESUME:   # check if already exists
            if check_exist(output_dir, agent.scene_id, agent.ep_id):
                logger.info(f"Skipping existing episode: Episode {i} / {envs.number_of_episodes} - {agent.scene_id}/{agent.ep_id}")
                continue

        current_episode = envs.habitat_env.current_episode
        # reference path from datasets
        path = current_episode.reference_path + [current_episode.goals[0].position]

        save_idx = 0

        planner_pose_inputs = build_surrogate_planner_pose_inputs(
            obs.get('world_pos', agent.start_position),
            obs.get('world_rotation', agent.start_rotation),
        )
        # Reset history for this episode using world pose + raw rgb/depth.
        history.reset(obs.get('world_pos', agent.start_position), obs.get('world_rotation', agent.start_rotation), 
                      rgbd[:, :, 0:3], rgbd[:, :, 3], planner_pose_inputs)
      
        # keep episode_data always defined for json logging
        episode_data = {
            "episode_id": agent.ep_id,
            "scene_id": agent.scene_id,
            "instruction": current_episode.instruction.instruction_text,
            "steps": []
        }

        # here is time step 0(can not save history image when step_idx=0)
        
        agent_state = envs.habitat_env.sim.get_agent_state()
        step_data = {
            "step_idx": save_idx,
            "instruction": current_episode.instruction.instruction_text,
            "agent_position": agent_state.position.tolist(),
            "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z],
            "action_id": None,
            "action_name": "INIT"
        }
        # keep visualization data for visualization logging
        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": save_idx,
            "rgb_vis": agent.rgb_vis,
            "depth_vis": agent.depth_vis,
            "top_down_map": infos['top_down_map_vlnce'] if 'top_down_map_vlnce' in infos else infos['top_down_map'],
            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
            "save_paths": agent.paths,
            "infos": infos,
            "action_id": None,
            "action_name": "INIT",
        }

        if WRITE_HISTORY_VISUALIZE:
            results = history.get_hist_img(
                episode_data=vis_data,
                length=1000,
            )
            fpv_imgs_multi = results.get('fpv_imgs_multi_colored')
            fpv_imgs_blue = results.get('fpv_imgs_blue')
            fpv_imgs_no_draw = results.get('fpv_imgs_no_draw')
            full_map_img = results.get('full_map_img')
            local_map_img = results.get('local_map_img')
            step_types = results.get('step_types')
          

            history_saved_paths_multi = save_history_images(agent, fpv_imgs_multi, save_idx, agent.paths['his_tra_multip_colored'])
            history_saved_paths_blue = save_history_images(agent, fpv_imgs_blue, save_idx, agent.paths['his_tra_single_colored'])
            history_saved_paths_no_draw = save_history_images(agent, fpv_imgs_no_draw, save_idx, agent.paths['his_tra_without_colored'])
            local_bev_path, full_bev_path = save_bev_image(agent, local_map_img, full_map_img, save_idx)

            if hist_alg == "cv":
                straight_history_saved_paths, turn_history_saved_paths = save_history_type_images(agent, fpv_imgs_blue, save_idx, step_types,
                                                                                                    cv_straight_path=agent.paths['his_cv_straight'], 
                                                                                                    cv_turn_path=agent.paths['his_cv_turn'])
                step_data.update({
                    "history_vis_multi": history_saved_paths_multi,
                    "history_vis_single": history_saved_paths_blue,
                    "history_vis_no_traj": history_saved_paths_no_draw,
                    "history_cv_vis_straight": straight_history_saved_paths,
                    "history_cv_vis_turn": turn_history_saved_paths,
                    "local_occupancy_explored_colored": local_bev_path or f"{agent.paths['local_occupancy_explored_colored']}/{save_idx:03d}.png",
                    "full_occupancy_explored_colored": full_bev_path or f"{agent.paths['full_occupancy_explored_colored']}/{save_idx:03d}.png",
                })
            else:
                step_data.update({
                    "history_vis_multi": history_saved_paths_multi,
                    "history_vis_single": history_saved_paths_blue,
                    "history_vis_no_traj": history_saved_paths_no_draw,
                    "local_occupancy_explored_colored": local_bev_path or f"{agent.paths['local_occupancy_explored_colored']}/{save_idx:03d}.png",
                    "full_occupancy_explored_colored": full_bev_path or f"{agent.paths['full_occupancy_explored_colored']}/{save_idx:03d}.png",
                })

        episode_data["steps"].append(step_data)
        
        save_idx += 1       # step index = 1 now

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

                # Reached this waypoint (or follower says STOP) for the last step
                if best_action is None or best_action == HabitatSimActions.stop:
                    if is_final_wp:
                        # Step STOP to terminate the episode within success radius
                        obs, rgbd, done, infos = agent.step(HabitatSimActions.stop)
                        last_infos = infos
                        episode_done = bool(done)
                        episode_steps += 1

                        planner_pose_inputs = build_surrogate_planner_pose_inputs(
                            obs.get('world_pos', agent.start_position),
                            obs.get('world_rotation', agent.start_rotation),
                        )
                        history.update(
                            obs['world_pos'],
                            obs['world_rotation'],
                            int(HabitatSimActions.stop),
                            rgbd[:, :, :3].astype(np.uint8),
                            rgbd[:, :, 3],
                            planner_pose_inputs,
                        )


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

                        vis_data = {
                            "instruction": current_episode.instruction.instruction_text,
                            "save_idx": save_idx,
                            "rgb_vis": agent.rgb_vis,
                            "depth_vis": agent.depth_vis,
                            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                            "save_paths": agent.paths,
                            "infos": infos,
                            "action_id": int(HabitatSimActions.stop),
                            "action_name": "STOP",
                        }

                        if WRITE_HISTORY_VISUALIZE:
                            results = history.get_hist_img(
                                episode_data=vis_data,
                                length=1000,
                            )
                            fpv_imgs_multi = results.get('fpv_imgs_multi_colored')
                            fpv_imgs_blue = results.get('fpv_imgs_blue')
                            fpv_imgs_no_draw = results.get('fpv_imgs_no_draw')
                            full_map_img = results.get('full_map_img')
                            local_map_img = results.get('local_map_img')
                            step_types = results.get('step_types')
                        

                            history_saved_paths_multi = save_history_images(agent, fpv_imgs_multi, save_idx, agent.paths['his_tra_multip_colored'])
                            history_saved_paths_blue = save_history_images(agent, fpv_imgs_blue, save_idx, agent.paths['his_tra_single_colored'])
                            history_saved_paths_no_draw = save_history_images(agent, fpv_imgs_no_draw, save_idx, agent.paths['his_tra_without_colored'])
                            local_bev_path, full_bev_path = save_bev_image(agent, local_map_img, full_map_img, save_idx)

                            if hist_alg == "cv":
                                straight_history_saved_paths, turn_history_saved_paths = save_history_type_images(agent, fpv_imgs_blue, save_idx, step_types,
                                                                                                        cv_straight_path=agent.paths['his_cv_straight'], 
                                                                                                        cv_turn_path=agent.paths['his_cv_turn'])
                                step_data.update({
                                    "history_vis_multi": history_saved_paths_multi,
                                    "history_vis_single": history_saved_paths_blue,
                                    "history_vis_no_traj": history_saved_paths_no_draw,
                                    "history_cv_vis_straight": straight_history_saved_paths,
                                    "history_cv_vis_turn": turn_history_saved_paths,
                                    "local_occupancy_explored_colored": local_bev_path or f"{agent.paths['local_occupancy_explored_colored']}/{save_idx:03d}.png",
                                    "full_occupancy_explored_colored": full_bev_path or f"{agent.paths['full_occupancy_explored_colored']}/{save_idx:03d}.png",
                                    })
                            else:
                                step_data.update({
                                    "history_vis_multi": history_saved_paths_multi,
                                    "history_vis_single": history_saved_paths_blue,
                                    "history_vis_no_traj": history_saved_paths_no_draw,
                                    "local_occupancy_explored_colored": local_bev_path or f"{agent.paths['local_occupancy_explored_colored']}/{save_idx:03d}.png",
                                    "full_occupancy_explored_colored": full_bev_path or f"{agent.paths['full_occupancy_explored_colored']}/{save_idx:03d}.png",
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
                    planner_pose_inputs = build_surrogate_planner_pose_inputs(
                        obs['world_pos'],
                        obs['world_rotation'],
                    )
                    # update history
                    history.update(
                        obs['world_pos'],
                        obs['world_rotation'],
                        action_id,
                        rgbd[:, :, :3].astype(np.uint8),
                        rgbd[:, :, 3],
                        planner_pose_inputs,
                    )

                    agent_state = envs.habitat_env.sim.get_agent_state()
                    step_data = {
                        "step_idx": save_idx,
                        "instruction": current_episode.instruction.instruction_text,
                        "agent_position": agent_state.position.tolist(),
                        "agent_rotation": [agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z],
                        "action_id": action_id,
                        "action_name": action_name,
                        "waypoint_id": wp_i,
                        "reference_waypoint": point.tolist() if hasattr(point, 'tolist') else list(point),
                        "instruction": current_episode.instruction.instruction_text
                    }
                    vis_data = {
                        "instruction": current_episode.instruction.instruction_text,
                        "save_idx": save_idx,
                        "rgb_vis": agent.rgb_vis,
                        "depth_vis": agent.depth_vis,
                        "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                        "save_paths": agent.paths,
                        "infos": infos,
                        "action_id": action_id,
                        "action_name": action_name,
                    }

                    if WRITE_HISTORY_VISUALIZE:
                        # Save FPV & BEV images via helper
                        results = history.get_hist_img(
                            episode_data=vis_data,
                            length=1000,
                        )
                        fpv_imgs_multi = results.get('fpv_imgs_multi_colored')
                        fpv_imgs_blue = results.get('fpv_imgs_blue')
                        fpv_imgs_no_draw = results.get('fpv_imgs_no_draw')
                        full_map_img = results.get('full_map_img')
                        local_map_img = results.get('local_map_img')
                        step_types = results.get('step_types')
                       

                        history_saved_paths_multi = save_history_images(agent, fpv_imgs_multi, save_idx, agent.paths['his_tra_multip_colored'])
                        history_saved_paths_blue = save_history_images(agent, fpv_imgs_blue, save_idx, agent.paths['his_tra_single_colored'])
                        history_saved_paths_no_draw = save_history_images(agent, fpv_imgs_no_draw, save_idx, agent.paths['his_tra_without_colored'])
                        local_bev_path, full_bev_path = save_bev_image(agent, local_map_img, full_map_img, save_idx)

                        if hist_alg == "cv":
                            straight_history_saved_paths, turn_history_saved_paths = save_history_type_images(agent, fpv_imgs_blue, save_idx, step_types,
                                                                                                    cv_straight_path=agent.paths['his_cv_straight'], 
                                                                                                    cv_turn_path=agent.paths['his_cv_turn'])

                            step_data.update({
                                "history_vis_multi": history_saved_paths_multi,
                                "history_vis_single": history_saved_paths_blue,
                                "history_vis_no_traj": history_saved_paths_no_draw,
                                "history_cv_vis_straight": straight_history_saved_paths,
                                "history_cv_vis_turn": turn_history_saved_paths,
                                "local_occupancy_explored_colored": local_bev_path or f"{agent.paths['local_occupancy_explored_colored']}/{save_idx:03d}.png",
                                "full_occupancy_explored_colored": full_bev_path or f"{agent.paths['full_occupancy_explored_colored']}/{save_idx:03d}.png",
                            })
                        else:
                            step_data.update({
                                "history_vis_multi": history_saved_paths_multi,
                                "history_vis_single": history_saved_paths_blue,
                                "history_vis_no_traj": history_saved_paths_no_draw,
                                "local_occupancy_explored_colored": local_bev_path or f"{agent.paths['local_occupancy_explored_colored']}/{save_idx:03d}.png",
                                "full_occupancy_explored_colored": full_bev_path or f"{agent.paths['full_occupancy_explored_colored']}/{save_idx:03d}.png",
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

        # write episode data (always)
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