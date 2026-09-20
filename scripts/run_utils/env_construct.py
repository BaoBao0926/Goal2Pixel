import numpy as np
from omegaconf import DictConfig
import torch

from habitat.datasets import make_dataset
from habitat.config import read_write
from habitat import logger

from task_patch.env_register.env import AnnotatedGymHabitatEnv

def construct_envs(config: "DictConfig"):
    if config.environment == 'habitat':
        # Load dataset
        if config.habitat.dataset.type in ['R2RVLN-v2', 'RxRVLN-v2']:
            dataset = make_dataset(id_dataset=config.habitat.dataset.type,
                                config=config.habitat.dataset,
                                )
            pass
        elif config.habitat.dataset.type == 'ObjectNav-MP3D':
            dataset = make_dataset(id_dataset=config.habitat.dataset.type,
                                config=config.habitat.dataset,
                                exp_mode=config.experiment_mode,
                                )
        else:
            raise NotImplementedError(f"Dataset type {config.habitat.dataset.type} not supported. Go check the configuration.")

        # get the scenes
        scenes = config.habitat.dataset.content_scenes
        if "*" in config.habitat.dataset.content_scenes:
            scenes = dataset.get_scenes_to_load(config.habitat.dataset)

        if len(scenes) > 0:
            contentss = scenes

            # TODO: option to filter out the bad scenes
            bad_scense = []
            good_scense = [contentss[i] for i in range(len(contentss)) if contentss[i] not in bad_scense]

            with read_write(config):
                config.habitat.dataset.content_scenes = good_scense

        logger.info("Number of scenes to be replayed: %d", len(scenes))


        with read_write(config):
            config.habitat.simulator.scene = dataset.episodes[0].scene_id

            config.habitat.environment.iterator_options.shuffle = True

            config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = config.mapping.env_frame_width
            config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = config.mapping.env_frame_height
            config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = config.mapping.hfov
            config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0, config.habitat.simulator.agents.main_agent.height, 0]

            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = config.mapping.env_frame_width
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = config.mapping.env_frame_height
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = config.mapping.hfov
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = config.mapping.min_depth
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = config.mapping.max_depth
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0, config.habitat.simulator.agents.main_agent.height, 0]

            config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = config.mapping.env_frame_width
            config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = config.mapping.env_frame_height
            config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = config.mapping.hfov
            config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.position = [0, config.habitat.simulator.agents.main_agent.height, 0]

            config.habitat.simulator.agents.main_agent.height = config.habitat.simulator.agents.main_agent.height

            config.habitat.simulator.scene = dataset.episodes[0].scene_id

        logger.info(
            f"Split: {config.habitat.dataset.split}, "
            f"Dataset: {config.habitat.dataset.data_path.split('/')[-3]}, "
            f"which has {dataset.num_episodes} episodes.")

        env = AnnotatedGymHabitatEnv(config=config.habitat,
                                     dataset=dataset,
                                     device=config.device)

        return env

    elif config.environment == 'real_world':
        raise NotImplementedError



def modify_config(config: "DictConfig"):
    with read_write(config):
        # Calculate map grid size (number of cells)
        config.mapping.map_size_cells = int(config.mapping.map_size / config.mapping.map_resolution + 1e-5) 
        config.mapping.global_width, config.mapping.global_height = config.mapping.map_size_cells, config.mapping.map_size_cells
        config.mapping.local_width = int(config.mapping.global_width / config.mapping.global_downscaling)
        config.mapping.local_height = int(config.mapping.global_height / config.mapping.global_downscaling)
        config.mapping.device = "cuda:0" if torch.cuda.is_available() and bool(config.get("cuda", False)) else "cpu"

        # set agent params
        config.mapping.camera_height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position[1]
        config.mapping.hfov = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov
        config.mapping.agent_radius = config.habitat.simulator.agents.main_agent.radius

        # set sensor params
        config.mapping.min_depth = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth
        config.mapping.max_depth = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth

        config.mapping.visualization.fov_hfov = config.mapping.hfov
        config.mapping.visualization.fov_max_depth_m = config.mapping.max_depth

        config.device = config.mapping.device

    return config
