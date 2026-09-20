import logging

from habitat import get_config

from task_patch.utils.get_config import register_plugins
from task_patch.utils.env_wrapper import FilteredEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Set log message format
    datefmt='%Y-%m-%d %H:%M:%S',  # Set the date format
)

def collect_split_stat(config_path, dataset):
    # Register custom hydra plugin
    register_plugins()
    config = get_config(config_path=config_path)

    # with SimpleRLEnv(config=config) as env:
    #     logging.info('*' * 50)
    #     logging.info('Before cleaning')
    #     logging.info(f"dataset name: {dataset} | split: {config.habitat.dataset.split}")
    #
    #     scene_ids = []
    #     object_ids = []
    #     for ep in env.episodes:
    #         scene_ids.append(ep.scene_id.split('/')[-1][:-4])
    #         object_ids.append(ep.object_category)
    #     unique_scene_ids = set(scene_ids)
    #     unique_object_ids = set(object_ids)
    #     logging.info(f"Number of unique scene IDs: {len(unique_scene_ids)}")
    #     logging.info(f"Number of Episodes: {len(env.episodes)}")
    #     logging.info(f"Number of unique object IDs: {len(unique_object_ids)}")
    #
    #     env.close()

    env = None
    # release the memory

    with FilteredEnv(config=config, dataset=None) as env:
        logging.info('*' * 50)
        logging.info('After cleaning')
        logging.info(f"dataset name: {dataset} | split: {config.habitat.dataset.split}")

        scene_ids = []
        object_ids = []
        for ep in env.episodes:
            scene_ids.append(ep.scene_id.split('/')[-1][:-4])
            object_ids.append(ep.object_category)
        unique_scene_ids = set(scene_ids)
        unique_object_ids = set(object_ids)
        logging.info(f"Number of unique scene IDs: {len(unique_scene_ids)}")
        logging.info(f"Number of Episodes: {len(env.episodes)}")
        logging.info(f"Number of unique object IDs: {len(unique_object_ids)}")

        env.close()


if __name__ == "__main__":
    dataset = "mp3d"

    # config_path = "configs/statistic/mp3d_val_stat.yaml"
    """
    2024-12-11 13:42:01,480 Initializing task ObjectNav-v1
    2024-12-11 13:42:01 - INFO - **************************************************
    2024-12-11 13:42:01 - INFO - Before cleaning
    2024-12-11 13:42:01 - INFO - dataset name: mp3d | split: val
    2024-12-11 13:42:01 - INFO - Number of unique scene IDs: 11
    2024-12-11 13:42:01 - INFO - Number of Episodes: 2195
    2024-12-11 13:42:01 - INFO - Number of unique object IDs: 21
    2024-12-11 13:42:09 - INFO - Number of episodes before cleaning: 2195
    2024-12-11 16:10:38 - INFO - Number of episodes after single floor cleaning: 1688
    2024-12-11 16:10:38 - INFO - Number of episodes after navigable space cleaning: 1148
    2024-12-11 13:42:12 - INFO - **************************************************
    2024-12-11 13:42:12 - INFO - After cleaning
    2024-12-11 13:42:12 - INFO - dataset name: mp3d | split: val
    2024-12-11 13:42:12 - INFO - Number of unique scene IDs: 11
    2024-12-11 13:42:12 - INFO - Number of Episodes: 1148
    2024-12-11 13:42:12 - INFO - Number of unique object IDs: 21

    """
    # config_path = "configs/statistic/mp3d_HD_70k_stat.yaml"

    """
    2024-12-11 13:53:08 - INFO - **************************************************
    2024-12-11 13:53:08 - INFO - Before cleaning
    2024-12-11 13:53:08 - INFO - dataset name: mp3d | split: train 70k
    2024-12-11 13:53:08 - INFO - Number of unique scene IDs: 56
    2024-12-11 13:53:08 - INFO - Number of Episodes: 70176
    2024-12-11 13:53:08 - INFO - Number of unique object IDs: 28
    2024-12-11 14:00:48 - INFO - Number of episodes before cleaning: 70176
    2024-12-11 16:04:26 - INFO - Number of episodes after single floor cleaning: 52387
    2024-12-11 16:04:27 - INFO - Number of episodes after navigable space cleaning: 25688
    2024-12-11 16:04:27 - INFO - Number of episodes after replay length cleaning: 23890
    2024-12-11 16:32:29 - INFO - Number of episodes after object goal cleaning: 22210
    2024-12-11 16:04:27 - INFO - **************************************************
    2024-12-11 16:04:27 - INFO - After cleaning
    2024-12-11 16:04:27 - INFO - dataset name: mp3d | split: train
    2024-12-11 16:04:28 - INFO - Number of unique scene IDs: 56
    2024-12-11 16:04:28 - INFO - Number of Episodes: 22210
    2024-12-11 16:04:28 - INFO - Number of unique object IDs: 21
    """

    config_path = "configs/statistic/mp3d_HD_35k_stat.yaml"

    """
    2024-12-11 14:08:20 - INFO - **************************************************
    2024-12-11 14:08:20 - INFO - Before cleaning
    2024-12-11 14:08:20 - INFO - dataset name: mp3d | split: train 35k
    2024-12-11 14:08:20 - INFO - Number of unique scene IDs: 28
    2024-12-11 14:08:20 - INFO - Number of Episodes: 34641
    2024-12-11 14:08:20 - INFO - Number of unique object IDs: 28
    2024-12-11 14:23:38 - INFO - Number of episodes before cleaning: 34641
    2024-12-11 14:23:39 - INFO - Number of episodes after single floor cleaning: 26262
    2024-12-11 14:23:39 - INFO - Number of episodes after navigable space cleaning: 21092
    2024-12-11 14:23:38 - INFO - Number of episodes after replay length cleaning: 19388
    2024-12-11 16:23:37 - INFO - Number of episodes after object goal cleaning: 17927
    2024-12-11 14:24:16 - INFO - **************************************************
    2024-12-11 14:24:16 - INFO - After cleaning
    2024-12-11 14:24:16 - INFO - dataset name: mp3d | split: train 35k
    2024-12-11 14:24:16 - INFO - Number of unique scene IDs: 28
    2024-12-11 14:24:16 - INFO - Number of Episodes: 17927
    2024-12-11 14:24:16 - INFO - Number of unique object IDs: 21

    """

    # config_path = "configs/statistic/mp3d_HD_50k_stat.yaml"
    """
    2024-12-11 14:15:43 - INFO - Before cleaning
    2024-12-11 14:15:43 - INFO - dataset name: mp3d | split: train 50k
    2024-12-11 14:15:43 - INFO - Number of unique scene IDs: 40
    2024-12-11 14:15:43 - INFO - Number of Episodes: 49778
    2024-12-11 14:15:43 - INFO - Number of unique object IDs: 28
    2024-12-11 14:43:35 - INFO - Number of episodes before cleaning: 49778
    2024-12-11 14:43:36 - INFO - Number of episodes after single floor cleaning: 37037
    2024-12-11 14:43:36 - INFO - Number of episodes after navigable space cleaning: 23219
    2024-12-11 14:43:36 - INFO - Number of episodes after replay length cleaning: 21239
    2024-12-11 16:27:18 - INFO - Number of episodes after object goal cleaning: 19659
    2024-12-11 14:18:35 - INFO - **************************************************
    2024-12-11 14:44:14 - INFO - After cleaning
    2024-12-11 14:44:14 - INFO - dataset name: mp3d | split: train 50k
    2024-12-11 14:44:14 - INFO - Number of unique scene IDs: 40
    2024-12-11 14:44:14 - INFO - Number of Episodes: 19659
    2024-12-11 14:44:14 - INFO - Number of unique object IDs: 21
    """



    collect_split_stat(config_path, dataset)