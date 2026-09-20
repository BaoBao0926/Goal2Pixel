import habitat_sim
from habitat import logger

from typing import Sequence
import numpy as np
import math
from tqdm import tqdm  # Added tqdm for progress bar

REPLY_LENGTH_MIN = 1
REPLY_LENGTH_MAX = 500
ISLAND_RADIUS_LIMIT = 1.5
GEODESIC_DISTANCE_MIN = 1.0
GEODESIC_DISTANCE_MAX = 50.0

def filter_episodes_by_reply_action(env):
    episodes = env.habitat_env.episodes

    # Check if the first episode has a reference_replay attribute
    if not hasattr(episodes[0], 'reference_replay') or episodes[0].reference_replay is None:
        return env

    # Filter out episodes that have the 'STOP' action in between (not at the first index)
    filtered_episodes = [
        episode for episode in tqdm(episodes, total=len(episodes), desc="Filtering episodes with STOP in between")
        if 'STOP' not in [spec.action for spec in episode.reference_replay][1:-1]
    ]

    # Update environment episodes
    env.habitat_env.episodes = filtered_episodes

    logger.info(f"Number of episodes after replay action cleaning: {len(env.habitat_env.episodes)}")

    return env


def filter_episodes_by_reply_length(env):
    """
    First Check: 一个episode的长度必须在一个阈值范围内 1-500
    """
    episodes = env.habitat_env.episodes

    # Check if the first episode has a reference_replay attribute
    if not hasattr(episodes[0], 'reference_replay') or episodes[0].reference_replay is None:
        return env

    # Filter out episodes that do not meet the reply length requirements
    filtered_episodes = [
        episode for episode in tqdm(episodes, total=len(episodes), desc="Filtering by reply length")
        if REPLY_LENGTH_MIN <= len(episode.reference_replay) <= REPLY_LENGTH_MAX
    ]

    # Update environment episodes
    env.habitat_env.episodes = filtered_episodes

    logger.info(f"Number of episodes after replay length cleaning: {len(env.habitat_env.episodes)}")
    return env


def filter_episodes_by_diff_floor_obj(env):
    """
    Fourth Check: 两点之间的高度变化必须低于一个阈值 0.25
    """
    episodes = env.habitat_env.episodes

    # Filter episodes based on goals meeting height difference criteria
    filtered_episodes = []
    for episode in tqdm(episodes, total=len(episodes), desc="Filtering by height difference"):
        # Clean the goals to meet the height difference threshold
        episode.goals = [
            goal for goal in episode.goals
            if abs(episode.start_position[1] - goal.view_points[0].agent_state.position[1]) <= HEIGHT_DIFFERENCE_THRESHOLD
        ]

        # Only add episode if it has valid goals remaining
        if episode.goals:
            filtered_episodes.append(episode)

    # Replace original episodes with the filtered list
    env.habitat_env.episodes = filtered_episodes
    env.habitat_env.number_of_episodes = len(filtered_episodes)
    env.number_of_episodes = len(filtered_episodes)

    logger.info(f"Number of episodes after single floor cleaning: {len(env.habitat_env.episodes)}")

    return env



def filter_episodes_by_nav_area(env):
    """
    First Check: Ensure that the starting point is not on a small isolated navigable area.
    """
    episodes = env.habitat_env.episodes

    # Use tqdm to add a progress bar
    island_radius_episodes = [
        episode for episode in tqdm(episodes, desc="Filtering by navigable space", total=len(episodes))
        if env.habitat_env.sim.pathfinder.island_radius(episode.start_position) >= ISLAND_RADIUS_LIMIT
    ]

    # Use tqdm to add a progress bar
    navigable_episodes = [
        episode for episode in tqdm(island_radius_episodes, desc="Filtering by navigable", total=len(island_radius_episodes))
        if env.habitat_env.sim.pathfinder.is_navigable(episode.start_position)
    ]

    # Replace original episodes with the filtered list
    env.habitat_env.episodes = navigable_episodes
    env.habitat_env.number_of_episodes = len(navigable_episodes)
    env.number_of_episodes = len(navigable_episodes)

    logger.info(f"Number of episodes after 21 objects, navigable radius and navigable cleaning: {len(env.habitat_env.episodes)}")

    return env



def filter_episodes_by_geodesic_distance(env):
    """
    Second Check: 起始点到目标的最短路径必须有效且在一定范围内 1-30
    """

    episodes = env.habitat_env.episodes  # Use a consistent reference
    sim = env.habitat_env.sim
    path = habitat_sim.ShortestPath()

    filtered_episodes = []

    # Iterate through episodes and retain only those that meet geodesic distance requirements
    for episode in tqdm(episodes, total=len(episodes), desc="Filtering by geodesic distance"):
        start_position = episode.start_position
        goals = episode.goals

        # Extract the view points' positions
        vps = [vp.agent_state.position for g in goals for vp in g.view_points]

        if not vps:
            raise ValueError("No view points found for episode")

        # Calculate the geodesic distance to the closest goal
        geo_dist, closest_point = geodesic_distance(sim, start_position, vps)

        # If geodesic distance is invalid or out of range, skip this episode
        if geo_dist is None or not np.isfinite(geo_dist) or geo_dist < GEODESIC_DISTANCE_MIN or geo_dist > GEODESIC_DISTANCE_MAX:
            print(f"geo_dist: {geo_dist}")
            continue

        # Ensure there's a valid navigable path for the closest goal
        path.requested_start = start_position
        path.requested_end = closest_point
        found_path = sim.pathfinder.find_path(path)

        if not found_path:
            print(f"Path not found for episode: {episode.episode_id}")
            continue

        # If all checks pass, add this episode to the filtered list
        filtered_episodes.append(episode)

    # Replace the original list of episodes with the filtered episodes
    env.habitat_env.episodes = filtered_episodes

    logger.info(f"Number of episodes after geodesic distance and path finding cleaning: {len(env.habitat_env.episodes)}")

    return env

def filter_no_goal(env):
    """
    Filter out episodes that have no goal.
    """
    episodes = env.habitat_env.episodes

    # Create a filtered list of episodes that have at least one goal
    filtered_episodes = [
        episode for episode in tqdm(episodes, total=len(episodes), desc="Filtering episodes with no goals")
        if len(episode.goals) > 0
    ]

    # Update the environment's episodes with the filtered list
    env.habitat_env.episodes = filtered_episodes

    logger.info(f"Number of episodes after no goal cleaning: {len(env.habitat_env.episodes)}")
    return env

def filter_objectgoal(env):
    """
    Filter out episodes that have object category in ["foodstuff", "stationery","fruit", "plaything", "hand_tool", "game_equipment", "kitchenware"]
    """
    episodes = env.habitat_env.episodes

    filtered_episodes = [
        episode for episode in tqdm(episodes, total=len(episodes), desc="Filtering episodes with object goal")
        if episode.object_category not in ["foodstuff", "stationery","fruit", "plaything", "hand_tool", "game_equipment", "kitchenware"]
    ]

    env.habitat_env.episodes = filtered_episodes
    logger.info(f"Number of episodes after object goal cleaning: {len(env.habitat_env.episodes)}")
    return env

def filter_fail_HD(env):
    """
    Filter out episodes that have no success in HD.
    """
    # episodes = env.habitat_env.episodes
    num_episodes = len(env.habitat_env.episodes)

    fail_episodes = {}
    fail_no = 0

    filtered_episodes = []
    possible_actions = ["stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down"]
    total_success = 0
    total_spl = 0
    total_timesteps = 0
    total_reward = 0

    # for episode in tqdm(episodes, total=len(episodes), desc="Filtering failed episodes"):
    for i in range(num_episodes):
        env.reset()
        step_index = 1

        current_episode = env.habitat_env.current_episode
        scene_id = current_episode.scene_id.split("/")[-1][:-4]
        ep_id = current_episode.episode_id
        object_goal = current_episode.object_category

        reference_replay = current_episode.reference_replay

        for data in reference_replay[step_index:]:
            converted_action = data.action.lower()
            action = possible_actions.index(converted_action)

            # Check the success of the episode at each step (simulate the replay)
            observations, reward, done, info = env.step(action)

            step_index += 1

            if math.isnan(reward) or math.isnan(info["spl"]):
                logger.info("Nan reward: ", scene_id, ep_id, object_goal, step_index)
                break

            if converted_action == "stop" or done:
                break

        if info["success"] < 1 or math.isnan(reward) or math.isnan(info["spl"]):
            logger.info(f"Failed episode: {i}, scene: {scene_id}, object goal: {object_goal}, episode id: {ep_id}")
            logger.info(f"Success: {info['success']}, SPL: {info['spl']}")
            fail_episodes[fail_no] = {"scene_id": scene_id,
                                      "object_goal": object_goal,
                                      "episode_id": ep_id,
                                      "success": info["success"], "spl": info["spl"]
                                      }
            fail_no += 1

        # Add the episode to filtered list if it is successful
        else:
            total_success += info["success"]
            total_spl += info["spl"]
            total_timesteps += info["elasp_step"]

            filtered_episodes.append(current_episode)

    # Update the environment's episodes with the filtered list
    env.habitat_env.episodes = filtered_episodes

    logger.info(f"Number of failed episodes: {fail_no}")
    logger.info(f"Number of episodes after filtering failed ones: {len(env.habitat_env.episodes)}")
    logger.info(f"Average success rate: {total_success / (num_episodes - fail_no)}")
    logger.info(f"Average SPL: {total_spl / (num_episodes - fail_no)}")
    logger.info(f"Average timesteps: {total_timesteps / (num_episodes - fail_no)}")


def filter_hm3d(env):
    """
    Filter out episodes that is finding plants. "plant": 2
    """

def geodesic_distance(sim, position_a, position_b):
    """
    calculate the geodesic distance between a start position and one or multiple possible end positions
    """
    path = habitat_sim.MultiGoalShortestPath()
    if isinstance(position_b[0], (Sequence, np.ndarray)):
        path.requested_ends = np.array(position_b, dtype=np.float32)
    else:
        path.requested_ends = np.array([np.array(position_b, dtype=np.float32)])
    path.requested_start = np.array(position_a, dtype=np.float32)
    sim.pathfinder.find_path(path)
    end_pt = path.points[-1] if len(path.points) else np.array([])
    return path.geodesic_distance, end_pt


def filter_seen16_object_goal(env):
    """
    Filter out episodes that have object category in ["counter", "bed", "toilet", "chest_of_drawers", "plant"]
    These 5 should be used for unseen5
    """
    episodes = env.habitat_env.episodes

    filtered_episodes = [
        episode for episode in tqdm(episodes, total=len(episodes), desc="Filtering episodes with object goal")
        if episode.object_category not in ["counter", "bed", "toilet", "chest_of_drawers", "plant"]
    ]

    env.habitat_env.episodes = filtered_episodes
    logger.info(f"Number of episodes remain seen 16 types object goal : {len(env.habitat_env.episodes)}")
    return env

def filter_unseen5_object_goal(env):
    """
    Remain episodes that have object category in ["counter", "bed", "toilet", "chest_of_drawers", "plant"]
    For zero shot evaluation
    """
    episodes = env.habitat_env.episodes

    filtered_episodes = [
        episode for episode in tqdm(episodes, total=len(episodes), desc="Filtering episodes with object goal")
        if episode.object_category in ["counter", "bed", "toilet", "chest_of_drawers", "plant"]
    ]

    env.habitat_env.episodes = filtered_episodes
    logger.info(f"Number of episodes remain 5 types object goal: {len(env.habitat_env.episodes)}")
    return env