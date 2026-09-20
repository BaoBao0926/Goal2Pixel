def get_r2r_episode_info_display(episode_data):
    """
    Extract episode info for R2R VLN visualization.
    Includes: instruction, action, navigation metrics (SR, SPL, OSR, etc.)
    """
    save_idx = episode_data.get('save_idx', 0)
    return {
        'save_idx': save_idx,
        'instruction': episode_data.get('instruction', 'N/A'),
        'action': episode_data.get('action_name', 'N/A'),
        'SR': episode_data.get('infos', {}).get('success', 'N/A'),
        'SPL': episode_data.get('infos', {}).get('spl', 'N/A'),
        'distance_to_goal': episode_data.get('infos', {}).get('distance_to_goal', 'N/A'),
        'OSR': episode_data.get('infos', {}).get('oracle_success', 'N/A'),
        'OSPL': episode_data.get('infos', {}).get('oracle_spl', 'N/A'),
        'NE': episode_data.get('infos', {}).get('oracle_navigation_error', 'N/A')
    }


def get_hdt_episode_info_display(episode_data):
    """
    Extract episode info for HDT object navigation visualization.
    Includes: object_category, action, navigation metrics (success, SPL, distance_to_goal)
    """
    save_idx = episode_data.get('save_idx', 0)
    return {
        'save_idx': save_idx,
        'object_category': episode_data.get('object_category', 'N/A'),
        'action': episode_data.get('action_name', 'N/A'),
        'success': episode_data.get('infos', {}).get('success', 'N/A'),
        'SPL': episode_data.get('infos', {}).get('spl', 'N/A'),
        'distance_to_goal': episode_data.get('infos', {}).get('distance_to_goal', 'N/A'),
    }