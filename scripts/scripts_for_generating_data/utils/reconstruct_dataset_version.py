import json
import gzip
import random

# v1: filter out specific scene from the dataset
# v2: Uniformly sample N episodes from the dataset
# v3: according to current rollout, find the episode that has not been rollout

# python scripts/scripts_for_generating_data/utils/reconstruct_dataset_version.py


def reconstruct_dataset_version_v1():
    dir = "../VLN_dataset/data/datasets/instruction_follow/mp3d/r2r/v1-3/train/train.json.gz"
    goal_scene = ["B6ByNegPMKs", "5LpN3gDmAk7", "759xd9YjKW5", "JmbYfDe2QKZ", "D7N2EKCX4Sj", 
                  "1LXtFkjw3qL", "jh4fc5c5qoQ", "VLzqgDo317F", "b8cTxDM8gDG", "PuKPg4mmafe", "ULsKaCPVFJR"]

    # Load original json.gz
    with gzip.open(dir, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    assert "episodes" in data, "Expected key 'episodes' in json"

    filtered_episodes = []
    count = 0

    for item in data["episodes"]:
        if isinstance(item, dict) and "scene_id" in item:
            scene_id_full = item["scene_id"]
            scene_id = scene_id_full.split("/")[1]  # mp3d/xxx/xxx.glb → xxx
            if scene_id in goal_scene:
                filtered_episodes.append(item)
                count += 1

    print(f"Total kept episodes: {count}")

    # 🔑 keep original structure, only replace episodes
    new_data = dict(data)
    new_data["episodes"] = filtered_episodes

    # Save new json.gz
    output_path = dir.replace(".json.gz", "_filtered.json.gz")
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f)

    print(f"Filtered data saved to {output_path}")


def reconstruct_dataset_version_v2():
    
    # Uniformly sample N episodes from the dataset
    N = 50
    SEED = 42

    SOURCE_PATH = "../VLN_dataset/data/datasets/instruction_follow/mp3d/r2r/v1-3/train/train_filtered.json.gz"
    # SOURCE_PATH = "../VLN_dataset/data/datasets/instruction_follow/mp3d/r2r/v1-3/val_unseen/val_unseen.json.gz"

    # Load original json.gz
    with gzip.open(SOURCE_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes")
    assert isinstance(episodes, list), "Expected key 'episodes' in json with list value"

    if N > len(episodes):
        raise ValueError(f"Requested {N} episodes but dataset only has {len(episodes)} episodes")

    rng = random.Random(SEED)
    selected_indices = rng.sample(range(len(episodes)), N)
    selected_indices.sort()  # keep original ordering for reproducibility when writing

    filtered_episodes = [episodes[i] for i in selected_indices]

    print(f"Total episodes in dataset: {len(episodes)}")
    print(f"Sampled episodes: {N}")

    # Keep original structure, only replace episodes
    new_data = dict(data)
    new_data["episodes"] = filtered_episodes

    # Save new json.gz
    output_path = SOURCE_PATH.replace(".json.gz", f"_sample_{N}.json.gz")
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f)

    print(f"Filtered data saved to {output_path}")

 
def reconstruct_dataset_version_v3():
    import json
    import gzip
    import os

    # python scripts/scripts_for_generating_data/utils/reconstruct_dataset_version_v3.py
    # according to current rollout, find the episode that has not been rollout


    SOURCE_PATH = "../VLN_dataset/data/datasets/instruction_follow/mp3d/r2r/v1-3/train/train.json.gz"
    ROLLOUT_PATH = "training_data/data_mp3d_r2r/v1-3/train_history/annotations"


    def collect_completed_pairs(root_dir: str):
        completed = set()
        for scene_name in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue
            for filename in os.listdir(scene_path):
                if not filename.endswith(".json"):
                    continue
                episode_id = filename[:-5]
                completed.add((scene_name, episode_id))
        return completed


    def extract_scene_name(scene_path: str) -> str:
        parts = scene_path.split("/")
        return parts[1] if len(parts) > 1 else parts[0]


    with gzip.open(SOURCE_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes")
    assert isinstance(episodes, list), "Expected key 'episodes' in json with list value"

    completed_pairs = collect_completed_pairs(ROLLOUT_PATH)

    pending_episodes = []
    for episode in episodes:
        scene_name = extract_scene_name(episode["scene_id"])
        episode_id = str(episode["episode_id"])
        if (scene_name, episode_id) not in completed_pairs:
            pending_episodes.append(episode)

    print(f"Total episodes in dataset: {len(episodes)}")
    print(f"Completed rollouts: {len(completed_pairs)}")
    print(f"Episodes pending rollout: {len(pending_episodes)}")

    new_data = dict(data)
    new_data["episodes"] = pending_episodes

    output_path = SOURCE_PATH.replace(".json.gz", "_pending.json.gz")
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f)

    print(f"Pending episodes saved to {output_path}")



# reconstruct_dataset_version_v1()
reconstruct_dataset_version_v2()
# reconstruct_dataset_version_v3()