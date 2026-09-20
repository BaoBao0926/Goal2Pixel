import json
from pathlib import Path

# python ./scripts/scripts_for_generating_data/utils/count_rollout_time.py
current_script = Path(__file__).resolve()
project_root = current_script.parent.parent.parent.parent

# path = project_root / "training_data/data_mp3d_r2r/v1-3/train_trajectory/finished_episodes.json"
path = project_root / "training_data/data_mp3d_r2r/v1-3/train_pixel/finished_episodes.json"
# path = project_root / "data_mp3d_r2r_v1/v1-3/train_history/finished_episodes.json"

# path = project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_trajectory/finished_episodes.json"
# path = project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_pixel/finished_episodes.json"
# path = project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_history/finished_episodes.json"

path = project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_trajectory/finished_episodes.json"
# path = project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_pixel/finished_episodes.json"
path = project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_history/finished_episodes.json"

scene = ["ac26ZMwG7aT"]

total_episodes = 0
total_time_all = 0.0
scene_stats = {s: {"count": 0, "time": 0.0} for s in scene}

if path.exists():
    with path.open("r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    total_episodes += 1
                    time_val = data.get("time", 0.0)
                    total_time_all += time_val

                    s_id = data.get("scene_id")
                    if s_id in scene_stats:
                        scene_stats[s_id]["count"] += 1
                        scene_stats[s_id]["time"] += time_val

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

    print(f"Total episodes recorded: {total_episodes}")
    if total_episodes > 0:
        print(f"Total Time: {total_time_all / 3600}h")
        print(f"Average time (Total): {total_time_all / total_episodes:.2f} seconds")

    for s in scene:
        stats = scene_stats[s]
        count = stats["count"]
        time_sum = stats["time"]
        print(f"Scene {s}: Total time = {time_sum:.2f} seconds, Count = {count}")
        if count > 0:
            print(f"Scene {s}: Average time = {time_sum / count:.2f} seconds")
else:
    print(f"File not found: {path}")
