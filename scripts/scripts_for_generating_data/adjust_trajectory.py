

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple
import json

try:
	from tqdm import tqdm
except ImportError:  # pragma: no cover
	tqdm = None

# 1. Anchor to the script, then go up 3 levels to the Goal2Pixel project root.
# python scripts/scripts_for_generating_data/adjust_trajectory.py --mode r2r
current_script = Path(__file__).resolve()
project_root = current_script.parent.parent.parent 

MODE_CONFIG = {
	"r2r": (
		project_root / "training_data/data_mp3d_r2r/v1-3/train_trajectory/annotations",
		project_root / "training_data/data_mp3d_r2r/v1-3/train_trajectory/adjusted_annotations",
	),
	"rxr-en-US": (
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_trajectory/annotations",
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_trajectory/adjusted_annotations",
	),	
 	"rxr-en-IN": (
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_trajectory/annotations",
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_trajectory/adjusted_annotations",
	),
}


def parse_args() -> str:
	parser = ArgumentParser(description="Adjust trajectory annotations for different datasets.")
	parser.add_argument(
		"--mode",
		choices=sorted(MODE_CONFIG.keys()),
		default="r2r",
		help="Select which dataset configuration to process.",
	)
	return parser.parse_args().mode


def resolve_dirs(mode: str) -> Tuple[Path, Path]:
	try:
		return MODE_CONFIG[mode]
	except KeyError as exc:
		raise ValueError(f"Unknown mode '{mode}'. Expected one of: {', '.join(MODE_CONFIG)}") from exc


def adjust_actions(data: dict) -> dict:
	steps = data.get("steps", [])
	if len(steps) < 2:
		return data

	prev_action_ids = [step.get("action_id") for step in steps]
	prev_action_names = [step.get("action_name") for step in steps]

	for idx in range(1, len(steps)):
		steps[idx - 1]["action_id"] = prev_action_ids[idx]
		steps[idx - 1]["action_name"] = prev_action_names[idx]

	return data


def remove_initial_step(data: dict) -> dict:
	steps = data.get("steps", [])
	filtered_steps = [step for step in steps if step.get("step_idx") != 0]
	if len(filtered_steps) != len(steps):
		data["steps"] = filtered_steps
	return data


HISTORY_WINDOWS: List[int] = [5, -1]

IMAGE_PATH_KEYS = (
	"rgb_path",
	"rgb_padded_path",
	"rgb_padded_down_path",
)


def _normalize_action_name(action: Optional[str]) -> Optional[str]:
	if not action:
		return action
	# here only should be last step is STOP
	if action.upper() == "STOP":
		return "move_forward"
	return action


def _history_key(window: int) -> str:
	return f"action_history_{window}"


def _format_recent_history(history: List[str], window: int) -> str:
	if not history:
		return "none"
	if window == -1:
		return ", ".join(history)
	if window <= 0:
		return "none"
	return ", ".join(history[-window:])


def annotate_history(data: dict) -> dict:
	steps = data.get("steps", [])
	history: List[str] = []
	windows = HISTORY_WINDOWS or []
	for step in steps:
		for window in windows:
			step[_history_key(window)] = _format_recent_history(history, window)
		action_name = _normalize_action_name(step.get("action_name"))
		if action_name:
			history.append(action_name)
	return data


def convert_image_paths_to_jpg(data: dict) -> dict:
	"""Convert known RGB-related paths to use .jpg files."""
	steps = data.get("steps", [])
	for step in steps:
		for key in IMAGE_PATH_KEYS:
			value = step.get(key)
			if isinstance(value, str) and value.lower().endswith(".png"):
				step[key] = str(Path(value).with_suffix(".jpg"))
	return data


def process_file(src_path: Path, input_root: Path, output_root: Path) -> None:
	with src_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	trimmed = remove_initial_step(data)
	adjusted = adjust_actions(trimmed)
	annotated = annotate_history(adjusted)
	converted = convert_image_paths_to_jpg(annotated)

	relative_path = src_path.relative_to(input_root)
	out_path = output_root / relative_path
	out_path.parent.mkdir(parents=True, exist_ok=True)

	with out_path.open("w", encoding="utf-8") as f:
		json.dump(converted, f, indent=4)


def main() -> None:
	mode = parse_args()
	input_dir, output_dir = resolve_dirs(mode)
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Project Root detected: {project_root}")
	print(f"Selected mode: {mode}")
	print(f"Input Exists: {input_dir.exists()}")
	print(f"Outputting to: {output_dir}")

	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	json_paths = sorted(input_dir.rglob("*.json"))
	iterator = json_paths
	if tqdm is not None:
		iterator = tqdm(json_paths, desc=f"Processing {mode}", unit="file")

	for json_path in iterator:
		process_file(json_path, input_dir, output_dir)


if __name__ == "__main__":
	main()