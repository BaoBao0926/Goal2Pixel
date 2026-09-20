from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
import json

# 1. Determine the Goal2Pixel project root from this script location.
# python scripts/scripts_for_generating_data/adjust_pixel.py --mode r2r
current_script = Path(__file__).resolve()
project_root = current_script.parent.parent.parent 

MODE_CONFIG = {
	"r2r": (
		project_root / "training_data/data_mp3d_r2r/v1-3/train_pixel/annotations",
		project_root / "training_data/data_mp3d_r2r/v1-3/train_pixel/adjusted_annotations",
	),
	"rxr-en-US": (
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_pixel/annotations",
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_pixel/adjusted_annotations",
	),
 	"rxr-en-IN": (
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_pixel/annotations",
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_pixel/adjusted_annotations",
	),
}


def parse_args() -> str:
	parser = ArgumentParser(description="Adjust pixel annotations for different datasets.")
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

# Image sizes used to normalize pixel coordinates.
H_STANDARD = 256
H_PADDED = 256 + 16
W_STANDARD = 256
W_PADDED = 256 + 16
PAD_X = (W_PADDED - W_STANDARD) // 2
PAD_Y = (H_PADDED - H_STANDARD) // 2
DIST_THRESHOLD = 1.0  # meters
# 
WAYPOINT4_PIXEL_STOP = {"x": 136, "y": 272}
WAYPOINT4_PIXEL_LEFT = {"x": 0, "y": 136}
WAYPOINT4_PIXEL_RIGHT = {"x": 272, "y": 136}
PIXEL_LEFT_WO_PADDING = {"x": 0, "y": 128}
PIXEL_RIGHT_WO_PADDING = {"x": 256, "y": 128}
PIXEL_STOP_WO_PADDING = {"x": 128, "y": 256}


def _normalized_pixel(pixel: dict | None, width: int, height: int) -> dict | None:
	if not isinstance(pixel, dict):
		return None
	if "x" not in pixel or "y" not in pixel:
		return None
	return {
		"x": round(pixel["x"] / width, 4),
		"y": round(pixel["y"] / height, 4),
	}
 

def pixel_without_padding(pixel: dict | None) -> dict | None:
	if not isinstance(pixel, dict):
		return None
	if "x" not in pixel or "y" not in pixel:
		return None

	if pixel == WAYPOINT4_PIXEL_LEFT:
		return dict(PIXEL_LEFT_WO_PADDING)
	if pixel == WAYPOINT4_PIXEL_RIGHT:
		return dict(PIXEL_RIGHT_WO_PADDING)
	if pixel == WAYPOINT4_PIXEL_STOP:
		return dict(PIXEL_STOP_WO_PADDING)

	y = H_STANDARD if pixel["y"] == 272 else pixel["y"]
	
	return {
		"x": max(0, min(W_STANDARD, pixel["x"] - PAD_X)),
		"y": y,
	}


def _shift_actions(data: dict) -> None:
	steps = data.get("steps", [])
	if len(steps) < 2:
		return

	prev_action_ids = [step.get("action_id") for step in steps]
	prev_action_names = [step.get("action_name") for step in steps]

	for idx in range(1, len(steps)):
		steps[idx - 1]["action_id"] = prev_action_ids[idx]
		steps[idx - 1]["action_name"] = prev_action_names[idx]




def adjust_pixels(data: dict) -> dict:
	_shift_actions(data)
	ref_pos = None
	for step in reversed(data.get("steps", [])):
		rp = step.get("reference_waypoint")
		if isinstance(rp, list) and len(rp) >= 3:
			ref_pos = rp
			break

	waypoint4_ratio = _normalized_pixel(WAYPOINT4_PIXEL_STOP, W_PADDED, H_PADDED)

	for step in data.get("steps", []):
		if step.get("step_idx") == 0:
			continue
		wp1 = step.get("waypoint_pixel_1")
		wp2 = step.get("waypoint_pixel_2")
		wp3 = step.get("waypoint_pixel_3")
		agent_pos = step.get("agent_position")

		ratio1 = _normalized_pixel(wp1, W_STANDARD, H_PADDED)
		ratio2 = _normalized_pixel(wp2, W_STANDARD, H_PADDED)
		ratio3 = _normalized_pixel(wp3, W_PADDED, H_PADDED)

		if ratio1 is not None:
			step["pixel_ratio_1"] = ratio1
		if ratio2 is not None:
			step["pixel_ratio_2"] = ratio2
		if ratio3 is not None:
			step["waypoint_pixel_3_rotio"] = ratio3

		if (
			ref_pos is not None
			and isinstance(agent_pos, list)
			and len(agent_pos) >= 3
			and waypoint4_ratio is not None
		):
			dx = agent_pos[0] - ref_pos[0]
			dy = agent_pos[1] - ref_pos[1]
			dz = agent_pos[2] - ref_pos[2]
			dist = (dx * dx + dy * dy + dz * dz) ** 0.5
			if dist < DIST_THRESHOLD:
				step["waypoint_pixel_4"] = WAYPOINT4_PIXEL_STOP
				step["waypoint_pixel_4_rotio"] = waypoint4_ratio
				step["action_id"] = 0
				step["action_name"] = "STOP"
			else:
				if wp3 is not None and ratio3 is not None:
					step["waypoint_pixel_4"] = wp3
					step["waypoint_pixel_4_rotio"] = ratio3
				else:
					step["waypoint_pixel_4"] = WAYPOINT4_PIXEL_STOP
					step["waypoint_pixel_4_rotio"] = waypoint4_ratio

		if step.get("waypoint_pixel_4_rotio") is None:
			ratio4 = _normalized_pixel(step.get("waypoint_pixel_4"), W_PADDED, H_PADDED)
			if ratio4 is not None:
				step["waypoint_pixel_4_rotio"] = ratio4

		wp3_without_padding = pixel_without_padding(wp3)
		if wp3_without_padding is not None:
			step["waypoint_pixel_3_without_padding"] = wp3_without_padding
			ratio3_without_padding = _normalized_pixel(wp3_without_padding, W_STANDARD, H_STANDARD)
			if ratio3_without_padding is not None:
				step["waypoint_pixel_3_ratio_without_padding"] = ratio3_without_padding

		wp4_without_padding = pixel_without_padding(step.get("waypoint_pixel_4"))
		if wp4_without_padding is not None:
			step["waypoint_pixel_4_without_padding"] = wp4_without_padding
			ratio4_without_padding = _normalized_pixel(wp4_without_padding, W_STANDARD, H_STANDARD)
			if ratio4_without_padding is not None:
				step["waypoint_pixel_4_ratio_without_padding"] = ratio4_without_padding


	return data


def process_file(src_path: Path, input_root: Path, output_root: Path) -> None:
	with src_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	adjusted = adjust_pixels(data)

	relative_path = src_path.relative_to(input_root)
	out_path = output_root / relative_path
	out_path.parent.mkdir(parents=True, exist_ok=True)

	with out_path.open("w", encoding="utf-8") as f:
		json.dump(adjusted, f, indent=4)


def main() -> None:
	mode = parse_args()
	input_dir, output_dir = resolve_dirs(mode)
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Project Root: {project_root}")
	print(f"Selected mode: {mode}")
	print(f"Input Exists: {input_dir.exists()}")
	print(f"Output Path:  {output_dir}")

	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	for json_path in sorted(input_dir.rglob("*.json")):
		print(json_path)
		process_file(json_path, input_dir, output_dir)


if __name__ == "__main__":
	main()
