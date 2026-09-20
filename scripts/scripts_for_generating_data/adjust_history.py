from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
import json

# bascially, this python file is useless

# 1. Anchor to the script, then go up 3 levels to the Goal2Pixel project root.
# python scripts/scripts_for_generating_data/adjust_history.py -mode r2r
current_script = Path(__file__).resolve()
project_root = current_script.parent.parent.parent 

MODE_CONFIG = {
	"r2r": (
		project_root / "training_data/data_mp3d_r2r/v1-3/train_history/annotations",
		project_root / "training_data/data_mp3d_r2r/v1-3/train_history/adjusted_annotations",
	),
	"rxr-en-US": (
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_history/annotations",
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_history/adjusted_annotations",
	),
 	"rxr-en-IN": (
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_history/annotations",
		project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_history/adjusted_annotations",
	),
}


def parse_args() -> str:
	parser = ArgumentParser(description="Adjust history annotation assets for different datasets.")
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

HISTORY_IMAGE_KEYS = (
	"history_vis_multi",
	"history_vis_single",
	"history_vis_no_traj",
)


def _convert_png_to_jpg(path_str: str) -> str:
	"""Swap a .png suffix with .jpg while preserving the path."""
	return str(Path(path_str).with_suffix(".jpg")) if path_str.lower().endswith(".png") else path_str


def _replace_filename_with_001(png_path: str) -> str:
	"""Change trailing 000.png to 001.png when present."""
	return png_path[:-7] + "001.png" if png_path.endswith("000.png") else png_path


def adjust_local_history_paths(payload):
	"""Recursively adjust local history occupancy PNG references."""
	if isinstance(payload, dict):
		for key, value in payload.items():
			if key == "local_occupancy_explored_colored" and isinstance(value, str):
				payload[key] = _replace_filename_with_001(value)
			else:
				adjust_local_history_paths(value)
	elif isinstance(payload, list):
		for item in payload:
			adjust_local_history_paths(item)


def _convert_history_visualizations(payload):
	"""Recursively convert history visualization PNGs to JPGs."""
	if isinstance(payload, dict):
		for key, value in payload.items():
			if key in HISTORY_IMAGE_KEYS and isinstance(value, list):
				payload[key] = [
					_convert_png_to_jpg(item) if isinstance(item, str) else item
					for item in value
				]
			else:
				_convert_history_visualizations(value)
	elif isinstance(payload, list):
		for item in payload:
			_convert_history_visualizations(item)


def process_file(src_path: Path, input_root: Path, output_root: Path) -> None:
	with src_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	adjust_local_history_paths(data)
	_convert_history_visualizations(data)

	relative_path = src_path.relative_to(input_root)
	out_path = output_root / relative_path
	out_path.parent.mkdir(parents=True, exist_ok=True)

	with out_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=4)


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

	for json_path in sorted(input_dir.rglob("*.json")):
		print(f"Processing JSON: {json_path}")
		process_file(json_path, input_dir, output_dir)


if __name__ == "__main__":
	main()