import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from tqdm import tqdm


current_script = Path(__file__).resolve()
project_root = current_script.parent.parent.parent

PATH = {
	"r2r": project_root / "training_data/data_mp3d_r2r/v1-3/train_history/his_tra_single_colored",
	"rxr-en-US": project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_history/his_tra_single_colored",
	"rxr-en-IN": project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_history/his_tra_single_colored",
}

OUTPUT_PATH = {
	"r2r": project_root / "training_data/data_mp3d_r2r/v1-3/train_history/annotation_mask",
	"rxr-en-US": project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_history/annotation_mask",
	"rxr-en-IN": project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_history/annotation_mask",
}

VIS_PATH = {
	"r2r": project_root / "training_data/data_mp3d_r2r/v1-3/train_history/mask_debug",
	"rxr-en-US": project_root / "training_data/data_mp3d_rxr/rxr_guide_en-US/train_history/mask_debug",
	"rxr-en-IN": project_root / "training_data/data_mp3d_rxr/rxr_guide_en-IN/train_history/mask_debug",
}

DEFAULT_GRID_SIZE = 16
DEFAULT_TOKEN_PIXELS = 16
DEFAULT_BLUE_TOL = 25


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate 16x16 trajectory token masks from history images. "
			"Input folder format: scene_id/episode_id/step_id/image_file."
		)
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="r2r",
		choices=PATH.keys(),
		help="Dataset mode to process.",
	)
	parser.add_argument(
		"--image-root",
		type=str,
		default=None,
		help="Override input image root. If unset, use the path for selected --mode.",
	)
	parser.add_argument(
		"--output-root",
		type=str,
		default=None,
		help=(
			"Override output root. If unset, use the output path for selected --mode. "
			"JSON files are written as output_root/scene_id/episode_id.json "
			"to mirror train_history/annotations organization."
		),
	)
	parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
	parser.add_argument(
		"--token-pixels",
		type=int,
		default=DEFAULT_TOKEN_PIXELS,
		help="Pixel size per token. 16x16=256 by default.",
	)
	parser.add_argument(
		"--min-blue-pixels",
		type=int,
		default=1,
		help="Minimum number of blue pixels in a token to mark it as containing trajectory.",
	)
	parser.add_argument(
		"--blue-tol",
		type=int,
		default=DEFAULT_BLUE_TOL,
		help=(
			"Tolerance for pure-blue matching in saved BGR images. "
			"Detects pixels near BGR=(255,0,0)."
		),
	)
	parser.add_argument(
		"--include-empty-steps",
		action="store_true",
		help="Include steps with no image files.",
	)
	parser.add_argument("--scene-id", type=str, default=None, help="Only process one scene id.")
	parser.add_argument("--episode-id", type=str, default=None, help="Only process one episode id.")
	parser.add_argument(
		"--vis-image",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Save debug images with token-grid and detected trajectory overlay (default: enabled).",
	)
	parser.add_argument(
		"--vis-root",
		type=str,
		default=None,
		help=(
			"Optional output root for debug visualization images. "
			"If unset, use the debug path for selected --mode."
		),
	)
	return parser.parse_args()


def _sorted_dirs(path: Path) -> List[Path]:
	return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda x: x.name)


def _sorted_images(path: Path) -> List[Path]:
	exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
	return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts], key=lambda x: x.name)


def _center_crop_or_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
	h, w = img.shape[:2]
	if h >= target_h and w >= target_w:
		top = (h - target_h) // 2
		left = (w - target_w) // 2
		return img[top : top + target_h, left : left + target_w]
	return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def detect_blue_mask(img_bgr: np.ndarray, blue_tol: int = DEFAULT_BLUE_TOL) -> np.ndarray:
	"""Detect trajectory pixels using near-pure-blue matching.

	In write_mp3d_data_history.py, trajectory is drawn as single_color=(0,0,255)
	in RGB, then saved via RGB->BGR before cv2.imwrite. So saved trajectory is
	near BGR=(255,0,0). We keep a tolerance for JPEG artifacts.
	"""
	tol = max(0, int(blue_tol))
	b = img_bgr[:, :, 0].astype(np.int16)
	g = img_bgr[:, :, 1].astype(np.int16)
	r = img_bgr[:, :, 2].astype(np.int16)

	return (b >= 255 - tol) & (g <= tol) & (r <= tol)


def draw_token_grid_overlay(
	img_bgr: np.ndarray,
	blue_mask: np.ndarray,
	token_mask: np.ndarray,
	token_pixels: int,
) -> np.ndarray:
	"""Render a debug image with blue-pixel mask and 16x16 token grid overlay."""
	vis = img_bgr.copy()

	# Highlight detected blue trajectory pixels in red for quick verification.
	red = np.zeros_like(vis)
	red[:, :, 2] = 255
	blended = cv2.addWeighted(vis, 1.0, red, 0.35, 0.0)
	vis[blue_mask] = blended[blue_mask]

	grid_size = token_mask.shape[0]
	h, w = vis.shape[:2]

	for row in range(grid_size):
		for col in range(grid_size):
			if token_mask[row, col] == 1:
				y0 = row * token_pixels
				y1 = min(y0 + token_pixels, h)
				x0 = col * token_pixels
				x1 = min(x0 + token_pixels, w)
				cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)

	# Draw a subtle grey grid on a separate layer and blend it for readability.
	grid_layer = vis.copy()
	for r in range(1, grid_size):
		y = min(r * token_pixels, h - 1)
		cv2.line(grid_layer, (0, y), (w - 1, y), (140, 140, 140), 1)
	for c in range(1, grid_size):
		x = min(c * token_pixels, w - 1)
		cv2.line(grid_layer, (x, 0), (x, h - 1), (140, 140, 140), 1)
	vis = cv2.addWeighted(grid_layer, 0.6, vis, 0.4, 0.0)

	return vis


def build_trajectory_token_mask(
	img_bgr: np.ndarray,
	grid_size: int = 16,
	token_pixels: int = DEFAULT_TOKEN_PIXELS,
	min_blue_pixels: int = 2,
	blue_tol: int = DEFAULT_BLUE_TOL,
) -> List[List[int]]:
	target_h = grid_size * token_pixels
	target_w = grid_size * token_pixels
	proc = _center_crop_or_resize(img_bgr, target_h, target_w)

	blue_mask = detect_blue_mask(proc, blue_tol=blue_tol)

	token_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
	for row in range(grid_size):
		for col in range(grid_size):
			y0 = row * token_pixels
			y1 = y0 + token_pixels
			x0 = col * token_pixels
			x1 = x0 + token_pixels
			patch = blue_mask[y0:y1, x0:x1]
			token_mask[row, col] = 1 if int(np.count_nonzero(patch)) >= min_blue_pixels else 0

	return token_mask.tolist()


def build_trajectory_token_mask_with_debug(
	img_bgr: np.ndarray,
	grid_size: int = 16,
	token_pixels: int = DEFAULT_TOKEN_PIXELS,
	min_blue_pixels: int = 2,
	blue_tol: int = DEFAULT_BLUE_TOL,
):
	"""Return token mask and intermediates for optional visualization output."""
	target_h = grid_size * token_pixels
	target_w = grid_size * token_pixels
	proc = _center_crop_or_resize(img_bgr, target_h, target_w)
	blue_mask = detect_blue_mask(proc, blue_tol=blue_tol)

	token_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
	for row in range(grid_size):
		for col in range(grid_size):
			y0 = row * token_pixels
			y1 = y0 + token_pixels
			x0 = col * token_pixels
			x1 = x0 + token_pixels
			patch = blue_mask[y0:y1, x0:x1]
			token_mask[row, col] = 1 if int(np.count_nonzero(patch)) >= min_blue_pixels else 0

	return token_mask.tolist(), proc, blue_mask, token_mask


def parse_step_idx(step_name: str) -> Optional[int]:
	try:
		return int(step_name)
	except ValueError:
		return None


def _to_project_relative(path: Path) -> str:
	try:
		return str(path.relative_to(project_root))
	except Exception:
		return str(path)


def build_episode_entry(
	image_root: Path,
	scene_dir: Path,
	episode_dir: Path,
	grid_size: int,
	token_pixels: int,
	min_blue_pixels: int,
	blue_tol: int,
	include_empty_steps: bool,
	vis_image: bool,
	vis_root: Optional[Path],
) -> Dict:
	steps: List[Dict] = []

	step_dirs = _sorted_dirs(episode_dir)
	for step_dir in tqdm(
		step_dirs,
		desc=f"Steps {scene_dir.name}/{episode_dir.name}",
		leave=False,
	):
		step_idx = parse_step_idx(step_dir.name)
		if step_idx is None:
			continue

		image_files = _sorted_images(step_dir)
		if not image_files and not include_empty_steps:
			continue

		image_paths: List[str] = []
		image_masks: List[List[List[int]]] = []

		for img_path in tqdm(
			image_files,
			desc=f"Images {scene_dir.name}/{episode_dir.name}/{step_dir.name}",
			leave=False,
		):
			img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
			if img is None:
				continue
			image_paths.append(_to_project_relative(img_path))
			token_list, proc, blue_mask, token_mask_arr = build_trajectory_token_mask_with_debug(
				img,
				grid_size=grid_size,
				token_pixels=token_pixels,
				min_blue_pixels=min_blue_pixels,
				blue_tol=blue_tol,
			)
			image_masks.append(token_list)

			if vis_image:
				if vis_root is None:
					raise ValueError("vis_root must not be None when vis_image is enabled")
				debug_img = draw_token_grid_overlay(proc, blue_mask, token_mask_arr, token_pixels)
				rel = img_path.relative_to(image_root)
				debug_path = vis_root / rel
				debug_path.parent.mkdir(parents=True, exist_ok=True)
				cv2.imwrite(str(debug_path), debug_img)

		if not image_paths and not include_empty_steps:
			continue

		steps.append(
			{
				"step_idx": step_idx,
				"history_vis_single": image_paths,
				"history_vis_single_trajectory_mask": image_masks,
			}
		)

	scene_id = scene_dir.name
	episode_id = int(episode_dir.name) if episode_dir.name.isdigit() else episode_dir.name
	return {
		"scene_id": scene_id,
		"episode_id": episode_id,
		"steps": steps,
	}


def write_single_episode_json(
	episode: Dict,
	output_root: Path,
	grid_size: int,
	token_pixels: int,
	image_root: Path,
) -> Path:
	"""Write one episode JSON to output_root/scene_id/episode_id.json."""
	output_root.mkdir(parents=True, exist_ok=True)
	scene_id = str(episode.get("scene_id"))
	episode_id = str(episode.get("episode_id"))

	scene_dir = output_root / scene_id
	scene_dir.mkdir(parents=True, exist_ok=True)

	episode_out = {
		"episode_id": episode.get("episode_id"),
		"scene_id": episode.get("scene_id"),
		"steps": episode.get("steps", []),
		"mask_meta": {
			"grid_size": grid_size,
			"token_pixels": token_pixels,
			"image_root": _to_project_relative(image_root),
		},
	}

	out_file = scene_dir / f"{episode_id}.json"
	with out_file.open("w", encoding="utf-8") as f:
		json.dump(episode_out, f, indent=2)
	return out_file


def generate_masks(
	image_root: Path,
	output_root: Path,
	grid_size: int,
	token_pixels: int,
	min_blue_pixels: int,
	blue_tol: int,
	include_empty_steps: bool,
	scene_id: Optional[str],
	episode_id: Optional[str],
	vis_image: bool,
	vis_root: Optional[Path],
) -> Dict:
	episodes: List[Dict] = []
	written = 0

	scene_dirs = _sorted_dirs(image_root)
	for scene_dir in tqdm(scene_dirs, desc="Scenes", leave=True):
		if scene_id is not None and scene_dir.name != scene_id:
			continue
		episode_dirs = _sorted_dirs(scene_dir)
		for episode_dir in tqdm(
			episode_dirs,
			desc=f"Episodes {scene_dir.name}",
			leave=False,
		):
			if episode_id is not None and episode_dir.name != episode_id:
				continue
			episode = build_episode_entry(
					image_root=image_root,
					scene_dir=scene_dir,
					episode_dir=episode_dir,
					grid_size=grid_size,
					token_pixels=token_pixels,
					min_blue_pixels=min_blue_pixels,
					blue_tol=blue_tol,
					include_empty_steps=include_empty_steps,
					vis_image=vis_image,
					vis_root=vis_root,
				)
			episodes.append(episode)
			write_single_episode_json(
				episode=episode,
				output_root=output_root,
				grid_size=grid_size,
				token_pixels=token_pixels,
				image_root=image_root,
			)
			written += 1

	return {
		"image_root": _to_project_relative(image_root),
		"grid_size": grid_size,
		"token_pixels": token_pixels,
		"episodes": episodes,
		"written": written,
	}


def main() -> None:
	args = parse_args()
	image_root = Path(args.image_root) if args.image_root else Path(PATH[args.mode])
	output_root = Path(args.output_root) if args.output_root else Path(OUTPUT_PATH[args.mode])
	if not image_root.exists():
		raise FileNotFoundError(f"Input path does not exist: {image_root}")

	vis_root: Optional[Path] = None
	if args.vis_image:
		if args.vis_root is not None:
			vis_root = Path(args.vis_root)
		else:
			vis_root = Path(VIS_PATH[args.mode])
		vis_root.mkdir(parents=True, exist_ok=True)

	result = generate_masks(
		image_root=image_root,
		output_root=output_root,
		grid_size=args.grid_size,
		token_pixels=args.token_pixels,
		min_blue_pixels=args.min_blue_pixels,
		blue_tol=args.blue_tol,
		include_empty_steps=args.include_empty_steps,
		scene_id=args.scene_id,
		episode_id=args.episode_id,
		vis_image=args.vis_image,
		vis_root=vis_root,
	)

	written = int(result.get("written", 0))

	print(f"mode: {args.mode}")
	print(f"image_root: {image_root}")
	print(f"Saved episode json files to: {output_root}")
	print(f"Processed episodes: {len(result['episodes'])}")
	print(f"Written episode json files: {written}")
	if args.vis_image:
		print(f"Saved visualization images to: {vis_root}")


if __name__ == "__main__":
	main()
