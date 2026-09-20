import json, os, argparse
import re

import torch
from prompt import *

# ---------------- ARGPARSE ----------------
# you should run this script in the dir CL_COTNav
# python scripts/scripts_for_generating_data/dataset_reconstruction_for_intervl/dataset_reconstruct_for_internvl.py --mode r2r
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="r2r", required=True, choices=["r2r", "rxr-en-US", "rxr-en-IN"], help="Which data split to process")

    p.add_argument("--if_verbose", default=False, help="output more information")
    p.add_argument(
        "--template",
        nargs="+",
        default=["RGB_HisKFSingleColor_ActionPixelpar", "RGB_HisKFSingleColor_ActionPixelSeq"],
        choices=[
            "RGB", "RGB_HisKFSingleColor", "RGB_HisKFSingleColor_trajMASK",
            # ablation for history image
            "RGB_His1Interval8", "RGB_His5Interval8", "RGB_HisUniformSampling8", "RGB_HisDiffFrequency8", "RGB_HisKFNoColor",
            # output paradigm
            "RGB_HisKFSingleColor_1Action", "RGB_HisKFSingleColor_4Action", "RGB_HisKFSingleColor_ActionPixelpar", "RGB_HisKFSingleColor_ActionPixelSeq"
        ],
        help="Which conversation template to use",
    )
    return p

args = build_parser().parse_args()

MIN_PATH_LENGTH = 2.5


def format_pixel_3digit(value):
    pixel = min(999, max(0, int(round(value * 1000))))
    return f"{pixel:03d}"


def normalize_action_name(action_name):
    if action_name is None:
        return "STOP"
    return str(action_name).replace("_", " ").upper()


def get_mask_subdir_candidates(mode):
    if mode == "r2r":
        return ["train_history/annotation_mask"]
    if mode == "rxr-en-US":
        return ["train_history/annotation_mask"]
    if mode == "rxr-en-IN":
        # Prefer annotation_mask because adjusted_annotations may not contain
        # history_vis_single_trajectory_mask for each step.
        return ["train_history/annotation_mask", "train_history/adjusted_annotations"]
    raise ValueError(f"Unsupported mode for mask loading: {mode}")


def load_traj_mask_by_step_idx(mask_file_path):
    with open(mask_file_path, "r") as f:
        mask_data = json.load(f)

    masks_by_step_idx = {}
    for mask_step in mask_data.get("steps", []):
        step_idx = mask_step.get("step_idx")
        if step_idx is None:
            continue
        masks_by_step_idx[step_idx] = {
            "paths": mask_step.get("history_vis_single", []),
            "masks": mask_step.get("history_vis_single_trajectory_mask"),
        }
    return masks_by_step_idx


def normalize_history_path_key(path):
    path = str(path).replace("\\", "/")
    marker = "/train_history/"
    if marker in path:
        return path.split(marker, 1)[1]
    marker = "train_history/"
    if marker in path:
        return path.split(marker, 1)[1]
    return "/".join(path.split("/")[-5:])


def infer_history_mask_step_idx(step):
    history_paths = step.get("history_vis_single", [])
    if not history_paths:
        return step.get("step_idx")
    history_path = str(history_paths[0]).replace("\\", "/")
    try:
        return int(history_path.rstrip("/").split("/")[-2])
    except (TypeError, ValueError, IndexError):
        return step.get("step_idx")


def build_history_images_and_traj_mask(step, traj_mask_value, max_history=8, if_verbose=False, context=''):
    """
    Build aligned history image list and trajectory mask list.
    Rules:
    1) history length <= max_history
    2) output history keeps the existing latest->oldest order
    3) masks are aligned by history image path when mask metadata is available
    """
    if 'history_vis_single' not in step:
        raise ValueError(f"Missing history_vis_single in {context}")

    history_full = list(step['history_vis_single'])
    history_selected = history_full[:max_history]

    mask_paths = None
    if isinstance(traj_mask_value, dict):
        mask_paths = traj_mask_value.get("paths")
        mask_full = traj_mask_value.get("masks")
    else:
        mask_full = traj_mask_value

    if not isinstance(mask_full, list):
        if if_verbose:
            print(f"Skipping {context}: trajectory mask value is not a list")
        return None, None

    mask_full = list(mask_full)

    # If mask includes current frame (current + history), drop current entry.
    if len(mask_full) == len(history_full) + 1:
        mask_full = mask_full[1:]

    mask_selected = None
    if isinstance(mask_paths, list) and len(mask_paths) == len(mask_full):
        mask_by_path = {
            normalize_history_path_key(path): mask
            for path, mask in zip(mask_paths, mask_full)
        }
        mask_selected = [
            mask_by_path.get(normalize_history_path_key(hist_path))
            for hist_path in history_selected
        ]
        if any(mask is None for mask in mask_selected):
            if if_verbose:
                print(f"Skipping {context}: failed to align one or more trajectory masks by path")
            return None, None
    elif len(mask_full) < len(history_selected):
        if if_verbose:
            print(
                f"Skipping {context}: mask shorter than selected history "
                f"({len(mask_full)} < {len(history_selected)})"
            )
        return None, None
    else:
        # Legacy fallback: older mask files did not carry image paths. Those
        # masks were produced by sorting image filenames ascending, while
        # history_vis_single is stored latest->oldest.
        mask_selected = mask_full[:len(history_selected)][::-1]

    history_images_rel = ["/".join(hist_path.split("/")[3:]) for hist_path in history_selected]
    return history_images_rel, mask_selected

def main(args, template):
    PWD = os.path.dirname(os.path.abspath(__file__))

    if args.mode == 'r2r':
        ROOT_DIR = ["training_data/data_mp3d_r2r/v1-3"]
        DATA_NAME = ["r2r"]
    elif args.mode == 'rxr-en-US':
        ROOT_DIR = ["training_data/data_mp3d_rxr/rxr_guide_en-US"]
        DATA_NAME = ["rxr-en-US"]
    elif args.mode == 'rxr-en-IN':
        ROOT_DIR = ["training_data/data_mp3d_rxr/rxr_guide_en-IN"]
        DATA_NAME = ["rxr-en-IN"]
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if len(ROOT_DIR) != len(DATA_NAME):
        raise ValueError(f"ROOT_DIR and DATA_NAME must have same length, got {len(ROOT_DIR)} and {len(DATA_NAME)}")

    TEMPLATE = template
    IF_VERBOSE = args.if_verbose


    output_file = f"training_data/jsonl/{args.mode}/{TEMPLATE}.jsonl"
    print(f"output_file: {output_file}")
    
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = 0  # counter for number of data pairs

    with open(output_file, "w") as jsonl_file:

        for root_dir, data_name in zip(ROOT_DIR, DATA_NAME):
            # use annotations to find other files
            base_dir = os.path.join(root_dir, "annotations")
            print(f"base_dir ({data_name}): {base_dir}")

            if not os.path.isdir(base_dir):
                raise FileNotFoundError(f"Annotation directory not found: {base_dir}")

            # for each scene
            for scene_id in os.listdir(base_dir):
                scene_dir = os.path.join(base_dir, scene_id)
                if not os.path.isdir(scene_dir):
                    continue  # skip non-folder files

                print(f"Processing {data_name} scene: {scene_id}")

                # for each episode/trajectory
                for file_name in os.listdir(scene_dir):
                    if not file_name.endswith(".json"):
                        continue

                    file_path = os.path.join(scene_dir, file_name)
                    
                    # for debug
                    print(f"Loading file: {file_path}")
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"\nJSON decode error while reading: {file_path}")
                        print(
                            f"Details -> line: {e.lineno}, column: {e.colno}, char: {e.pos}, msg: {e.msg}"
                        )
                        raise

                    ep_id = file_name.replace(".json", "")

                    traj_mask_by_step_idx = {}
                    if TEMPLATE == "RGB_HisKFSingleColor_trajMASK":
                        mask_file_path = None
                        checked_candidates = []
                        for mask_subdir in get_mask_subdir_candidates(args.mode):
                            candidate_path = os.path.join(root_dir, mask_subdir, scene_id, file_name)
                            checked_candidates.append(candidate_path)
                            if not os.path.isfile(candidate_path):
                                continue

                            candidate_masks = load_traj_mask_by_step_idx(candidate_path)
                            # Keep candidate only if at least one step has a usable mask.
                            has_any_mask = any(
                                isinstance(v, dict) and v.get("masks") is not None
                                for v in candidate_masks.values()
                            )
                            if has_any_mask:
                                mask_file_path = candidate_path
                                traj_mask_by_step_idx = candidate_masks
                                break

                        if mask_file_path is None:
                            print(
                                f"Warning: no usable mask file for {args.mode}/{scene_id}/{file_name}. "
                                f"Checked: {checked_candidates}. Steps without masks will be skipped."
                            )

                    print(f"Loaded scene dir: {scene_dir}, file name: {file_name}")

                    # Skip very short trajectories at the episode level.
                    path_length = data.get("final_metrics", {}).get("path_length")
                    if path_length is None:
                        if IF_VERBOSE:
                            print(f"Skipping {scene_dir}/{ep_id}: Missing final_metrics.path_length")
                        continue
                    if path_length < MIN_PATH_LENGTH:
                        if IF_VERBOSE:
                            print(
                                f"Skipping {scene_dir}/{ep_id}: path_length={path_length:.4f} < {MIN_PATH_LENGTH}"
                            )
                        continue

                    # for each step
                    for i, step in enumerate(data['steps']):
                        if step['step_idx'] == 0:
                            continue

                        step_ground_truth = None

                        # base
                        if TEMPLATE == "RGB":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:])
                                ],
                                "width": [272],
                                "height": [272],
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}\nOutput: \n<GOAL_PIXEL_X>, <GOAL_PIXEL_Y>\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    },
                                ],
                            }
                        
                        elif TEMPLATE == "RGB_HisKFSingleColor":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_single' in step:
                                for hist_path in step['history_vis_single'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_single in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_HisKFSingleColor_trajMASK":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            mask_step_idx = infer_history_mask_step_idx(step)
                            traj_mask_value = traj_mask_by_step_idx.get(mask_step_idx)
                            if traj_mask_value is None:
                                traj_mask_value = traj_mask_by_step_idx.get(step['step_idx'])
                            if traj_mask_value is None:
                                print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing history_vis_single_trajectory_mask")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            history_images_rel, traj_mask_value_aligned = build_history_images_and_traj_mask(
                                step=step,
                                traj_mask_value=traj_mask_value,
                                max_history=8,
                                if_verbose=IF_VERBOSE,
                                context=f"{scene_dir}/{ep_id} step {i}",
                            )
                            if history_images_rel is None or traj_mask_value_aligned is None:
                                print(f"Skipping {scene_dir}/{ep_id} step {i}: Failed to build aligned history and mask")
                                continue

                            for hist_path_rel in history_images_rel:
                                images.append(hist_path_rel)
                                widths.append(256)
                                heights.append(256)

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    },
                                    {
                                        "from": "Traj_MASK",
                                        "value": traj_mask_value_aligned
                                    }
                                ],
                            }
                            
                        # history ablation study
                        elif TEMPLATE == "RGB_HisKFNoColor":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_no_traj' in step:
                                for hist_path in step['history_vis_no_traj'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_no_traj in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{HIS_KEYFRAME_NO_COLORED(len(images)-1)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_His1Interval8":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_interval_1' in step:
                                for hist_path in step['history_vis_interval_1'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_interval_1 in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{build_interval_history_prompt(len(images)-1, 1, len(images)-1)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_His5Interval8":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_interval_5' in step:
                                for hist_path in step['history_vis_interval_5'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_interval_5 in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{build_interval_history_prompt(len(images)-1, 5, len(images)-1)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_HisUniformSampling8":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_uniform_sampling' in step:
                                for hist_path in step['history_vis_uniform_sampling'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_uniform_sampling in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{build_uniform_history_prompt(len(images)-1, 8)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_HisDiffFrequency8":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            gpt_value = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_diff_frequency' in step:
                                for hist_path in step['history_vis_diff_frequency'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_diff_frequency in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{build_diff_frequency_history_prompt(len(images)-1, 8, 4)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }
                        
                        # output paradigm ablation study
                        elif TEMPLATE == "RGB_HisKFSingleColor_1Action":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            current_action = normalize_action_name(step.get('action_name'))
                            if current_action == "MOVE FORWARD":
                                gpt_value = "MOVE_FORWARD"
                            elif current_action == "TURN LEFT":
                                gpt_value = "TURN_LEFT"
                            elif current_action == "TURN RIGHT":
                                gpt_value = "TURN_RIGHT"
                            elif current_action == "STOP":
                                gpt_value = "STOP"
                            else:
                                raise ValueError(f"Unknown action name {current_action} in {scene_dir}/{ep_id} step {i}")

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_single' in step:
                                for hist_path in step['history_vis_single'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_single in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_action}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}\n{OUTPUT_ACTION}\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_HisKFSingleColor_4Action":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            action_tokens = []
                            for action_step in data['steps'][i:i + 4]:
                                action_name = normalize_action_name(action_step.get('action_name'))
                                if action_name == "MOVE FORWARD":
                                    action_tokens.append("MOVE_FORWARD")
                                elif action_name == "TURN LEFT":
                                    action_tokens.append("TURN_LEFT")
                                elif action_name == "TURN RIGHT":
                                    action_tokens.append("TURN_RIGHT")
                                elif action_name == "STOP":
                                    action_tokens.append("STOP")
                                else:
                                    raise ValueError(f"Unknown action name {action_name} in {scene_dir}/{ep_id} step {i}")
                            while len(action_tokens) < 4:
                                action_tokens.append("STOP")
                            gpt_value = ", ".join(action_tokens)

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_single' in step:
                                for hist_path in step['history_vis_single'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_single in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_4action}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}\n{OUTPUT_ACTION_4}\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_HisKFSingleColor_ActionPixelpar":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # output
                            waypoint_pixel_x, waypoint_pixel_y = step['waypoint_pixel_4_rotio']['x'], step['waypoint_pixel_4_rotio']['y']
                            x = min(999, int(round(waypoint_pixel_x * 1000)))
                            y = min(999, int(round(waypoint_pixel_y * 1000)))
                            pixel_gt = f"({x:03d}, {y:03d})"

                            action_name = normalize_action_name(step.get('action_name'))
                            if action_name == "TURN LEFT" or pixel_gt == "(000, 500)":
                                gpt_value = "LEFT"
                            elif action_name == "TURN RIGHT" or pixel_gt == "(999, 500)":
                                gpt_value = "RIGHT"
                            elif action_name == "STOP":
                                gpt_value = "STOP"
                            elif action_name == "MOVE FORWARD":
                                gpt_value = f"FORWARD {pixel_gt}"
                            else:
                                raise ValueError(f"Unknown action name {action_name} in {scene_dir}/{ep_id} step {i}")

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_single' in step:
                                for hist_path in step['history_vis_single'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_single in {ep_id} step {i}")

                            step_ground_truth = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human", 
                                        "value": f"{introduction_action_pixel}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}\n{OUTPUT_ACTION_PIXEL}\n"
                                    },
                                    {
                                        "from": "gpt", 
                                        "value": gpt_value
                                    }
                                ],
                            }

                        elif TEMPLATE == "RGB_HisKFSingleColor_ActionPixelSeq":
                            # input
                            if step['waypoint_pixel_4_rotio'] is None:
                                if IF_VERBOSE:
                                    print(f"Skipping {scene_dir}/{ep_id} step {i}: Missing waypoint_pixel_4_rotio")
                                continue
                            if IF_VERBOSE:
                                print(f"id is {step['step_idx']}")
                            instruction = step["instruction"]

                            # GT pixel from normalized coordinates
                            waypoint_pixel_x = step['waypoint_pixel_4_rotio']['x']
                            waypoint_pixel_y = step['waypoint_pixel_4_rotio']['y']
                            pixel_gt = f"{format_pixel_3digit(waypoint_pixel_x)}, {format_pixel_3digit(waypoint_pixel_y)}"

                            # Round-1 GT label from pixel region
                            if pixel_gt == "000, 500":
                                action_seq_gt = "TURN_LEFT"
                            elif pixel_gt == "999, 500":
                                action_seq_gt = "TURN_RIGHT"
                            elif pixel_gt == "500, 999":
                                action_seq_gt = "STOP"
                            else:
                                action_seq_gt = "SELECT_PIXEL"

                            images = [
                                "/".join(step['rgb_padded_down_path'].split("/")[3:]), # Padded RGB image
                            ]
                            widths = [272]
                            heights = [272]

                            if 'history_vis_single' in step:
                                for hist_path in step['history_vis_single'][:8]:
                                    images.append("/".join(hist_path.split("/")[3:]))
                                    widths.append(256)
                                    heights.append(256)
                            else:
                                raise ValueError(f"Missing history_vis_single in {ep_id} step {i}")

                            # Round 1 sample: action decision only
                            step_ground_truth_round1 = {
                                "id": f"{data_name}-{scene_id}-{ep_id}-{i}-r1",
                                "image": images,
                                "width": widths,
                                "height": heights,
                                "conversations": [
                                    {
                                        "from": "human",
                                        "value": (
                                            f"{introduction_pixel_in_text}Instruction: {instruction}\n\n"
                                            f"Input: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}"
                                            "Output: only one token from {TURN_LEFT, TURN_RIGHT, STOP, SELECT_PIXEL}.\n"
                                        )
                                    },
                                    {
                                        "from": "gpt",
                                        "value": action_seq_gt
                                    }
                                ],
                            }
                            jsonl_file.write(json.dumps(step_ground_truth_round1) + "\n")
                            count += 1

                            # Round 2 sample: only for SELECT_PIXEL
                            if action_seq_gt == "SELECT_PIXEL":
                                step_ground_truth_round2 = {
                                    "id": f"{data_name}-{scene_id}-{ep_id}-{i}-r2",
                                    "image": images,
                                    "width": widths,
                                    "height": heights,
                                    "conversations": [
                                        {
                                            "from": "human",
                                            "value": (
                                                f"{introduction_pixel_in_text}Instruction: {instruction}\n\n"
                                                f"Input: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}"
                                                "First round decision: SELECT_PIXEL.\n"
                                                "Output: a pixel coordinate pair in the format XXX, YYY\n"
                                            )
                                        },
                                        {
                                            "from": "gpt",
                                            "value": pixel_gt
                                        }
                                    ],
                                }
                                jsonl_file.write(json.dumps(step_ground_truth_round2) + "\n")
                                count += 1

                            continue


                        if step_ground_truth is not None:
                            jsonl_file.write(json.dumps(step_ground_truth) + "\n")
                            count += 1  # increment counter

    print(f"\n✅ Finished! Generated {count} data pairs in {output_file}")



if __name__ == "__main__":
    args = build_parser().parse_args()
    template = args.template
    for t in template:
        main(args, t)
