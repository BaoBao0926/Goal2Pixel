import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
# import grounding_dino.groundingdino.datasets.transforms as T

# TODO:
# 21 classes for indoor objects
TEXT_PROMPT = (
    # original 21 classes
    "chair. table. picture. cushion. sofa. fireplace. cabinet. seating. stool. "
    "shower. tv monitor. towel. gym equipment. sink. clothes. bathtub. "
    "counter. chest of drawers. bed. toilet. plant. "
    # additional 
    "pillow. wardrobe. nightstand. dresser. painting. lamp. curtain. refrigerator. oven. stove. closet. "
    # room
    "living room. bedroom. kitchen. bathroom. dining room. corridor. office room. gym. lounge. laundry room."
)


def detect_segment(config, sam2_predictor, grounding_model,
                   rgb_path, save_idx, OUTPUT_DIR, DUMP_JSON_RESULTS=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = TEXT_PROMPT
    image_source, image = load_image(rgb_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=config.BOX_THRESHOLD,
        text_threshold=config.TEXT_THRESHOLD,
        device=device
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    if input_boxes.size == 0:
        # 可视化：直接保存原图或加一行文字
        img = cv2.imread(rgb_path)
        cv2.imwrite(f"{OUTPUT_DIR}/{save_idx:03d}.png", img)

        # JSONL 追加一条空注释
        results = {
            "image_path": rgb_path,
            "annotations": [],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.jsonl"), "a") as f:
            f.write(json.dumps(results) + "\n")
        return  # 早退

    # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    torch.autocast(device_type=device, dtype=torch.float16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(rgb_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(f"{OUTPUT_DIR}/{save_idx:03d}.png", annotated_frame)


    # mask_annotator = sv.MaskAnnotator()
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    """
    Dump the results in standard format and save as json files
    """

    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": rgb_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }


        if save_idx == 0 and os.path.exists(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.jsonl")):
            os.remove(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.jsonl"))

        json_file = os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.jsonl")

        # 每次写一行
        with open(json_file, "a") as f:
            f.write(json.dumps(results) + "\n")