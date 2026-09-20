import cv2
import os
import numpy as np
import skimage.morphology
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import line_aa, line
import math
import torch
from PIL import ImageDraw, Image
from typing import Optional, Sequence, Tuple, List, Dict, Any


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def line_list(text, line_length=22):
    text_list = []
    for i in range(0, len(text), line_length):
        text_list.append(text[i:(i + line_length)])
    return text_list


def add_text_list(image: np.ndarray, text_list: list, position=(10, 20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(0, 0, 0), thickness=1, highlight_line_index=[]):
    highlight_color = (0, 0, 0)
    not_highlight_color = (128, 128, 128)
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * 15)
        color = highlight_color if len(highlight_line_index) == 0 or i in highlight_line_index else not_highlight_color
        cv2.putText(image, text, position_i, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


# overall image size and margins
HEIGHT = 1000
WIDTH = 2500
MARGIN_TOP = 100
MARGIN_LEFT = 120
MARGIN_BOTTOM = 25

# widths of different panels
SMALL_PANEL_W = 215      # Goal Object / Reasoning Graph
MID_PANEL_L_W = 640      # RGB / Semantic
MID_PANEL_W   = 480      # Occ / Occ + Frontier GT
COLUMN_GAP    = 40         # 列与列之间的空隙，避免标题挤在一起

COL1_LEFT = MARGIN_LEFT
COL2_LEFT = MARGIN_LEFT + SMALL_PANEL_W + COLUMN_GAP
COL3_LEFT = MARGIN_LEFT + SMALL_PANEL_W + COLUMN_GAP + MID_PANEL_L_W + COLUMN_GAP
COL4_LEFT = MARGIN_LEFT + SMALL_PANEL_W + COLUMN_GAP + MID_PANEL_L_W + COLUMN_GAP + MID_PANEL_W + COLUMN_GAP

# heights of different panels
PANEL_H_SMALL = 240
PANEL_H = 480
ROW_GAP = 40         # 行与行之间的空隙，避免标题挤在一起

ROW1_TOP = PANEL_H_SMALL + ROW_GAP
ROW2_TOP = PANEL_H + COLUMN_GAP + + MARGIN_BOTTOM
ROW3_TOP = PANEL_H + COLUMN_GAP + PANEL_H + COLUMN_GAP + MARGIN_BOTTOM


# === 标题通用参数 ===
TITLE_BASE_Y = 50
TITLE_LEFT_PADDING = 25     # 轻微向右内缩，避免贴边
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
COLOR = (20, 20, 20)        # BGR
THICKNESS = 2


def _draw_title(vis, text, box_left, box_width, base_y=TITLE_BASE_Y, dx=0, dy=0):
    """在给定盒子范围内水平居中绘制标题，支持微调 dx/dy。"""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONTSCALE, THICKNESS)
    x = box_left + (box_width - tw) // 2 + TITLE_LEFT_PADDING + dx
    y = (base_y + th) // 2 + dy
    return cv2.putText(vis, text, (x, y), FONT, FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)

def init_vis_image(tmp_dir):
    """
    Background image for visualization
    """
    vis_image = np.ones(
        (HEIGHT - MARGIN_TOP,
         WIDTH - MARGIN_LEFT - MARGIN_BOTTOM,
         3),
        dtype=np.uint8
    ) * 255

    # Goal Object / Goal Graph for Reasoning - COL1_LEFT
    vis_image = _draw_title(vis_image, "Goal Object Category",
                            box_left=COL1_LEFT, box_width=SMALL_PANEL_W)

    vis_image = _draw_title(vis_image, "Ground Truth Action",
                            box_left=COL1_LEFT, box_width=SMALL_PANEL_W,
                            dy=ROW1_TOP)

    # RGB / Semantic - COL2_LEFT
    vis_image = _draw_title(vis_image, "Observation RGB",
                            box_left=COL2_LEFT, box_width=MID_PANEL_L_W)

    vis_image = _draw_title(vis_image, "Observation Semantic",
                            box_left=COL2_LEFT, box_width=MID_PANEL_L_W,
                            dy=ROW2_TOP)

    # Occ / Occ + Frontier GT - COL3_LEFT
    vis_image = _draw_title(vis_image, "Occupancy Map",
                            box_left=COL3_LEFT, box_width=MID_PANEL_W)

    vis_image = _draw_title(vis_image, "Occupancy + Frontier GT",
                            box_left=COL3_LEFT, box_width=MID_PANEL_W,
                            dy=ROW2_TOP)

    # Semantic Map - COL4_LEFT
    vis_image = _draw_title(vis_image, "Semantic Map",
                            box_left=COL4_LEFT, box_width=MID_PANEL_W)

    # 保存背景图（缩放为 1/2）
    os.makedirs(tmp_dir, exist_ok=True)
    h, w, _ = vis_image.shape
    cv2.imwrite(os.path.join(tmp_dir, 'background_debug.png'),
                cv2.resize(vis_image, (w, h)))

    return vis_image



def plot_local_bev_pil(
    local_map,
    planner_pose_inputs_row: Optional[Sequence[float]] = None,
    *,
    # Which rendering mode?
    classify_with_explored: bool = False,
    # Occupancy/exploration thresholds
    occ_thr: float = 0.5,
    exp_thr: float = 0.1,
    # Tones (L mode)
    gray_unknown: int = 127,
    gray_free: int = 255,
    gray_occ: int = 0,
    # --- Trail (trajectory) appearance ---
    trail_color: Tuple[int, int, int] = (0, 0, 255),  # Blue
    trail_alpha: int = 200,
    trail_erode_ksize: int = 3,
    # --- Arrow (robot pose and orientation) appearance ---
    arrow_color: Tuple[int, int, int] = (255, 0, 0),      # Red
    arrow_len_px: int = 22,
    arrow_width: int = 4,
    head_length: int = 10,
    head_width: int = 8,
    mark_radius: int = 4,
    # Coordinate/display options
    flip_y_up: bool = True,                 # True → y-axis points upward in the image
    output_size: Optional[int] = None,      # if not None, resize to NxN before drawing arrow
    # Pose→pixel mapping (optional, only needed when using window indices)
    map_resolution: Optional[float] = None, # in m/cell (needed with gx1..gy2 path)
    # --- NEW: Rotation and zoom ---
    rotate_to_heading: bool = True,         # If True, rotate map so agent heading points upward
    crop_radius: Optional[int] = None,      # If set, crop a window of this radius (in cells) around agent
    # --- NEW: Multi-color trajectory ---
    key_frame: Optional[List[int]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> Image.Image:
    """
    Returns a PIL.Image visualizing a local BEV map.

    Channels expected in local_map[N,C,H,W]:
      ch 0: occupancy likelihood in [0,1] (0=free, 1=occupied)
      ch 1: explored mask/confidence (higher => explored)
      ch 2: agent footprint (optional)
      ch 3: visited/trail mask (optional)

    Two modes:
      • classify_with_explored=False (default):
          Base = grayscale from occupancy (0→white, 1→black), no explicit “unknown”.
      • classify_with_explored=True:
          Base = {unknown, free, occupied} from (exp_thr, occ_thr).

    The function draws:
      • optional trail overlay from ch-3 (cyan with alpha)
      • agent arrow from planner_pose_inputs_row (yaw in degrees)
      • optional center mark from ch-2 footprint
    """
    # --- extract arrays ---
    lm = local_map.detach().cpu().numpy() if isinstance(local_map, torch.Tensor) else np.asarray(local_map)
    lm = lm.squeeze(0).transpose(1,2,0)
    assert lm.ndim == 3, f"local_map must be (H,W,C); got {lm.shape}"
    H, W, C = lm.shape

    occ = np.clip(lm[:, :, 0], 0.0, 1.0)                       # (H,W)
    exp = lm[:, :, 1] if C >= 2 else np.ones_like(occ)         # (H,W)
    agent_mask = (lm[:, :, 2] > 0.5) if C >= 3 else None       # (H,W)
    trail_mask = (lm[:, :, 3] > 0.5) if C >= 4 else None       # (H,W)

    # --- Get agent position and yaw for rotation/crop ---
    agent_row, agent_col, agent_yaw_deg = None, None, 0.0
    if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 7 and map_resolution is not None:
        # Extract from planner_pose_inputs: [x_m, y_m, yaw_deg, gx1, gx2, gy1, gy2]
        start_x_m, start_y_m, start_o_deg, gx1, gx2, gy1, gy2 = planner_pose_inputs_row[:7]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # Convert agent position from meters to cells in local window
        row_idx = start_y_m / map_resolution  # global row index
        col_idx = start_x_m / map_resolution  # global col index
        agent_row = int(row_idx - gx1)  # local row
        agent_col = int(col_idx - gy1)  # local col
        agent_yaw_deg = float(start_o_deg)
    elif agent_mask is not None and agent_mask.any():
        # Fallback: use agent_mask centroid
        ys, xs = np.nonzero(agent_mask)
        agent_row = int(ys.mean())
        agent_col = int(xs.mean())
        if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 3:
            agent_yaw_deg = float(planner_pose_inputs_row[2])
    else:
        # Last resort: center of map
        agent_row = H // 2
        agent_col = W // 2

    # --- Rotation: rotate map so agent heading points upward ---
    rotation_angle = 0.0
    if rotate_to_heading and agent_row is not None:
        # Calculate rotation angle to make agent point upward (negative Y in image coords)
        # agent_yaw_deg=0 means facing +X (right), we want it to point up (-Y)
        # So we need to rotate by: 90 - agent_yaw_deg
        rotation_angle = 90.0 - agent_yaw_deg

        import cv2
        # Get rotation matrix around agent position
        rot_mat = cv2.getRotationMatrix2D((agent_col, agent_row), rotation_angle, 1.0)

        # Rotate all map layers
        occ = cv2.warpAffine(occ, rot_mat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0.5)
        exp = cv2.warpAffine(exp, rot_mat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0.0)
        if agent_mask is not None:
            agent_mask = cv2.warpAffine(agent_mask.astype(np.float32), rot_mat, (W, H),
                                       flags=cv2.INTER_NEAREST, borderValue=0.0) > 0.5
        if trail_mask is not None:
            trail_mask = cv2.warpAffine(trail_mask.astype(np.float32), rot_mat, (W, H),
                                       flags=cv2.INTER_NEAREST, borderValue=0.0) > 0.5

    # --- Crop: zoom in around agent ---
    if crop_radius is not None and agent_row is not None:
        r1 = max(0, agent_row - crop_radius)
        r2 = min(H, agent_row + crop_radius)
        c1 = max(0, agent_col - crop_radius)
        c2 = min(W, agent_col + crop_radius)

        occ = occ[r1:r2, c1:c2]
        exp = exp[r1:r2, c1:c2]
        if agent_mask is not None:
            agent_mask = agent_mask[r1:r2, c1:c2]
        if trail_mask is not None:
            trail_mask = trail_mask[r1:r2, c1:c2]

        # Update dimensions
        H, W = occ.shape
        # Update agent position to cropped coords
        agent_row = agent_row - r1
        agent_col = agent_col - c1

    # --- base L image ---
    if not classify_with_explored:
        # Simple grayscale: 0→white, 1→black
        panel_L = ((1.0 - occ) * 255.0).astype(np.uint8)
    else:
        explored = (exp >= exp_thr)
        occupied = (occ >= occ_thr) & explored
        free = (~occupied) & explored
        panel_L = np.full((H, W), np.uint8(np.clip(gray_unknown, 0, 255)), dtype=np.uint8)
        panel_L[free] = np.uint8(np.clip(gray_free, 0, 255))
        panel_L[occupied] = np.uint8(np.clip(gray_occ, 0, 255))

    # Optional flip so +y is up in the final image
    if flip_y_up:
        panel_L = np.flipud(panel_L)

    # --- convert to RGBA for overlays ---
    panel = Image.fromarray(panel_L, mode="L").convert("RGBA")
    panel.putalpha(255)

    # --- trail overlay ---
    if trail_mask is not None and trail_mask.any():
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        
        if key_frame is not None and colors is not None and time_step_map is not None:
            # Multi-color logic
            ts_map = np.flipud(time_step_map) if flip_y_up else time_step_map
            
            # Determine max step in the map to bound the segments
            # Ignore -1 values (unvisited/background)
            valid_steps = ts_map[ts_map >= 0]
            max_step = int(np.max(valid_steps)) if valid_steps.size > 0 else 0
            
            # Filter key frames to be within range
            valid_key_frames = sorted([k for k in key_frame if k < max_step])
            boundaries = [0] + valid_key_frames + [max_step]
            boundaries = sorted(list(set(boundaries)))
            
            num_segments = len(boundaries) - 1
            
            for i in range(num_segments):
                start = boundaries[i]
                end = boundaries[i+1]
                
                # Mask for this segment
                if i == 0:
                    seg_mask = (ts_map >= start) & (ts_map <= end)
                else:
                    seg_mask = (ts_map > start) & (ts_map <= end)
                
                if not seg_mask.any():
                    continue
                
                # Determine color based on recency (reverse order)
                # i=num_segments-1 is the most recent -> index 0
                recency_idx = (num_segments - 1) - i
                
                if recency_idx < 5:
                    # Use provided colors if available, else default to black or cycle
                    if recency_idx < len(colors):
                        seg_color = colors[recency_idx]
                    else:
                        seg_color = (0, 0, 0)
                else:
                    seg_color = (0, 0, 0) # Black for older segments
                
                # Erode if needed (reuse trail_erode_ksize)
                if trail_erode_ksize and trail_erode_ksize > 1:
                    try:
                        import cv2
                        k = np.ones((trail_erode_ksize, trail_erode_ksize), np.uint8)
                        sm_u8 = (seg_mask.astype(np.uint8) * 255)
                        sm_u8 = cv2.erode(sm_u8, k, iterations=1)
                        seg_mask = sm_u8 > 0
                    except Exception:
                        pass

                # Draw this segment
                seg_overlay = Image.new("RGBA", (W, H), seg_color + (0,))
                seg_alpha = np.zeros((H, W), dtype=np.uint8)
                seg_alpha[seg_mask] = np.uint8(np.clip(trail_alpha, 0, 255))
                seg_overlay.putalpha(Image.fromarray(seg_alpha, mode="L"))
                
                overlay = Image.alpha_composite(overlay, seg_overlay)

        else:
            tm = np.flipud(trail_mask) if flip_y_up else trail_mask

            # 形态学腐蚀，减小轨迹粗细
            if trail_erode_ksize and trail_erode_ksize > 1:
                try:
                    import cv2
                    k = np.ones((trail_erode_ksize, trail_erode_ksize), np.uint8)
                    tm_u8 = (tm.astype(np.uint8) * 255)
                    tm_u8 = cv2.erode(tm_u8, k, iterations=1)
                    tm = tm_u8 > 0
                except Exception:
                    # 没有 cv2 就退化为简单的子采样变细
                    from scipy.ndimage import minimum_filter  # 如无scipy仍会报错，按需装
                    tm = minimum_filter(tm.astype(np.uint8), size=trail_erode_ksize) > 0

            overlay = Image.new("RGBA", (W, H), trail_color + (0,))
            alpha = np.zeros((H, W), dtype=np.uint8)
            alpha[tm] = np.uint8(np.clip(trail_alpha, 0, 255))
            overlay.putalpha(Image.fromarray(alpha, mode="L"))
            
        panel = Image.alpha_composite(panel, overlay)

    # Resize before drawing vector graphics if requested
    if output_size is not None:
        sx = sy = output_size
        panel = panel.resize((sx, sy), resample=Image.NEAREST)
    else:
        sx, sy = W, H

    # --- arrow drawing ---
    draw = ImageDraw.Draw(panel)

    # After rotation and crop, use agent position in the transformed map
    x_pix = agent_col * (sx / W) if agent_col is not None else sx / 2.0
    y_pix = ((H - 1 - agent_row) if flip_y_up else agent_row) * (sy / H) if agent_row is not None else sy / 2.0

    # If we rotated, the agent now always points upward (yaw=90 in image coords)
    # If flip_y_up, upward means -90 degrees (pointing towards decreasing y)
    if rotate_to_heading:
        yaw_deg = -90.0 if flip_y_up else 90.0
    else:
        # Original logic: extract yaw from planner_pose_inputs
        if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 3:
            yaw_deg = float(planner_pose_inputs_row[2])
            yaw_deg = -yaw_deg if flip_y_up else yaw_deg
        else:
            yaw_deg = 0.0

    # --- draw bigger arrow ---
    theta = math.radians(yaw_deg)
    dx, dy = math.cos(theta) * arrow_len_px, math.sin(theta) * arrow_len_px
    x1, y1 = float(x_pix), float(y_pix)
    x2, y2 = x1 + dx, y1 + dy

    vlen = math.hypot(dx, dy) or 1.0
    ux, uy = dx / vlen, dy / vlen
    bx, by = x2 - ux * head_length, y2 - uy * head_length

    draw.line([(x1, y1), (bx, by)], fill=arrow_color, width=int(arrow_width))
    px, py = -uy, ux
    hx1, hy1 = bx + px * (head_width / 2.0), by + py * (head_width / 2.0)
    hx2, hy2 = bx - px * (head_width / 2.0), by - py * (head_width / 2.0)
    draw.polygon([(x2, y2), (hx1, hy1), (hx2, hy2)], fill=arrow_color)

    # center marker
    if agent_mask is not None and agent_mask.any():
        ys, xs = np.nonzero(agent_mask)
        r0 = (H - 1 - ys.mean()) if flip_y_up else ys.mean()
        c0 = xs.mean()
        cx = c0 * (sx / W)
        cy = r0 * (sy / H)
        r = int(mark_radius)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=arrow_color, width=2)

    return panel.convert("RGB")


# --- thin aliases for your existing names (optional) ---

def plot_local_occupancy_with_pose_pil(
    local_map, planner_pose_inputs_row, map_resolution,
    crop_radius=150, output_size=480, **kwargs
):
    # Simple view = original first function
    # Allow overriding defaults from kwargs if not explicitly provided
    return plot_local_bev_pil(
        local_map,
        planner_pose_inputs_row,
        classify_with_explored=False,
        map_resolution=map_resolution,
        output_size=output_size,
        rotate_to_heading=False,  # No rotation - keep original orientation
        crop_radius=crop_radius,
        **kwargs,
    )

def plot_local_occupency_explore_with_pose_pil(
    local_map, planner_pose_inputs_row, map_resolution,
    crop_radius=150, output_size=480, **kwargs
):
    # Exploration-aware view = original second function, but returns PIL
    return plot_local_bev_pil(
        local_map,
        planner_pose_inputs_row,
        classify_with_explored=True,
        map_resolution=map_resolution,
        output_size=output_size,
        rotate_to_heading=False,  # No rotation - keep original orientation
        crop_radius=crop_radius,
        **kwargs,
    )

# --- add this helper ---
def _to_label_index(idx0: int, label_start: int) -> int:
    return idx0 + (label_start or 0)

# add frontier points on this occ_exp_pil
def overlay_frontier_centroids_on_pil(
        pil_img: Image.Image,
        frontier_u8: np.ndarray,
        centroids,  # [(row, col), ...] in BEV cell coords (ORIGINAL map space)
        *,
        flip_y_up=True,  # True if BEV did np.flipud
        dot_radius=4,
        color=(0, 255, 0), # green, frontier point color
        outline=(255, 255, 255),
        width=2,
        enumerate_points=True,  # whether to draw frontier point indices
        label_start: int = 1,  # <--- NEW: 1 means labels 1..N
        font_size: int = 16,
        font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # --- NEW: transformation parameters for rotated/cropped maps ---
        rotation_angle: float = 0.0,  # degrees (rotation applied to map)
        agent_row: Optional[int] = None,  # agent position in original map (before rotation/crop)
        agent_col: Optional[int] = None,
        crop_r1: int = 0,  # crop bounds in original map
        crop_r2: Optional[int] = None,
        crop_c1: int = 0,
        crop_c2: Optional[int] = None,
):
    """
    Overlay frontier points on a PIL image of the BEV map.
    Handles rotated and cropped maps by transforming centroid coordinates.
    """
    H_orig, W_orig = frontier_u8.shape  # Original BEV map-grid size
    sx, sy = pil_img.size  # Output image size

    # Determine actual displayed map size after crop
    if crop_r2 is None:
        crop_r2 = H_orig
    if crop_c2 is None:
        crop_c2 = W_orig
    H_displayed = crop_r2 - crop_r1
    W_displayed = crop_c2 - crop_c1

    draw = ImageDraw.Draw(pil_img)

    # load font safely
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    import cv2
    for i0, (r, c) in enumerate(centroids):   # i0 is 0-based, (r,c) in original map coords
        # Transform centroid coordinates to match rotated/cropped map
        r_transformed = r
        c_transformed = c

        # 1. Apply rotation if needed
        if rotation_angle != 0.0 and agent_row is not None and agent_col is not None:
            # Get rotation matrix (same as used in plot_local_bev_pil)
            rot_mat = cv2.getRotationMatrix2D((agent_col, agent_row), rotation_angle, 1.0)
            # Apply rotation to this point
            point = np.array([[c, r]], dtype=np.float32).reshape(1, 1, 2)
            rotated = cv2.transform(point, rot_mat)
            c_transformed = rotated[0, 0, 0]
            r_transformed = rotated[0, 0, 1]

        # 2. Apply crop offset
        r_transformed -= crop_r1
        c_transformed -= crop_c1

        # 3. Check if point is within cropped bounds
        if not (0 <= r_transformed < H_displayed and 0 <= c_transformed < W_displayed):
            continue  # Skip points outside visible area

        # 4. Convert to pixel coordinates in displayed image
        if flip_y_up:
            y = (H_displayed - 1 - r_transformed) * (sy / H_displayed)
        else:
            y = r_transformed * (sy / H_displayed)
        x = c_transformed * (sx / W_displayed)

        # draw frontier point (circle with outline)
        bbox = [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)]
        draw.ellipse(bbox, outline=outline, width=width)
        draw.ellipse([(x - dot_radius + 1, y - dot_radius + 1),
                      (x + dot_radius - 1, y + dot_radius - 1)],
                     outline=color, width=width,
                     fill=color)

        # optional: numerate
        if enumerate_points:
            label = _to_label_index(i0, label_start)

            # draw "bold" by outlining with white, then filling with color
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((x + dot_radius + 2 + dx, y - dot_radius - 2 + dy),
                          str(label), font=font, fill=outline)

            draw.text((x + dot_radius + 2, y - dot_radius - 2),
                      str(label), font=font, fill=color)

    return pil_img


def overlay_frontier_centroids_on_rotated_map(
    pil_img: Image.Image,
    local_map,
    planner_pose_inputs_row,
    map_resolution: float,
    frontier_u8: np.ndarray,
    centroids,
    *,
    rotate_to_heading: bool = True,
    crop_radius: Optional[int] = None,
    **overlay_kwargs
) -> Image.Image:
    """
    Wrapper function that handles coordinate transformation for rotated/cropped maps.
    Automatically extracts transformation parameters and passes them to overlay_frontier_centroids_on_pil.
    """
    # Extract agent position and yaw (same logic as in plot_local_bev_pil)
    lm = local_map.detach().cpu().numpy() if isinstance(local_map, torch.Tensor) else np.asarray(local_map)
    H, W, C = lm.shape

    agent_mask = (lm[:, :, 2] > 0.5) if C >= 3 else None

    agent_row, agent_col, agent_yaw_deg = None, None, 0.0
    if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 7 and map_resolution is not None:
        start_x_m, start_y_m, start_o_deg, gx1, gx2, gy1, gy2 = planner_pose_inputs_row[:7]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        row_idx = start_y_m / map_resolution
        col_idx = start_x_m / map_resolution
        agent_row = int(row_idx - gx1)
        agent_col = int(col_idx - gy1)
        agent_yaw_deg = float(start_o_deg)
    elif agent_mask is not None and agent_mask.any():
        ys, xs = np.nonzero(agent_mask)
        agent_row = int(ys.mean())
        agent_col = int(xs.mean())
        if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 3:
            agent_yaw_deg = float(planner_pose_inputs_row[2])
    else:
        agent_row = H // 2
        agent_col = W // 2

    # Calculate rotation angle
    rotation_angle = 0.0
    if rotate_to_heading and agent_row is not None:
        rotation_angle = 90.0 - agent_yaw_deg

    # Calculate crop bounds
    crop_r1, crop_r2 = 0, H
    crop_c1, crop_c2 = 0, W
    if crop_radius is not None and agent_row is not None:
        crop_r1 = max(0, agent_row - crop_radius)
        crop_r2 = min(H, agent_row + crop_radius)
        crop_c1 = max(0, agent_col - crop_radius)
        crop_c2 = min(W, agent_col + crop_radius)

    # Call the main overlay function with transformation parameters
    return overlay_frontier_centroids_on_pil(
        pil_img,
        frontier_u8,
        centroids,
        rotation_angle=rotation_angle,
        agent_row=agent_row,
        agent_col=agent_col,
        crop_r1=crop_r1,
        crop_r2=crop_r2,
        crop_c1=crop_c1,
        crop_c2=crop_c2,
        **overlay_kwargs
    )


# ------------------------------
# Utilities
# ------------------------------
def _wrap_angle_rad(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

def _pose_pixels_from_inputs(
    planner_pose_inputs_row: Optional[Tuple[float, ...]],
    *,
    panel_size: Tuple[int, int],
    bev_hw_cells: Tuple[int, int],
    map_resolution: Optional[float],
    flip_y_up: bool
) -> Tuple[float, float, float]:
    """
    Returns (x_pix, y_pix, yaw_deg_display) for the arrow tip origin on the already-rendered panel.
    Matches your plot_local_bev_pil() convention, including flip_y_up behavior.
    """
    sx, sy = panel_size
    H, W = bev_hw_cells

    x_pix = y_pix = None
    yaw_deg = 0.0

    if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 7 and map_resolution is not None:

        start_x_m, start_y_m, start_o_deg, gx1, gx2, gy1, gy2 = planner_pose_inputs_row[:7]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        H_win, W_win = (gx2 - gx1), (gy2 - gy1)   # rows, cols in CELLS

        # meters -> cells (global indices), map_resolution is already in m/cell
        row_idx = start_y_m / map_resolution
        col_idx = start_x_m / map_resolution

        # local window coordinates
        row_rel = row_idx - gx1
        col_rel = col_idx - gy1

        # → pixel on panel
        x_pix = col_rel * (sx / W_win)
        y_pix = (H_win - row_rel) * (sy / H_win) if flip_y_up else row_rel * (sy / H_win)
        yaw_deg = float(-start_o_deg) if flip_y_up else float(start_o_deg)

    return float(x_pix), float(y_pix), float(yaw_deg)

def _panelpix_to_bev_rc(
    x_pix: float, y_pix: float,
    panel_size: Tuple[int, int],
    bev_hw_cells: Tuple[int, int],
    *,
    flip_y_up: bool
) -> Tuple[float, float]:
    """
    Convert panel pixel coords back to BEV cell index space (row, col).
    Note: uses linear scaling consistent with your rendering path.
    """
    sx, sy = panel_size
    H, W = bev_hw_cells
    col = (x_pix / sx) * W
    if flip_y_up:
        row = (H - 1) - (y_pix / sy) * H
    else:
        row = (y_pix / sy) * H
    return float(row), float(col)

# ------------------------------
# Frontier selection
# ------------------------------
def select_frontier(
    centroids: List[Tuple[float, float]],   # [(row, col), ...] in BEV cells
    panel_size: Tuple[int, int],            # (sx, sy) of the rendered panel
    bev_hw_cells: Tuple[int, int],          # (H, W) in BEV cells
    planner_pose_inputs_row: Tuple[float, ...],
    *,
    map_resolution: Optional[float],
    flip_y_up: bool = True,
    mode: str = "hybrid",                   # "nearest" | "heading" | "hybrid"
    # hybrid scoring params:
    w_theta: float = 0.7,                   # weight for angle (0..1)
    w_dist: float = 0.3,                    # weight for distance (0..1), w_theta + w_dist = 1
    dist_norm: Optional[float] = None,      # if None, uses max(H, W)
    angle_cone_deg: Optional[float] = None  # optional hard gate; only consider frontiers within ±cone
) -> Dict:
    """
    Returns a dict with:
      {
        "index": int (0-based),
        "mode": str,
        "robot_rc": (row, col),
        "yaw_rad": float,
        "scores": List[float],    # per frontier (distance, angle or hybrid score depending on mode)
        "angles_abs_deg": List[float],
        "dists": List[float]
      }
    """
    assert len(centroids) > 0, "No frontier centroids provided."

    sx, sy = panel_size
    H, W = bev_hw_cells

    # --- robot pixel & yaw in display axes (matches your arrow draw) ---
    rx, ry, yaw_deg_disp = _pose_pixels_from_inputs(
        planner_pose_inputs_row,
        panel_size=panel_size,
        bev_hw_cells=bev_hw_cells,
        map_resolution=map_resolution,
        flip_y_up=flip_y_up
    )
    yaw_rad = math.radians(yaw_deg_disp)
    heading_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)

    # helper: BEV (r,c) -> panel (x,y) pixels
    def rc_to_xy(r, c):
        x = c * (sx / W)
        y = (H - 1 - r) * (sy / H) if flip_y_up else r * (sy / H)
        return float(x), float(y)

    # --- per-frontier scores in *pixel space* ---
    dists_px, angles_abs_deg, fxy = [], [], []
    for (rf, cf) in centroids:
        fx, fy = rc_to_xy(rf, cf)
        fxy.append((fx, fy))

        dx, dy = (fx - rx), (fy - ry)
        dist = math.hypot(dx, dy)
        dists_px.append(dist)

        v = np.array([dx, dy], dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1e-6:
            ang_deg = 0.0
        else:
            dot = float(np.clip(np.dot(heading_vec, v / n), -1.0, 1.0))
            ang_deg = math.degrees(math.acos(dot))   # [0, 180]
        angles_abs_deg.append(ang_deg)

    # --- optional FOV gating (treat input as FULL FOV) ---
    valid = np.ones(len(centroids), dtype=bool)
    if angle_cone_deg is not None:
        half = abs(float(angle_cone_deg)) * 0.5    # ±half-angle
        valid = np.array([a <= half for a in angles_abs_deg], dtype=bool)

    # --- scoring ---
    if mode == "nearest":
        scores = [d if v else float("inf") for d, v in zip(dists_px, valid)]
    elif mode == "heading":
        scores = [a if v else float("inf") for a, v in zip(angles_abs_deg, valid)]
    else:  # hybrid in pixel space
        diag = math.hypot(sx, sy)
        scores = [w_dist*(d/diag) + w_theta*(a/180.0) if v else float("inf")
                  for d, a, v in zip(dists_px, angles_abs_deg, valid)]

    if all([s == float("inf") for s in scores]):
        # fallback to nearest frontier ignoring FOV if FOV gate zeroes-out everything so every score = in
        scores = dists_px
        best_idx = int(np.argmin(scores))
    else:
        best_idx = int(np.argmin(scores))

    # also report robot rc if you need it downstream
    r_robot, c_robot = _panelpix_to_bev_rc(rx, ry, panel_size, bev_hw_cells, flip_y_up=flip_y_up)

    return {
        "index": best_idx,                 # 0-based index (for code)
        "mode": mode,
        "robot_rc": (r_robot, c_robot),
        "yaw_rad": yaw_rad,
        "scores": scores,
        "angles_abs_deg": angles_abs_deg,  # angle between heading and frontier vector (deg)
        "dists": dists_px,                 # pixel distances (WYSIWYG)
        "robot_xy": (rx, ry),
        "frontier_xy": fxy
    }

# ------------------------------
# Visualization
# ------------------------------
def overlay_selected_frontier(
    pil_img: Image.Image,
    centroids: List[Tuple[float, float]],
    selection: Dict,
    *,
    bev_hw_cells: Tuple[int, int],
    flip_y_up: bool = True,
    highlight_color: Tuple[int, int, int] = (255, 215, 0),  # gold
    line_color: Tuple[int, int, int] = (255, 215, 0),
    line_width: int = 3,
    ring_radius: int = 7,
    show_index: bool = True,
    index_fill: Tuple[int, int, int] = (0, 0, 0),
    label_start: int = 1,  # <--- NEW

) -> Image.Image:
    """
    Draws a line from robot → selected frontier, and highlights the chosen frontier with a gold ring.
    """
    sx, sy = pil_img.size
    H, W = bev_hw_cells
    draw = ImageDraw.Draw(pil_img)

    # robot pixel from selection.robot_rc
    r_robot, c_robot = selection["robot_rc"]
    if flip_y_up:
        y_robot = (H - 1 - r_robot) * (sy / H)
    else:
        y_robot = r_robot * (sy / H)
    x_robot = c_robot * (sx / W)

    k0 = selection["index"]       # 0-based
    rf, cf = centroids[k0]
    y_f = (H - 1 - rf) * (sy / H) if flip_y_up else rf * (sy / H)
    x_f = cf * (sx / W)

    # line robot -> frontier
    draw.line([(x_robot, y_robot), (x_f, y_f)], fill=line_color, width=line_width)
    draw.ellipse([(x_f - ring_radius, y_f - ring_radius),
                  (x_f + ring_radius, y_f + ring_radius)],
                 outline=highlight_color, width=line_width)

    if show_index:
        label = _to_label_index(k0, label_start)
        draw.text((x_f + ring_radius + 2, y_f - ring_radius - 2), str(label), fill=index_fill)
    return pil_img

# ------------------------------
# Convenience wrapper
# ------------------------------
def select_and_visualize_frontier(
    base_panel: Image.Image,                 # output from plot_local_bev_pil(...)
    centroids: List[Tuple[float, float]],    # [(row, col), ...] in BEV cells
    planner_pose_inputs_row: Tuple[float, ...],
    *,
    bev_hw_cells: Tuple[int, int],
    map_resolution: Optional[float],
    flip_y_up: bool = True,
    mode: str = "",                   # "nearest" | "heading" | "hybrid"
    w_theta: float = 0.7,
    w_dist: float = 0.3,
    angle_cone_deg: Optional[float] = None,
    label_start: int = 1,
    # Styling parameters for overlay_selected_frontier
    highlight_color: Tuple[int, int, int] = (255, 215, 0),
    line_color: Tuple[int, int, int] = (255, 215, 0),
    line_width: int = 3,
    ring_radius: int = 7,
    show_index: bool = True,
    index_fill: Tuple[int, int, int] = (0, 0, 0),
) -> tuple[Any, int, Image, dict]: # type: ignore
    """
    Select a frontier (nearest/heading/hybrid), draw a gold link & ring on panel, and return:
      (best_index_0based, updated_panel, selection_info)
    """
    sx, sy = base_panel.size
    selection = select_frontier(
        centroids=centroids,
        panel_size=(sx, sy),
        bev_hw_cells=bev_hw_cells,
        planner_pose_inputs_row=planner_pose_inputs_row,
        map_resolution=map_resolution,
        flip_y_up=flip_y_up,
        mode=mode,
        w_theta=w_theta,
        w_dist=w_dist,
        angle_cone_deg=angle_cone_deg
    )
    out = overlay_selected_frontier(
        base_panel.copy(), centroids, selection,
        bev_hw_cells=bev_hw_cells, flip_y_up=flip_y_up,
        label_start=label_start,
        highlight_color=highlight_color,
        line_color=line_color,
        line_width=line_width,
        ring_radius=ring_radius,
        show_index=show_index,
        index_fill=index_fill,
    )

    best_idx0 = selection["index"]
    best_label = _to_label_index(best_idx0, label_start)

    return best_idx0, best_label, out, selection


def draw_fov_on_bev(
    local_map,
    planner_pose_inputs_row,
    map_resolution,
    *,
    crop_radius=150,
    output_size=480,
    fov_angle_deg=90,  # Field of view angle (e.g., 90 degrees)
    fov_range_cells=100,  # Max FOV distance in cells
    fov_color=(255, 255, 0, 100),  # Yellow with transparency (RGBA)
    **kwargs
) -> Image.Image:
    """
    Create a BEV visualization with FOV cone overlay.

    Args:
        local_map: Local map tensor
        planner_pose_inputs_row: Agent pose [x, y, yaw, gx1, gx2, gy1, gy2]
        map_resolution: Resolution in meters/cell
        fov_angle_deg: FOV cone angle in degrees
        fov_range_cells: FOV maximum range in cells
        fov_color: RGBA color for FOV cone overlay

    Returns:
        PIL Image with FOV cone drawn on BEV map
    """
    # Start with occupancy + exploration view
    base_pil = plot_local_occupency_explore_with_pose_pil(
        local_map,
        planner_pose_inputs_row,
        map_resolution,
        crop_radius=crop_radius,
        output_size=output_size,
        **kwargs
    )

    # Extract local map to get actual dimensions after crop
    lm = local_map.detach().cpu().numpy() if isinstance(local_map, torch.Tensor) else np.asarray(local_map)
    H_full, W_full, C = lm.shape

    # Extract agent position and heading
    if planner_pose_inputs_row is not None and len(planner_pose_inputs_row) >= 7:
        start_x_m, start_y_m, start_o_deg, gx1, gx2, gy1, gy2 = planner_pose_inputs_row[:7]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # Agent position in cells (local window coordinates before crop)
        row_idx = start_y_m / map_resolution
        col_idx = start_x_m / map_resolution
        agent_row = int(row_idx - gx1)
        agent_col = int(col_idx - gy1)

        # Calculate crop bounds (same as in plot_local_bev_pil)
        H_win, W_win = (gx2 - gx1), (gy2 - gy1)
        r1 = max(0, agent_row - crop_radius)
        r2 = min(H_win, agent_row + crop_radius)
        c1 = max(0, agent_col - crop_radius)
        c2 = min(W_win, agent_col + crop_radius)

        # Actual cropped dimensions in cells
        H = r2 - r1
        W = c2 - c1

        # Agent position in cropped coords (same calculation as in plot_local_bev_pil line 274-275)
        agent_row = agent_row - r1
        agent_col = agent_col - c1

        # After resize to output_size (same as arrow drawing logic line 323-326, 332-333)
        sx = sy = output_size
        x_pix = agent_col * (sx / W)
        y_pix = (H - 1 - agent_row) * (sy / H)  # flip_y_up=True

        # Yaw angle in degrees (same as arrow drawing)
        yaw_deg = float(start_o_deg)
        yaw_deg = -yaw_deg  # Negative for flip_y_up (same as line 342)
        theta = math.radians(yaw_deg)  # Convert to radians for pixel-space drawing

        # Draw FOV cone
        overlay = Image.new("RGBA", base_pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Calculate FOV cone based on actual depth sensor projection
        # fov_range_cells is the sensor's max_depth in cells
        # Use the full sensor range for accurate visualization
        fov_range_cells_capped = fov_range_cells

        # Calculate lateral extent at max depth using camera geometry
        # For HFOV angle, lateral extent = tan(HFOV/2) * depth
        half_fov_rad = math.radians(fov_angle_deg / 2.0)
        lateral_extent_cells = math.tan(half_fov_rad) * fov_range_cells_capped

        # Convert from grid cells to output pixels
        # Scale: cells_in_cropped_space * (output_pixels / cropped_cells)
        fov_forward_pix = fov_range_cells_capped * (sy / H)
        fov_lateral_pix = lateral_extent_cells * (sx / W)

        # Calculate FOV edges in pixel space (same approach as arrow drawing)
        # Forward direction in agent's frame
        forward_dx = math.cos(theta) * fov_forward_pix
        forward_dy = math.sin(theta) * fov_forward_pix

        # Lateral direction (perpendicular to forward)
        lateral_dx = -math.sin(theta) * fov_lateral_pix  # perpendicular
        lateral_dy = math.cos(theta) * fov_lateral_pix

        # Left corner: forward + left lateral
        x_left = x_pix + forward_dx - lateral_dx
        y_left = y_pix + forward_dy - lateral_dy

        # Right corner: forward + right lateral
        x_right = x_pix + forward_dx + lateral_dx
        y_right = y_pix + forward_dy + lateral_dy

        # Draw filled polygon for FOV cone
        fov_poly = [(x_pix, y_pix), (x_left, y_left), (x_right, y_right)]
        draw.polygon(fov_poly, fill=fov_color)

        # Composite overlay onto base image
        base_rgba = base_pil.convert("RGBA")
        combined = Image.alpha_composite(base_rgba, overlay)
        return combined.convert("RGB")

    return base_pil

