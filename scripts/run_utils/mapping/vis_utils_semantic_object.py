import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union, List


from scripts.run_utils.constant import mp3d_category
from habitat_sim.utils.common import d3_40_colors_rgb



# --- legend building (compact, top-right overlay) ---
def make_legend(categories, id_to_name, palette_flat,
                swatch=(18,18), pad=6, label_px_max=140, bg=(255,255,255,210)) -> Image.Image:
    if not categories:
        return Image.new("RGBA", (1,1), (0,0,0,0))
    try:
        font = ImageFont.load_default()  # compact default font
    except Exception:
        font = None

    # Measure text widths to compute a compact legend width
    dummy = Image.new("RGBA", (1,1))
    meas = ImageDraw.Draw(dummy)
    def text_w(s):  # robust width measurement
        if hasattr(meas, "textbbox"):
            return meas.textbbox((0,0), s, font=font)[2]
        return int(meas.textlength(s, font=font))  # fallback

    # Optional: ellipsize text to fit within max px
    def ellipsize_to_px(s, max_px):
        if text_w(s) <= max_px:
            return s
        base = s
        # keep trimming until "…"-appended fits
        while base and text_w(base + "…") > max_px:
            base = base[:-1]
        return base + "…" if base else "…"

    labels = [f"{int(cid)}: {name_for(id_to_name, cid)}" for cid in categories]

    # First compute the natural max width, then clamp to label_px_max
    natural_max = max((text_w(lbl) for lbl in labels), default=0)
    label_w = min(natural_max, label_px_max)
    # Create ellipsized labels that fit the clamp
    labels = [ellipsize_to_px(lbl, label_w) for lbl in labels]

    sw_w, sw_h = swatch
    row_h = max(sw_h, 16) + pad
    width  = pad + sw_w + pad + label_w + pad
    height = pad + len(categories) * row_h
    legend = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(legend)

    y = pad
    for cid, lbl in zip(categories, labels):
        color = color_from_palette(palette_flat, cid)
        draw.rectangle([pad, y, pad+sw_w, y+sw_h], fill=color, outline=(0,0,0,255))
        draw.text((pad+sw_w+pad, y), lbl, fill=(0,0,0,255), font=font)
        y += row_h
    return legend

def color_from_palette(palette_flat: list, idx: int, base: int = 40) -> tuple:
    j = (int(idx) % base) * 3
    return tuple(palette_flat[j:j+3])  # (R,G,B)

def name_for(id_to_name: Union[dict, List[str]], cid: int) -> str:
    cid = int(cid)
    if isinstance(id_to_name, dict):
        return id_to_name.get(cid, "unknown")
    if isinstance(id_to_name, (list, tuple)):
        return id_to_name[cid] if 0 <= cid < len(id_to_name) else "unknown"
    return "unknown"

def overlay_top_right(base_rgba: Image.Image, overlay_rgba: Image.Image, margin_px: int = 8) -> Image.Image:
    out = base_rgba.copy()
    x = out.width - overlay_rgba.width - margin_px
    y = margin_px
    # paste with alpha (overlay_rgba has an alpha channel)
    out.paste(overlay_rgba, (x, y), overlay_rgba)
    return out


def palette_flat_256(pal40: np.ndarray) -> list:
    base = pal40.reshape(-1).tolist()
    return base + [0] * (256*3 - len(base))


def colorize_by_ids(mask_ids: np.ndarray, palette_flat: list, base: int = 40) -> Image.Image:
    h, w = mask_ids.shape
    pal_img = Image.new("P", (w, h))
    pal_img.putpalette(palette_flat)
    pal_img.putdata((mask_ids.flatten() % base).astype(np.uint8))
    return pal_img.convert("RGBA")



def save_semantic_annotated_from_masks(
    save_path: str,
    seg_idx: np.ndarray,          # (H,W) int ids; 0=void, 1..40 map to mp3d_category
    render_void: bool = False,           # False => make void transparent; True => paint with void_color
    void_color=(200, 200, 200, 180),     # RGBA if render_void=True
    include_void_in_legend: bool = False,# whether to show "void" in legend
    skip_ids: tuple = (),                # additional ids to hide in legend (besides auto-handling of 0 if include_void_in_legend=False)
    legend_max_label_px: int = 140,  # tighter legend width cap
    legend_swatch=(18, 18),          # smaller swatches
    legend_pad: int = 6,             # tighter padding
    legend_bg=(255, 255, 255, 210),  # semi-opaque legend background
    margin: int = 8,                 # margin from top-right corner
):
    """
    Colorize seg_idx with D3-40 palette (IDs 1..40) and add legend from mp3d_category.
    ID 0 is treated as 'void' (transparent by default, or colored if render_void=True).
    """

    # --- sanity checks ---
    assert isinstance(mp3d_category, (list, tuple)) and len(mp3d_category) == 40, \
        "mp3d_category must be a list of length 40 mapping ids 1..40."

    # --- build id -> name mapping from the fixed list ---
    ids_present = np.unique(seg_idx)
    id_to_name = {}
    for cid in ids_present:
        cid = int(cid)
        if cid == 0:
            if include_void_in_legend:
                id_to_name[0] = "void"
            continue
        if cid in skip_ids:
            continue
        if 1 <= cid <= 40:
            id_to_name[cid] = mp3d_category[cid - 1]
        else:
            # Fallback in case of out-of-range ids
            id_to_name[cid] = f"class_{cid}"

    # --- render base semantic image using D3-40 palette ---
    # palette_flat_256(d3_40_colors_rgb) should produce a 256*3 flat list where
    # indices 0..39 correspond to the D3-40 colors. That means:
    #  - id=1 uses the 2nd color, id=40 uses the 40th color.
    #  - id=0 (void) gets the first color; we'll override handling below.
    palette_flat = palette_flat_256(d3_40_colors_rgb)
    semantic_img = colorize_by_ids(seg_idx, palette_flat, base=40)  # returns a PIL.Image ("P" or "RGBA" per your impl)

    # Ensure RGBA for editing alpha/colors
    if semantic_img.mode != "RGBA":
        semantic_rgba = semantic_img.convert("RGBA")
    else:
        semantic_rgba = semantic_img

    # --- handle void pixels (id==0) ---
    # If render_void is False => make void transparent; else paint with void_color
    void_mask = (seg_idx == 0)
    if void_mask.any():
        arr = np.array(semantic_rgba, dtype=np.uint8)  # (H,W,4)
        if render_void:
            vr, vg, vb, va = void_color
            arr[void_mask] = np.array([vr, vg, vb, va], dtype=np.uint8)
        else:
            # transparent
            arr[void_mask, 3] = 0  # set alpha to 0
        semantic_rgba = Image.fromarray(arr, mode="RGBA")

    # --- legend (omit void unless include_void_in_legend=True) ---
    cats_for_legend = [int(c) for c in ids_present
                       if (c != 0 or include_void_in_legend)
                       and int(c) not in skip_ids]

    legend_img = make_legend(
        cats_for_legend,
        id_to_name,
        palette_flat,
        swatch=legend_swatch,
        pad=legend_pad,
        label_px_max=legend_max_label_px,
        bg=legend_bg,
        # If your make_legend supports custom color for id 0, it will use void_color automatically
        # when include_void_in_legend is True. If not, it will pull the color from palette index 0.
        # (You can modify make_legend to special-case id==0 to use void_color if desired.)
    )

    vis = overlay_top_right(semantic_rgba, legend_img, margin_px=margin)
    vis.save(save_path)
    return vis


def save_semantic_annotated_from_masks_room(
    save_path: str,
    seg_idx: np.ndarray,          # (H,W) int ids; 0=
    seg_name: str,                # room name
    render_void: bool = False,  # False => make void transparent; True => paint with void_color
    void_color=(200, 200, 200, 180),  # RGBA if render_void=True
    include_void_in_legend: bool = False,  # whether to show "void" in legend
    skip_ids: tuple = (),
    # additional ids to hide in legend (besides auto-handling of 0 if include_void_in_legend=False)
    legend_max_label_px: int = 140,  # tighter legend width cap
    legend_swatch=(18, 18),  # smaller swatches
    legend_pad: int = 6,  # tighter padding
    legend_bg=(255, 255, 255, 210),  # semi-opaque legend background
    margin: int = 8,  # margin from top-right corner

):
    seg_idx_int = seg_idx.astype(int).flatten()
    seg_name_flat = seg_name.flatten()

    # Build dictionary
    id_to_name = dict(zip(seg_idx_int, seg_name_flat))

    ids_present = np.unique(seg_idx)

    palette_flat = palette_flat_256(d3_40_colors_rgb)
    semantic_img = colorize_by_ids(seg_idx, palette_flat, base=40)  # returns a PIL.Image ("P" or "RGBA" per your impl)

    # Ensure RGBA for editing alpha/colors
    if semantic_img.mode != "RGBA":
        semantic_rgba = semantic_img.convert("RGBA")
    else:
        semantic_rgba = semantic_img

    # --- handle void pixels (id==0) ---
    # If render_void is False => make void transparent; else paint with void_color
    void_mask = (seg_idx == 0)
    if void_mask.any():
        arr = np.array(semantic_rgba, dtype=np.uint8)  # (H,W,4)
        if render_void:
            vr, vg, vb, va = void_color
            arr[void_mask] = np.array([vr, vg, vb, va], dtype=np.uint8)
        else:
            # transparent
            arr[void_mask, 3] = 0  # set alpha to 0
        semantic_rgba = Image.fromarray(arr, mode="RGBA")

    # --- legend (omit void unless include_void_in_legend=True) ---
    cats_for_legend = [int(c) for c in ids_present
                       if (c != 0 or include_void_in_legend)
                       and int(c) not in skip_ids]

    legend_img = make_legend(
        cats_for_legend,
        id_to_name,
        palette_flat,
        swatch=legend_swatch,
        pad=legend_pad,
        label_px_max=legend_max_label_px,
        bg=legend_bg,
        # If your make_legend supports custom color for id 0, it will use void_color automatically
        # when include_void_in_legend is True. If not, it will pull the color from palette index 0.
        # (You can modify make_legend to special-case id==0 to use void_color if desired.)
    )

    vis = overlay_top_right(semantic_rgba, legend_img, margin_px=margin)
    vis.save(save_path)
    return vis
