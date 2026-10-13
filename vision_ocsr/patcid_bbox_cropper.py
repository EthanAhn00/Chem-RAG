import os
from PIL import Image

# Inactive: OCSRPipeline.ENABLE_BBOX_CROP_FALLBACK=False prevents calls.


def crop_by_patcid_bbox(image_path, bbox_inches, page_width_in, page_height_in, output_path, padding_frac=0.08):
    """Crop chemical structures using PatCID bbox coordinates in inches [x0,y0,x1,y1]. Unlike generic contour detection in chem_vision_cropper.py, use locations identified by the trained IBM PatCID detector rather than unrelated text/arrows/shapes. Return output_path on success or None on failure."""
    if not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size

        # Convert inches to pixels using actual page DPI separately for x/y,
        # supporting scans with different axis scales.
        dpi_x = img_w / page_width_in if page_width_in else img_w / 8.5
        dpi_y = img_h / page_height_in if page_height_in else img_h / 11.0

        x0, y0, x1, y1 = bbox_inches
        px0, py0, px1, py1 = x0 * dpi_x, y0 * dpi_y, x1 * dpi_x, y1 * dpi_y

        # Add proportional padding because boxes may be tight.
        w, h = px1 - px0, py1 - py0
        pad_x, pad_y = w * padding_frac, h * padding_frac
        left = max(0, int(px0 - pad_x))
        top = max(0, int(py0 - pad_y))
        right = min(img_w, int(px1 + pad_x))
        bottom = min(img_h, int(py1 + pad_y))

        if right <= left or bottom <= top:
            return None

        crop = img.crop((left, top, right, bottom))
        crop.save(output_path)
        return output_path
    except Exception:
        return None
