import os
from typing import List

import cv2
import numpy as np

from segmentation.grouping import BrailleCell


def crop_and_save_cells(
    binary_image: np.ndarray, cells: List[BrailleCell], output_dir: str
):
    target_size = (64, 64)
    h_img, w_img = binary_image.shape

    for cell in cells:
        x, y, w, h = cell.bbox

        # Ensure bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        crop = binary_image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Resize preserving aspect ratio by padding
        h_crop, w_crop = crop.shape
        scale = min(target_size[0] / w_crop, target_size[1] / h_crop)

        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)

        resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad to target size
        pad_w = target_size[0] - new_w
        pad_h = target_size[1] - new_h

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        final_crop = cv2.copyMakeBorder(
            resized_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

        cell.image = final_crop

        filename = os.path.join(output_dir, f"cell_{cell.order_index:03d}.png")
        cv2.imwrite(filename, final_crop)
