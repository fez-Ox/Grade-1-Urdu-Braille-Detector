import os
from typing import List

import cv2
import numpy as np

from segmentation.dot_detection import Dot
from segmentation.grouping import BrailleCell


def generate_overlays(
    image: np.ndarray,
    dots: List[Dot],
    lines: List[List[Dot]],
    cells: List[BrailleCell],
    output_dir: str,
):
    # If image is grayscale, convert to BGR for color drawing
    if len(image.shape) == 2:
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image.copy()

    #  Dots Overlay
    dots_overlay = img_color.copy()
    for dot in dots:
        x, y, w, h = dot.bbox
        cx, cy = dot.centroid
        cv2.rectangle(dots_overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.circle(dots_overlay, (int(cx), int(cy)), 2, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "dots_overlay.png"), dots_overlay)

    # Lines Overlay
    lines_overlay = img_color.copy()
    for i, line in enumerate(lines):
        if not line:
            continue
        min_y = min(dot.bbox[1] for dot in line)
        max_y = max(dot.bbox[1] + dot.bbox[3] for dot in line)

        cv2.rectangle(
            lines_overlay, (0, min_y - 2), (image.shape[1], max_y + 2), (255, 0, 0), 2
        )
        cv2.putText(
            lines_overlay,
            f"Line {i}",
            (10, min_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
    cv2.imwrite(os.path.join(output_dir, "lines_overlay.png"), lines_overlay)

    # Cells and Ordering Overlay
    cells_overlay = img_color.copy()
    for cell in cells:
        x, y, w, h = cell.bbox
        cv2.rectangle(cells_overlay, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(
            cells_overlay,
            str(cell.order_index),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    cv2.imwrite(os.path.join(output_dir, "cells_overlay.png"), cells_overlay)
