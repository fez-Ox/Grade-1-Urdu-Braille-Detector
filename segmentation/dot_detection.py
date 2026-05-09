from dataclasses import dataclass
from typing import List, Tuple

import cv2


@dataclass
class Dot:
    x: int
    y: int
    area: float
    bbox: Tuple[int, int, int, int]  # The Bounding Box with its boundaries
    centroid: Tuple[float, float]


def detect_dots(binary_image) -> List[Dot]:

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    dots = []

    # Skipping 1 since background
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        # Area filtering for noise or objects unlikely to be Braille Dots
        if area < 5 or area > 300:
            continue

        # Aspect ratio filtering
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        # Optional: Circularity check can be done via contours,
        # but bounding box and area are often enough for cleaned braille dots
        # For simplicity we assume if it passes aspect ratio and area it's a candidate dot.

        dots.append(
            Dot(x=int(cx), y=int(cy), area=area, bbox=(x, y, w, h), centroid=(cx, cy))
        )

    return dots
