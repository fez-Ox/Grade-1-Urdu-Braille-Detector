import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# OpenCV-equivalent Stat Constants
CC_STAT_LEFT = 0
CC_STAT_TOP = 1
CC_STAT_WIDTH = 2
CC_STAT_HEIGHT = 3
CC_STAT_AREA = 4


def connected_components_with_stats(binary_image, connectivity=8):

    # Ensure binary image is 0 and 1
    binary = (binary_image > 0).astype(np.int32)
    h, w = binary.shape

    # Pad image to avoid boundary checks during neighbor lookup
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    labels_pad = np.zeros_like(padded, dtype=np.int32)

    next_label = 1
    linked = [0]  # Index is label, value is parent label

    
    y_coords, x_coords = np.nonzero(binary)
    y_coords += 1  # Offset for padded array
    x_coords += 1

    for i in range(len(y_coords)):
        y = y_coords[i]
        x = x_coords[i]

        if connectivity == 8:
            # Check Top-Left, Top, Top-Right, Left
            neighbors = (
                labels_pad[y - 1, x - 1],
                labels_pad[y - 1, x],
                labels_pad[y - 1, x + 1],
                labels_pad[y, x - 1],
            )
        else:
            # Check Top, Left
            neighbors = (labels_pad[y - 1, x], labels_pad[y, x - 1])

        valid = [n for n in neighbors if n > 0]

        if not valid:
            linked.append(next_label)
            labels_pad[y, x] = next_label
            next_label += 1
        else:
            # Find the root parents for all valid neighbors
            roots = []
            for n in valid:
                root = n
                while linked[root] != root:
                    root = linked[root]
                roots.append(root)

            min_root = min(roots)
            labels_pad[y, x] = min_root

            # Union: link all neighbor roots to the smallest root
            for root in roots:
                if root != min_root:
                    linked[root] = min_root

    for i in range(1, next_label):
        root = i
        while linked[root] != root:
            root = linked[root]
        linked[i] = root

    linked_array = np.array(linked, dtype=np.int32)
    labels = linked_array[labels_pad[1 : h + 1, 1 : w + 1]]

    unique_labels, label_indices = np.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    labels_seq = label_indices.reshape((h, w))

    flat_labels = labels_seq.ravel()
    Y, X = np.indices((h, w))
    flat_X = X.ravel()
    flat_Y = Y.ravel()

    areas = np.bincount(flat_labels)

    stats = np.zeros((num_labels, 5), dtype=np.int32)
    centroids = np.zeros((num_labels, 2), dtype=np.float64)

    stats[:, CC_STAT_AREA] = areas

    # Left (min X) and Top (min Y)
    stats[:, CC_STAT_LEFT] = w + 1
    stats[:, CC_STAT_TOP] = h + 1
    np.minimum.at(stats[:, CC_STAT_LEFT], flat_labels, flat_X)
    np.minimum.at(stats[:, CC_STAT_TOP], flat_labels, flat_Y)

    # Max X and Max Y to calculate Width and Height
    max_x = np.zeros(num_labels, dtype=np.int32) - 1
    max_y = np.zeros(num_labels, dtype=np.int32) - 1
    np.maximum.at(max_x, flat_labels, flat_X)
    np.maximum.at(max_y, flat_labels, flat_Y)

    stats[:, CC_STAT_WIDTH] = max_x - stats[:, CC_STAT_LEFT] + 1
    stats[:, CC_STAT_HEIGHT] = max_y - stats[:, CC_STAT_TOP] + 1

    # Calculate centroids (sum of coords / area)
    sum_x = np.bincount(flat_labels, weights=flat_X)
    sum_y = np.bincount(flat_labels, weights=flat_Y)

    safe_areas = np.maximum(areas, 1)  # Prevent divide-by-zero
    centroids[:, 0] = sum_x / safe_areas
    centroids[:, 1] = sum_y / safe_areas

    return num_labels, labels_seq, stats, centroids


@dataclass
class Dot:
    x: int
    y: int
    area: float
    bbox: Tuple[int, int, int, int]  # The Bounding Box with its boundaries
    centroid: Tuple[float, float]


def detect_dots(binary_image) -> List[Dot]:

    num_labels, labels, stats, centroids = connected_components_with_stats(
        binary_image, connectivity=8
    )

    dots = []

    # Skipping 0 since background
    for i in range(1, num_labels):
        x = stats[i, CC_STAT_LEFT]
        y = stats[i, CC_STAT_TOP]
        w = stats[i, CC_STAT_WIDTH]
        h = stats[i, CC_STAT_HEIGHT]
        area = stats[i, CC_STAT_AREA]
        cx, cy = centroids[i]

        # Area filtering for noise or objects unlikely to be Braille Dots
        if area < 5 or area > 300:
            continue

        # Aspect ratio filtering
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        dots.append(
            Dot(x=int(cx), y=int(cy), area=area, bbox=(x, y, w, h), centroid=(cx, cy))
        )

    return dots
