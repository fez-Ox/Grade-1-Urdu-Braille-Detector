from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from segmentation.dot_detection import Dot


@dataclass
class BrailleCell:
    id: int
    dots: List[Dot]
    bbox: Tuple[int, int, int, int]
    line_index: int
    order_index: int
    image: np.ndarray = None


def group_into_lines(dots: List[Dot], eps: float = 15.0) -> List[List[Dot]]:
    if not dots:
        return []

    y_coords = np.array([[dot.y] for dot in dots])

    # Cluster dots into horizontal lines
    clustering = DBSCAN(eps=eps, min_samples=1).fit(y_coords)
    labels = clustering.labels_

    lines_dict = {}
    for dot, label in zip(dots, labels):
        if label not in lines_dict:
            lines_dict[label] = []
        lines_dict[label].append(dot)

    # Sort lines top-to-bottom based on average Y coordinate
    lines = list(lines_dict.values())
    lines.sort(key=lambda line: np.mean([dot.y for dot in line]))

    return lines


def segment_cells_from_lines(
    lines: List[List[Dot]],
    col_group_eps: float = 10.0,
    intra_cell_threshold: float = 17.0,
) -> List[BrailleCell]:
    cells = []
    order_idx = 0

    for line_idx, line in enumerate(lines):
        # 1. Group dots into vertical columns within the line
        sorted_line = sorted(line, key=lambda dot: dot.x)

        columns = []
        current_col_dots = [sorted_line[0]]
        for dot in sorted_line[1:]:
            # If dot is horizontally very close, it belongs to the same column
            if dot.x - np.mean([d.x for d in current_col_dots]) < col_group_eps:
                current_col_dots.append(dot)
            else:
                columns.append(current_col_dots)
                current_col_dots = [dot]
        columns.append(current_col_dots)

        # Calculate average X for each column to represent its position
        col_xs = [np.mean([d.x for d in col]) for col in columns]

        # 2. Group columns into cells based on distance
        current_cell_cols = [columns[0]]

        for i in range(1, len(columns)):
            dist = col_xs[i] - col_xs[i - 1]

            # If distance between this column and previous is less than the threshold,
            # and the current cell doesn't already have 2 columns, it's the 2nd column of the cell.
            if dist < intra_cell_threshold and len(current_cell_cols) < 2:
                current_cell_cols.append(columns[i])
            else:
                # Finalize the current cell
                cell_dots = [dot for col in current_cell_cols for dot in col]
                cells.append(_create_cell(cell_dots, line_idx, order_idx))
                order_idx += 1

                # Start a new cell with the current column
                current_cell_cols = [columns[i]]

        # Finalize the last cell in the line
        if current_cell_cols:
            cell_dots = [dot for col in current_cell_cols for dot in col]
            cells.append(_create_cell(cell_dots, line_idx, order_idx))
            order_idx += 1

    return cells


def _create_cell(dots: List[Dot], line_idx: int, order_idx: int) -> BrailleCell:
    # Compute bounding box
    min_x = min(dot.bbox[0] for dot in dots)
    min_y = min(dot.bbox[1] for dot in dots)
    max_x = max(dot.bbox[0] + dot.bbox[2] for dot in dots)
    max_y = max(dot.bbox[1] + dot.bbox[3] for dot in dots)

    padding = 5
    bbox = (
        max(0, min_x - padding),
        max(0, min_y - padding),
        (max_x - min_x) + 2 * padding,
        (max_y - min_y) + 2 * padding,
    )

    return BrailleCell(
        id=order_idx, dots=dots, bbox=bbox, line_index=line_idx, order_index=order_idx
    )
