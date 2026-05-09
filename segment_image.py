import sys
import time

from preprocessing.pipeline import preprocess_img
from segmentation.cropper import crop_and_save_cells
from segmentation.dot_detection import detect_dots
from segmentation.grouping import group_into_lines, segment_cells_from_lines
from visualization.overlays import generate_overlays


def main(image_path: str):
    start_time = time.time()
    print(f"Processing: {image_path}")

    output_dir = "outputs"

    # Phase 1: Preprocessing
    print("1. Restoring and Thresholding image...")
    img_bgr, img_gray, img_binary = preprocess_img(image_path, output_dir)

    # Phase 2: Dot Extraction
    print("2. Extracting dots...")
    dots = detect_dots(img_binary)
    print(f"   -> Detected {len(dots)} valid dots.")

    if not dots:
        print("No dots found. Exiting.")
        return

    # Phase 3: Grouping
    print("3. Grouping into lines and cells...")
    lines = group_into_lines(dots, eps=15.0)
    print(f"   -> Detected {len(lines)} lines.")

    cells = segment_cells_from_lines(
        lines, col_group_eps=10.0, intra_cell_threshold=17.0
    )
    print(f"   -> Detected {len(cells)} Braille cells.")

    # Phase 4: Cropping
    print("4. Cropping and saving individual cells...")
    # Clean output dir cells
    import glob
    import os

    for f in glob.glob(os.path.join(output_dir, "cells", "*.png")):
        os.remove(f)

    crop_and_save_cells(img_binary, cells, os.path.join(output_dir, "cells"))

    # Phase 5: Overlays
    print("5. Generating debug overlays...")
    generate_overlays(img_bgr, dots, lines, cells, os.path.join(output_dir, "overlays"))

    # Logging
    elapsed = time.time() - start_time
    avg_spacing = 0.0  # Could calculate this if needed

    log_content = (
        f"--- Execution Log ---\n"
        f"File: {image_path}\n"
        f"Detected dots: {len(dots)}\n"
        f"Detected lines: {len(lines)}\n"
        f"Segmented cells: {len(cells)}\n"
        f"Processing time: {elapsed:.3f} seconds\n"
    )

    with open(os.path.join(output_dir, "logs", "segmentation_log.txt"), "w") as f:
        f.write(log_content)

    print(log_content)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "inputs/images/chunk_00048.jpg"
    main(img_path)
