import os, glob
from preprocessing.pipeline import preprocess_img
from segmentation.dot_detection import detect_dots
from segmentation.grouping import group_into_lines, segment_cells_from_lines

img_files = sorted(glob.glob("inputs/images/*.jpg"))[:5]
for img_path in img_files:
    img_id = os.path.basename(img_path).split('.')[0]
    with open(f"inputs/dots/{img_id}.txt") as f:
        gt_dots = f.read().strip().split()
    try:
        _, _, bin = preprocess_img(img_path, "cnn_dataset/temp_dip")
        dots = detect_dots(bin)
        lines = group_into_lines(dots, eps=15.0)
        cells = segment_cells_from_lines(lines, 10.0, 17.0)
        print(f"{img_id}: Segmented {len(cells)}, Ground truth {len(gt_dots)}")
    except Exception as e:
        print(f"Error {img_path}: {e}")
