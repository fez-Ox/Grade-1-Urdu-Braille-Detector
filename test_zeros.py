import os, glob

img_files = sorted(glob.glob("inputs/images/*.jpg"))[:5]
for img_path in img_files:
    img_id = os.path.basename(img_path).split('.')[0]
    with open(f"inputs/dots/{img_id}.txt") as f:
        gt_dots = f.read().strip().split()
    non_zero = [d for d in gt_dots if d != "0"]
    print(f"{img_id}: GT Total {len(gt_dots)}, Non-Zero {len(non_zero)}")
