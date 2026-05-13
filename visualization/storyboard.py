import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def create_storyboard(
    original_img, 
    preprocessed_img, 
    binary_img, 
    dots, 
    lines, 
    cells, 
    output_path="outputs/storyboard.png"
):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#f0f0f0')
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    #  Original Image
    axes[0, 0].imshow(original_img[:, :, ::-1]) # Convert BGR to RGB
    axes[0, 0].set_title("1. Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Preprocessed (Denoised/Gray)
    axes[0, 1].imshow(preprocessed_img, cmap='gray')
    axes[0, 1].set_title("2. DIP Preprocessing\n(Median + Gaussian Blur)", fontsize=12)
    axes[0, 1].axis('off')

    #  Binary Threshold
    axes[0, 2].imshow(binary_img, cmap='gray')
    axes[0, 2].set_title("3. Adaptive Thresholding\n(Gaussian C Inverse)", fontsize=12)
    axes[0, 2].axis('off')

    #  Dot Detection Overlay
    axes[1, 0].imshow(original_img[:, :, ::-1])
    for dot in dots:
        x, y, w, h = dot.bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='lime', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].plot(dot.centroid[0], dot.centroid[1], 'r.', markersize=3)
    axes[1, 0].set_title("4. Dot Detection\n(Custom Connected Components)", fontsize=12)
    axes[1, 0].axis('off')

    #  Cell Segmentation Overlay
    axes[1, 1].imshow(original_img[:, :, ::-1])
    for cell in cells:
        x, y, w, h = cell.bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='orange', facecolor='none')
        axes[1, 1].add_patch(rect)
        axes[1, 1].text(x, y-2, str(cell.order_index), color='red', fontsize=8, fontweight='bold')
    
    # Draw line boundaries
    for i, line in enumerate(lines):
        if line:
            min_y = min(dot.bbox[1] for dot in line)
            max_y = max(dot.bbox[1] + dot.bbox[3] for dot in line)
            axes[1, 1].axhline(y=min_y, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
            axes[1, 1].axhline(y=max_y, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)

    axes[1, 1].set_title("5. Line & Cell Grouping\n(Spatial Clustering)", fontsize=12)
    axes[1, 1].axis('off')

    #  Final Result / Legend / Stats
    axes[1, 2].text(0.5, 0.5, 
        f"Pipeline Summary:\n\n"
        f"• Dots Detected: {len(dots)}\n"
        f"• Lines Identified: {len(lines)}\n"
        f"• Cells Segmented: {len(cells)}\n"
        f"• Processing: Custom diplib.py\n"
        f"• Status: SUCCESS",
        ha='center', va='center', fontsize=14, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=1')
    )
    axes[1, 2].set_title("6. Execution Metadata", fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path
