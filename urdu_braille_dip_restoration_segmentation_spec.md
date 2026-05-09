# Pakistani Urdu Braille Translation System
## DIP Restoration + Segmentation Technical Specification

---

# 1. Purpose of This Document

This document defines the complete implementation requirements for the **Digital Image Processing (DIP) Restoration and Segmentation stages** of a Pakistani Urdu Braille OCR system.

The goal is to provide sufficient technical context so that an implementation agent can:

1. Understand the project architecture.
2. Understand dataset assumptions.
3. Implement the complete restoration pipeline.
4. Implement robust Braille cell segmentation.
5. Produce reusable, modular code.
6. Generate debugging visualizations.
7. Prepare outputs compatible with later CNN classification and liblouis linguistic translation.

This document focuses ONLY on:

- Image restoration
- Binary preprocessing
- Dot extraction
- Braille cell segmentation
- Reading order reconstruction
- Debugging outputs

This document intentionally excludes:

- CNN training
- Deep learning optimization
- Urdu language modeling
- liblouis translation internals

Those are downstream modules.

---

# 2. High-Level System Architecture

The complete OCR pipeline consists of:

```text
Input Image
→ DIP Restoration
→ Dot Extraction
→ Braille Cell Segmentation
→ Ordered Cell Sequence
→ CNN Pattern Classification
→ Unicode Braille Conversion
→ liblouis Urdu Translation
→ Final Urdu Text
```

This document covers the first four stages only.

---

# 3. Dataset Assumptions

The dataset contains:

## 3.1 Image

A degraded photorealistic image containing multiple Braille cells.

Images may contain:

- Gaussian noise
- Salt-and-pepper noise
- Blur
- Morphological distortions
- Uneven illumination
- Dot deformation

---

## 3.2 Dot Metadata

Ground-truth Braille dot patterns for every cell.

Example:

```json
[
  "100000",
  "110000",
  "101010"
]
```

Each six-bit pattern represents:

```text
1 4
2 5
3 6
```

---

## 3.3 Chunk Metadata

The final Urdu translation.

Important:

The dataset stores Braille in:

```text
Logical Left-to-Right order
```

Therefore:

- image order
- segmentation order
- Unicode Braille order
- liblouis input order

must remain consistent.

DO NOT reverse cell ordering for Urdu.

Urdu visual RTL rendering is handled later by Unicode-aware rendering systems.

---

# 4. Required Implementation Modules

The implementation MUST produce modular components.

Required modules:

```text
project/
├── preprocessing/
│   ├── denoise.py
│   ├── threshold.py
│   ├── morphology.py
│   └── pipeline.py
│
├── segmentation/
│   ├── dot_detection.py
│   ├── line_detection.py
│   ├── cell_grouping.py
│   ├── ordering.py
│   └── cropper.py
│
├── visualization/
│   ├── debug_views.py
│   └── overlays.py
│
├── outputs/
│   ├── cleaned/
│   ├── binary/
│   ├── overlays/
│   ├── cells/
│   └── logs/
│
└── main.py
```

---

# 5. Input/Output Requirements

## 5.1 Input

Single image path:

```python
image_path: str
```

---

## 5.2 Output

The restoration + segmentation system MUST output:

### A. Cleaned image

```python
cleaned_image: np.ndarray
```

---

### B. Binary image

```python
binary_image: np.ndarray
```

---

### C. Detected dot candidates

```python
List[Dot]
```

Where:

```python
Dot(
    x: int,
    y: int,
    area: float,
    bbox: Tuple[int, int, int, int]
)
```

---

### D. Segmented Braille cells

```python
List[BrailleCell]
```

Where:

```python
BrailleCell(
    id: int,
    bbox: Tuple[int, int, int, int],
    image: np.ndarray,
    line_index: int,
    order_index: int
)
```

---

### E. Ordered cell sequence

The final sequence MUST preserve dataset logical order.

---

### F. Debug overlays

Images visualizing:

- dot detections
- line detections
- cell boundaries
- ordering indices

---

# 6. Restoration Pipeline Requirements

The restoration system MUST be modular and configurable.

The implementation should support experimentation.

---

# 7. Step 1 — Image Loading

Requirements:

- Use OpenCV.
- Validate image existence.
- Convert to grayscale.

Required output:

```python
gray_image
```

Recommended implementation:

```python
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

# 8. Step 2 — Noise Analysis

The implementation SHOULD estimate image quality.

Recommended metrics:

- Mean intensity
- Variance
- Histogram spread
- Local contrast

Purpose:

Adaptive preprocessing selection.

---

# 9. Step 3 — Denoising

The implementation MUST support multiple denoising strategies.

---

## 9.1 Median Filtering

Required for salt-and-pepper noise.

Recommended:

```python
cv2.medianBlur(gray, 3)
```

or:

```python
cv2.medianBlur(gray, 5)
```

---

## 9.2 Gaussian Filtering

Required for Gaussian noise.

Recommended:

```python
cv2.GaussianBlur(gray, (5,5), sigmaX=1.0)
```

---

## 9.3 Bilateral Filtering

Optional.

Useful when preserving edges.

---

# 10. Step 4 — Contrast Enhancement

Recommended techniques:

## CLAHE

Required support:

```python
cv2.createCLAHE()
```

Purpose:

Improve weak Braille dots.

---

# 11. Step 5 — Adaptive Thresholding

This stage is CRITICAL.

Global thresholding is NOT sufficient.

Required implementation:

```python
cv2.adaptiveThreshold()
```

Recommended configuration:

```python
adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)
```

The implementation MUST allow parameter tuning.

---

# 12. Step 6 — Morphological Cleanup

Required operations:

## Opening

Purpose:

Remove isolated noise.

```python
cv2.morphologyEx(..., cv2.MORPH_OPEN)
```

---

## Closing

Purpose:

Reconnect fragmented dots.

```python
cv2.morphologyEx(..., cv2.MORPH_CLOSE)
```

---

## Erosion

Optional.

Useful for separating merged blobs.

---

## Dilation

Optional.

Useful for strengthening weak dots.

---

# 13. Required Debug Outputs

The system MUST save intermediate outputs:

```text
01_gray.png
02_denoised.png
03_threshold.png
04_morphology.png
```

This is mandatory.

---

# 14. Dot Detection Requirements

The goal is to detect candidate Braille dots.

---

# 15. Connected Component Analysis

Required implementation:

```python
cv2.connectedComponentsWithStats()
```

Each component should produce:

- area
- centroid
- bounding box

---

# 16. Component Filtering

Components MUST be filtered using:

| Feature | Purpose |
|---|---|
| Area | Remove tiny noise |
| Aspect ratio | Remove elongated artifacts |
| Circularity | Keep dot-like regions |
| Bounding size | Reject impossible components |

---

# 17. Circularity Metric

Recommended formula:

```text
4πA / P²
```

Where:

- A = area
- P = perimeter

Dots should be approximately circular.

---

# 18. Dot Data Structure

Required structure:

```python
Dot(
    x,
    y,
    area,
    bbox,
    contour
)
```

---

# 19. Dot Visualization

Required overlay:

- draw bounding boxes
- draw centroids
- label indices

Save:

```text
dots_overlay.png
```

---

# 20. Line Detection

The segmentation system MUST group dots into text lines.

---

# 21. Horizontal Projection Profiles

Required implementation:

Compute:

```python
horizontal_sum = np.sum(binary_image, axis=1)
```

Use valleys between peaks to detect line boundaries.

---

# 22. Line Clustering

Alternative approach:

Cluster dot centroids by Y-coordinate.

Recommended:

- DBSCAN
- Agglomerative clustering
- Simple threshold grouping

---

# 23. Braille Cell Grouping

This is one of the most important modules.

The system MUST reconstruct:

```text
3 rows × 2 columns
```

Braille cell geometry.

---

# 24. Required Grouping Logic

Within each detected line:

1. Sort dots by X coordinate.
2. Estimate average horizontal spacing.
3. Estimate average vertical spacing.
4. Form candidate cells.
5. Validate geometric consistency.

---

# 25. Cell Geometry Constraints

A valid Braille cell should approximately satisfy:

| Constraint | Meaning |
|---|---|
| 2 columns | left/right dots |
| 3 rows | top/middle/bottom |
| consistent spacing | regular grid |
| bounded width | reject merged cells |

---

# 26. Cell Bounding Box Construction

The implementation MUST generate:

```python
(x, y, w, h)
```

for every cell.

Padding SHOULD be added.

Recommended:

```python
padding = 5
```

---

# 27. Cell Cropping

Each segmented cell MUST be cropped.

Recommended output size:

```text
64×64
```

Use:

```python
cv2.resize()
```

Preserve binary representation.

---

# 28. Cell Normalization

Required:

- binary normalization
- centered dots
- fixed dimensions

These outputs will feed the CNN later.

---

# 29. Ordering Requirements

IMPORTANT:

The dataset uses:

```text
Logical Left-to-Right Braille ordering
```

Therefore:

Within each line:

```python
sort by ascending x
```

NOT descending x.

---

# 30. Multi-Line Ordering

Required ordering:

```text
Top-to-bottom
Then left-to-right within each line
```

Equivalent to standard row-major reading order.

---

# 31. Ordered Sequence Output

Required final output:

```python
ordered_cells = [
    cell_1,
    cell_2,
    cell_3,
    ...
]
```

This sequence MUST align with:

- ground-truth dot metadata
- liblouis input order

---

# 32. Debug Visualization Requirements

The implementation MUST produce:

## A. Dot overlay

Detected dots.

---

## B. Line overlay

Detected line boundaries.

---

## C. Cell overlay

Detected Braille cells.

---

## D. Ordering overlay

Display:

```text
cell index
```

on top of each cell.

This is REQUIRED.

---

# 33. Output Directory Requirements

The implementation MUST save:

```text
outputs/
├── cleaned/
├── binary/
├── overlays/
├── cells/
└── logs/
```

---

# 34. Logging Requirements

The implementation MUST log:

- number of detected dots
- number of rejected components
- number of detected lines
- number of segmented cells
- average dot spacing
- processing time

---

# 35. Failure Handling

The implementation MUST gracefully handle:

| Failure | Required behavior |
|---|---|
| No dots found | log + continue |
| Weak thresholding | save diagnostics |
| Merged components | warn user |
| Empty crops | reject safely |
| Invalid image | raise clear exception |

---

# 36. Performance Requirements

The implementation should prioritize:

1. Transparency
2. Modularity
3. Debuggability
4. Deterministic behavior

Raw speed is secondary.

---

# 37. Recommended Libraries

Required:

| Library | Purpose |
|---|---|
| OpenCV | image processing |
| NumPy | numerical operations |
| matplotlib | visualization |
| scikit-image | optional morphology |
| scikit-learn | optional clustering |

---

# 38. Expected Final Deliverables

The implementation agent should produce:

## A. Modular source code

---

## B. Debug visualization outputs

---

## C. Saved segmented cell crops

---

## D. Ordered cell sequence

---

## E. Reusable preprocessing pipeline

---

## F. Configuration support

Thresholds and morphology parameters should be configurable.

---

# 39. Future Compatibility Requirements

The segmentation outputs MUST be compatible with:

```text
CNN classifier
→ 64-class Braille pattern recognition
```

Therefore:

Each cell crop MUST:

- contain exactly one Braille cell
- be normalized
- preserve dot geometry
- avoid excessive padding

---

# 40. Critical Design Principle

The system MUST remain:

```text
Interpretable and DIP-centric
```

The project MUST avoid:

- end-to-end black-box OCR
- YOLO-style object detection
- full-page deep-learning inference

The CNN is ONLY responsible for:

```text
single-cell pattern classification
```

All restoration and segmentation must remain mathematically transparent.

---

# 41. Recommended Development Order

The implementation agent should follow this order:

```text
1. Image loading
2. Grayscale conversion
3. Denoising
4. Adaptive thresholding
5. Morphological cleanup
6. Dot extraction
7. Dot filtering
8. Line detection
9. Cell grouping
10. Cell ordering
11. Cell cropping
12. Debug visualization
13. Output saving
```

DO NOT start with CNN training.

---

# 42. Final Objective

The final result of this stage should be:

```text
A robust ordered sequence of clean Braille cell crops
```

ready for:

```text
CNN pattern classification
→ Unicode Braille conversion
→ liblouis Urdu translation
```

