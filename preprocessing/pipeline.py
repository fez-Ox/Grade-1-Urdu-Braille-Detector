import os

import cv2


def preprocess_img(image_path, output_dir):
    # 1. Image Loading
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save grayscale
    cv2.imwrite(os.path.join(output_dir, "cleaned", "01_gray.png"), gray)

    # 2. Denoising
    denoised = cv2.medianBlur(gray, 3)
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
    cv2.imwrite(os.path.join(output_dir, "cleaned", "02_denoised.png"), denoised)

    # 3. Contrast Enhancement
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(denoised)
    # cv2.imwrite(os.path.join(output_dir, "cleaned", "03_enhanced.png"), enhanced)
    enhanced = denoised

    # 4. Adaptive Thresholding
    # Increased C from 5 to 12 to filter out shadow/lighting artifacts (ghost dots)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 12
    )
    cv2.imwrite(os.path.join(output_dir, "binary", "04_threshold.png"), binary)

    # 5. Morphological Cleanup
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    cleaned_binary = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    cv2.imwrite(os.path.join(output_dir, "binary", "05_morphology.png"), cleaned_binary)

    return img, gray, cleaned_binary
