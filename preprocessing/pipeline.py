import os
from . import diplib as dip


def preprocess_img(image_path, output_dir):
    # Load image
    img = dip.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = dip.bgr2gray(img)

    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, "cleaned"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "binary"), exist_ok=True)

    # Save grayscale
    dip.imwrite(os.path.join(output_dir, "cleaned", "01_gray.png"), gray)

    # denoise
    denoised = dip.median_blur(gray, 3)
    denoised = dip.gaussian_blur(denoised, 3, 0)
    dip.imwrite(os.path.join(output_dir, "cleaned", "02_denoised.png"), denoised)

    #  Contrast Enhancement
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # enhanced = clahe.apply(denoised)
    # cv2.imwrite(os.path.join(output_dir, "cleaned", "03_enhanced.png"), enhanced)
    enhanced = denoised

    #  Adaptive Thresholding
    # Increased C from 5 to 12 to filter out shadow/lighting artifacts (ghost dots)
    binary = dip.adaptive_threshold_gaussian_inv(
        enhanced, max_val=255, block_size=15, C=12
    )
    dip.imwrite(os.path.join(output_dir, "binary", "04_threshold.png"), binary)

    #  Morphological Cleanup
    kernel_open = dip.get_structuring_element_ellipse((2, 2))
    kernel_close = dip.get_structuring_element_ellipse((3, 3))

    opened = dip.morphology_open(binary, kernel_open)
    cleaned_binary = dip.morphology_close(opened, kernel_close)

    dip.imwrite(os.path.join(output_dir, "binary", "05_morphology.png"), cleaned_binary)

    return img, gray, cleaned_binary
