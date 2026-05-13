import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

def imread(path):
    try:
        img = Image.open(path).convert('RGB')
        # Convert RGB to BGR to mimic OpenCV's default behavior
        return np.array(img)[:, :, ::-1]
    except Exception:
        return None

def imwrite(path, img_array):
    # If the image is 3D (BGR), convert it back to RGB for saving
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, ::-1]
    Image.fromarray(img_array.astype(np.uint8)).save(path)

def bgr2gray(img):
    # Luminosity formula: 0.114*B + 0.587*G + 0.299*R
    gray = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    return gray.astype(np.uint8)

def _pad_image(img, kh, kw, mode='reflect', constant_values=0):
    pad_top = kh // 2
    pad_bottom = kh - pad_top - 1
    pad_left = kw // 2
    pad_right = kw - pad_left - 1
    
    if mode == 'constant':
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                      mode=mode, constant_values=constant_values)
    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)

def median_blur(img, ksize):
    padded = _pad_image(img, ksize, ksize, mode='reflect')
    windows = sliding_window_view(padded, (ksize, ksize))
    return np.median(windows, axis=(-2, -1)).astype(img.dtype)

def _gaussian_kernel(ksize, sigma=0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def gaussian_blur(img, ksize, sigma=0):
    kernel = _gaussian_kernel(ksize, sigma)
    padded = _pad_image(img, ksize, ksize, mode='reflect')
    windows = sliding_window_view(padded, (ksize, ksize))
    out = np.sum(windows * kernel, axis=(-2, -1))
    return out.astype(img.dtype)

def adaptive_threshold_gaussian_inv(img, max_val, block_size, C):
    # The local threshold is the gaussian-weighted mean of the block minus C
    kernel = _gaussian_kernel(block_size, sigma=0)
    padded = _pad_image(img, block_size, block_size, mode='reflect')
    windows = sliding_window_view(padded, (block_size, block_size))
    
    local_weighted_mean = np.sum(windows * kernel, axis=(-2, -1))
    threshold = local_weighted_mean - C
    
    binary = np.zeros_like(img)
    # Binary Inverse logic: if src(x,y) > T(x,y), out=0, else out=max_val
    binary[img <= threshold] = max_val
    return binary.astype(np.uint8)

def get_structuring_element_ellipse(ksize):
    h, w = ksize
    center_h, center_w = (h - 1) / 2., (w - 1) / 2.
    y, x = np.indices((h, w))
    
    # Ellipse equation normalized: (x-cx)^2/rx^2 + (y-cy)^2/ry^2 <= 1
    dy = (y - center_h) / (max(h, 1) / 2.0)
    dx = (x - center_w) / (max(w, 1) / 2.0)
    
    mask = (dy**2 + dx**2) <= 1.0
    return mask.astype(np.uint8)

def erode(img, kernel):
    kh, kw = kernel.shape
    padded = _pad_image(img, kh, kw, mode='constant', constant_values=255)
    windows = sliding_window_view(padded, (kh, kw))
    
    # Ignore pixels outside the kernel structure by treating them as max value
    masked_windows = np.where(kernel, windows, 255)
    return np.min(masked_windows, axis=(-2, -1)).astype(np.uint8)

def dilate(img, kernel):
    kh, kw = kernel.shape
    padded = _pad_image(img, kh, kw, mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (kh, kw))
    
    # Ignore pixels outside the kernel structure by treating them as min value
    masked_windows = np.where(kernel, windows, 0)
    return np.max(masked_windows, axis=(-2, -1)).astype(np.uint8)

def morphology_open(img, kernel):
    return dilate(erode(img, kernel), kernel)

def morphology_close(img, kernel):
    return erode(dilate(img, kernel), kernel)
