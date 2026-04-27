"""
Medical-style image filtering with convolution
Author: Manasvi Khandelwal

This script creates a synthetic MRI-like grayscale image and applies
several common convolution filters:
1. Gaussian blur for smoothing / denoising
2. Sobel edge detection
3. Sharpening

The goal is to show how convolution can be used in an imaging context.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def create_mri_phantom(size=512):
    """Create a synthetic MRI-like grayscale phantom image."""
    image = np.zeros((size, size), dtype=np.float64)
    cy, cx = size // 2, size // 2

    y, x = np.ogrid[:size, :size]

    # Outer ellipse
    outer = ((x - cx) ** 2 / (cx * 0.85) ** 2 +
             (y - cy) ** 2 / (cy * 0.92) ** 2) <= 1
    image[outer] = 0.3

    # Inner ellipse
    inner = ((x - cx) ** 2 / (cx * 0.6) ** 2 +
             (y - cy) ** 2 / (cy * 0.65) ** 2) <= 1
    image[inner] = 0.7

    # Small bright circular region
    lesion_y, lesion_x = cy - 60, cx + 40
    lesion = (x - lesion_x) ** 2 + (y - lesion_y) ** 2 <= 18 ** 2
    image[lesion] = 1.0

    # Add Gaussian noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.04, image.shape)
    image = np.clip(image + noise, 0, 1)

    return image


def gaussian_kernel(size=7, sigma=1.5):
    """Create a normalized 2D Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


SOBEL_X = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float64)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)

SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float64)


def apply_filter(image, kernel):
    """Apply a convolution filter using FFT-based convolution."""
    result = fftconvolve(image, kernel, mode="same")
    return np.clip(result, 0, 1)


def compute_edge_magnitude(image):
    """Compute edge magnitude using Sobel X and Sobel Y filters."""
    gx = fftconvolve(image, SOBEL_X, mode="same")
    gy = fftconvolve(image, SOBEL_Y, mode="same")

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    if magnitude.max() == 0:
        return magnitude

    return magnitude / magnitude.max()


def create_filter_comparison(image):
    """Create and save a 4-panel comparison of convolution filters."""
    gaussian = gaussian_kernel(size=7, sigma=1.5)

    denoised = apply_filter(image, gaussian)
    edges = compute_edge_magnitude(denoised)
    sharpened = apply_filter(image, SHARPEN_KERNEL)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    panels = [
        (image, "Original Phantom"),
        (denoised, "Gaussian Blur\nDenoising"),
        (edges, "Sobel Edge Detection"),
        (sharpened, "Sharpening Filter"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Convolution Filters on a Synthetic Medical-Style Image",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("medical_convolution_filters.png", dpi=300, bbox_inches="tight")
    print("Saved: medical_convolution_filters.png")
    plt.show()


def main():
    print("Generating synthetic phantom image...")
    phantom = create_mri_phantom(size=512)

    print("Applying convolution filters...")
    create_filter_comparison(phantom)


if __name__ == "__main__":
    main()