"""
Medical-style image filtering with convolution
Author: Manasvi Khandelwal

Applies common convolution filters to a real MRI brain scan:
1. Gaussian blur for smoothing / denoising
2. Sobel edge detection
3. Sharpening
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image


def load_image(path, size=512):
    # Grayscale image pulled from the internet
    img = Image.open(path).convert("L")
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float64)
    return arr / 255.0


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

    #apply filters sequentially
    t0 = time.perf_counter()
    denoised = apply_filter(image, gaussian)
    t_denoise = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges = compute_edge_magnitude(denoised)
    t_edges = time.perf_counter() - t0

    t0 = time.perf_counter()
    sharpened = apply_filter(image, SHARPEN_KERNEL)
    t_sharpen = time.perf_counter() - t0

    t_total = t_denoise + t_edges + t_sharpen

    print(f"\nFilter runtimes on {image.shape[0]}x{image.shape[1]} image:")
    print(f"  Gaussian blur      : {t_denoise*1000:.2f} ms")
    print(f"  Sobel edge detect  : {t_edges*1000:.2f} ms")
    print(f"  Sharpening         : {t_sharpen*1000:.2f} ms")
    print(f"  Total (3 filters)  : {t_total*1000:.2f} ms")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    panels = [
        (image,    "Original MRI"),
        (denoised, "Gaussian Blur\nDenoising"),
        (edges,    "Sobel Edge Detection"),
        (sharpened,"Sharpening Filter"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Convolution Filters on a Real MRI Brain Scan",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("medical_convolution_filters.png", dpi=300, bbox_inches="tight")
    print("Saved: medical_convolution_filters.png")
    plt.show()


def main():
    print("Loading MRI image...")
    image = load_image("image.png", size=512)

    print("Applying convolution filters...")
    create_filter_comparison(image)


if __name__ == "__main__":
    main()