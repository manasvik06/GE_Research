"""
Benchmarking 2D convolution methods:
1. Direct convolution with nested loops
2. Manual FFT-based convolution using numpy.fft
3. scipy.signal.fftconvolve as an optimized reference

"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


def direct_convolve2d(image, kernel):
    """Compute same-size 2D convolution using direct nested loops."""
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    pad_h = ker_h // 2
    pad_w = ker_w // 2

    # Zero padding keeps the output the same size as the input image.
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image, dtype=np.float64)

    # For each output pixel, take a local image patch and compute
    # the dot product with the kernel.
    for i in range(img_h):
        for j in range(img_w):
            patch = padded[i:i + ker_h, j:j + ker_w]
            output[i, j] = np.sum(patch * kernel)

    return output


def fft_convolve2d_manual(image, kernel):
    """
    Compute same-size 2D convolution using the convolution theorem:
    convolution in space = multiplication in frequency.
    """
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    # Padding to the full convolution size prevents circular wrap-around.
    out_h = img_h + ker_h - 1
    out_w = img_w + ker_w - 1

    # Transform both image and kernel into the frequency domain.
    image_fft = np.fft.fft2(image, s=(out_h, out_w))
    kernel_fft = np.fft.fft2(kernel, s=(out_h, out_w))

    # Pointwise multiplication in frequency domain, then inverse transform.
    full_result = np.fft.ifft2(image_fft * kernel_fft).real

    # Crop to match scipy's mode="same" output size.
    pad_h = ker_h // 2
    pad_w = ker_w // 2

    return full_result[pad_h:pad_h + img_h, pad_w:pad_w + img_w]


def fft_convolve2d_scipy(image, kernel):
    """Compute same-size 2D convolution using scipy's optimized FFT implementation."""
    return fftconvolve(image, kernel, mode="same")


def verify_outputs_match(image, kernel, tolerance=1e-6):
    """Check that all three implementations produce nearly identical outputs."""
    print("Step 1: Checking correctness...")

    direct_out = direct_convolve2d(image, kernel)
    manual_fft_out = fft_convolve2d_manual(image, kernel)
    scipy_fft_out = fft_convolve2d_scipy(image, kernel)

    diff_direct_manual = np.max(np.abs(direct_out - manual_fft_out))
    diff_manual_scipy = np.max(np.abs(manual_fft_out - scipy_fft_out))

    print(f"  Max difference, direct vs manual FFT: {diff_direct_manual:.2e}")
    print(f"  Max difference, manual FFT vs scipy:  {diff_manual_scipy:.2e}")

    passed = diff_direct_manual < tolerance and diff_manual_scipy < tolerance

    if passed:
        print("  Outputs match within tolerance.\n")
    else:
        print("  Outputs do not match. Check implementation.\n")

    return passed


def time_method(method, image, kernel, runs=3):
    """Return average runtime over multiple runs."""
    times = []

    for _ in range(runs):
        start = time.perf_counter()
        method(image, kernel)
        times.append(time.perf_counter() - start)

    return np.mean(times)


def run_benchmark(image_sizes, kernel_size=15, runs=3):
    """
    Benchmark direct, manual FFT, and scipy FFT convolution.

    The benchmark uses randomly generated images so that runtime depends mainly
    on image size and algorithm choice, not on loading files or image content.

    The kernel is a normalized averaging/box-blur filter. This is a simple,
    interpretable smoothing kernel often used to demonstrate convolution.
    """
    direct_times = []
    manual_fft_times = []
    scipy_fft_times = []

    # Normalized averaging kernel.
    # For a 15x15 kernel, each value is 1/225.
    # This makes each output pixel the average of its local 15x15 neighborhood.
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    kernel /= kernel.sum()

    for size in image_sizes:
        print(f"Testing {size}x{size} image with {kernel_size}x{kernel_size} kernel...")

        # We generate a random synthetic image
        # Values are in [0, 1], similar to a normalized grayscale image.
        # Using random data keeps the benchmark controlled and repeatable by size.
        image = np.random.rand(size, size).astype(np.float64)

        # Direct convolution becomes very slow for large images because it scales as O(N^2 K^2).
        # I only run it up to 256x256 so the benchmark finishes in reasonable time.
        if size <= 256:
            direct_time = time_method(direct_convolve2d, image, kernel, runs)
            direct_times.append(direct_time)
        else:
            direct_times.append(None)

        # Manual FFT shows the algorithmic pipeline clearly:
        # FFT image, FFT kernel, multiply, inverse FFT.
        manual_fft_times.append(time_method(fft_convolve2d_manual, image, kernel, runs))

        # scipy's implementation is included as an optimized library reference.
        scipy_fft_times.append(time_method(fft_convolve2d_scipy, image, kernel, runs))

    return direct_times, manual_fft_times, scipy_fft_times


def plot_results(image_sizes, direct_times, manual_fft_times, scipy_fft_times, kernel_size):
    """Plot runtime comparison."""
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        image_sizes,
        scipy_fft_times,
        "o-",
        linewidth=2.5,
        markersize=7,
        label="scipy fftconvolve"
    )

    ax.plot(
        image_sizes,
        manual_fft_times,
        "^--",
        linewidth=2.5,
        markersize=7,
        label="Manual FFT using numpy.fft2"
    )

    direct_sizes = [
        image_sizes[i]
        for i, runtime in enumerate(direct_times)
        if runtime is not None
    ]

    direct_values = [
        runtime
        for runtime in direct_times
        if runtime is not None
    ]

    ax.plot(
        direct_sizes,
        direct_values,
        "s--",
        linewidth=2.5,
        markersize=7,
        label=f"Direct convolution, K={kernel_size}"
    )

    if direct_sizes:
        ax.axvline(x=direct_sizes[-1], linestyle=":", linewidth=1.2)
        ax.text(
            direct_sizes[-1] + 10,
            max(direct_values) * 0.75,
            "direct method\nskipped after 256",
            fontsize=9
        )

    ax.set_xlabel("Image Size (N × N pixels)", fontsize=13)
    ax.set_ylabel("Average Runtime (seconds)", fontsize=13)
    ax.set_title(
        f"2D Convolution Benchmark\nKernel: {kernel_size}x{kernel_size}, averaged over 3 runs",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    plt.savefig("convolution_benchmark.png", dpi=150, bbox_inches="tight")
    print("\nChart saved: convolution_benchmark.png")
    plt.show()


def print_results_table(image_sizes, direct_times, manual_fft_times, scipy_fft_times):
    """Print benchmark results in table format."""
    print("\nResults Summary:")
    print(f"{'Size':<12} {'Direct':>14} {'Manual FFT':>14} {'Scipy FFT':>14}")
    print("-" * 56)

    for i, size in enumerate(image_sizes):
        direct = (
            f"{direct_times[i]:.4f}s"
            if direct_times[i] is not None
            else "skipped"
        )

        manual = f"{manual_fft_times[i]:.4f}s"
        scipy = f"{scipy_fft_times[i]:.4f}s"

        print(f"{size}x{size:<6}   {direct:>12}   {manual:>12}   {scipy:>12}")

def create_output_comparison(kernel_size=15):
    """
    Create a 4-panel visual comparison:
    original image, direct convolution, manual FFT convolution, and scipy FFT convolution.
    """
    size = 128
    image = np.zeros((size, size), dtype=np.float64)

    # Simple synthetic image with clear shapes and edges.
    image[25:95, 25:95] = 0.6

    y, x = np.ogrid[:size, :size]
    center_y, center_x = 64, 72
    radius = 20
    circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    image[circle] = 1.0

    for i in range(20, 110):
        image[i, i // 2 + 20:i // 2 + 23] = 0.9

    # Same averaging kernel as the benchmark.
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    kernel /= kernel.sum()

    direct_out = direct_convolve2d(image, kernel)
    manual_fft_out = fft_convolve2d_manual(image, kernel)
    scipy_fft_out = fft_convolve2d_scipy(image, kernel)

    diff_direct_manual = np.max(np.abs(direct_out - manual_fft_out))
    diff_manual_scipy = np.max(np.abs(manual_fft_out - scipy_fft_out))

    print("\nVisual output comparison:")
    print(f"  Max difference, direct vs manual FFT: {diff_direct_manual:.2e}")
    print(f"  Max difference, manual FFT vs scipy:  {diff_manual_scipy:.2e}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("white")

    panels = [
        (image, "Original"),
        (direct_out, "Direct"),
        (manual_fft_out, "Manual FFT"),
        (scipy_fft_out, "SciPy FFT"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Output Comparison: All Methods Produce the Same Filtered Image",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("convolution_output_comparison.png", dpi=300, bbox_inches="tight")
    print("Output comparison saved: convolution_output_comparison.png")
    plt.show()


def main():
    # Sizes chosen to show scaling from small cases to 1024x1024.
    # 512x512 is also a common size for medical image slices.
    image_sizes = [64, 128, 256, 512, 1024]

    # I chose a 15x15 kernel because it is large enough to show the FFT advantage.
    # With very small kernels like 3x3, direct convolution can still be competitive
    # because FFT has extra overhead from padding and transforms.
    kernel_size = 15

    runs = 3

    print("=" * 60)
    print("2D Convolution Benchmark")
    print("=" * 60)
    print(f"Kernel size: {kernel_size}x{kernel_size}")
    print(f"Image sizes: {image_sizes}\n")

    # Small correctness test before timing larger inputs.
    small_image = np.random.rand(32, 32).astype(np.float64)
    small_kernel = np.ones((5, 5), dtype=np.float64) / 25.0
    verify_outputs_match(small_image, small_kernel)

    print("Step 2: Running benchmark...\n")
    direct_times, manual_fft_times, scipy_fft_times = run_benchmark(
        image_sizes,
        kernel_size=kernel_size,
        runs=runs
    )

    print_results_table(image_sizes, direct_times, manual_fft_times, scipy_fft_times)

    print("\nStep 3: Generating plot...")
    plot_results(
        image_sizes,
        direct_times,
        manual_fft_times,
        scipy_fft_times,
        kernel_size
    )

    print("\nStep 4: Generating output comparison...")
    create_output_comparison(kernel_size=kernel_size)


if __name__ == "__main__":
    main()