"""
------------------------------------------------------------
Convolution Operation Animation
Author: Manasvi Khandelwal

Description:
This program creates an animation demonstrating how a 2D
convolution operation works in image processing.

A 3x3 blur kernel slides across a 5x5 input image. At each
position, the surrounding pixel values are multiplied by the
kernel weights and summed together to generate a new output
pixel.

The animation visually illustrates:
1. The input image
2. The convolution kernel
3. The progressively generated output image
4. The kernel's movement across the image

This is useful for understanding the fundamental mechanics
of convolution before discussing optimized methods such as
FFT-based convolution.

Output:
- Displays animation window
- Saves animation as: convolution_animation.gif
------------------------------------------------------------
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

# -----------------------------
# Data
# -----------------------------
image = np.array([
    [20, 24, 11, 12, 16, 19],
    [19, 17, 20, 23, 15, 9],
    [21, 40, 25, 13, 14, 8],
    [9, 18, 8, 6, 11, 22],
    [31, 3, 7, 9, 17, 23],
    [20, 12, 3, 11, 19, 30]
], dtype=float)

# Displayed kernel: easier to read
kernel_display = np.ones((3, 3), dtype=float)

# Actual kernel used in computation
kernel = kernel_display / 9

output = np.full((4, 4), np.nan)
positions = [(i, j) for i in range(4) for j in range(4)]

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_axes([0.05, 0.12, 0.35, 0.68])
ax2 = fig.add_axes([0.46, 0.24, 0.16, 0.40])
ax3 = fig.add_axes([0.70, 0.12, 0.25, 0.68])

input_cmap = ListedColormap(["#BFE8FF"])   # light blue
kernel_cmap = ListedColormap(["#FFE6A7"])  # light yellow
output_cmap = ListedColormap(["#C8F7C5"])  # light green

# -----------------------------
# Drawing helper
# -----------------------------
def draw_matrix(ax, data, title, cmap, show_grid=True, decimals=False):
    ax.clear()

    background = np.zeros_like(data, dtype=float)
    ax.imshow(background, cmap=cmap, vmin=0, vmax=1)

    ax.set_title(
        title,
        fontsize=22,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        pad=22
    )

    ax.set_xticks([])
    ax.set_yticks([])

    rows, cols = data.shape

    if show_grid:
        for x in np.arange(-0.5, cols, 1):
            ax.axvline(x, color="white", linewidth=2.5)
        for y in np.arange(-0.5, rows, 1):
            ax.axhline(y, color="white", linewidth=2.5)

    for i in range(rows):
        for j in range(cols):
            if np.isnan(data[i, j]):
                continue

            if decimals:
                text = f"{data[i, j]:.2f}"
            else:
                text = f"{data[i, j]:.0f}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="semibold",
                fontfamily="DejaVu Sans Mono",
                color="black"
            )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

# -----------------------------
# Animation update
# -----------------------------
def update(frame):
    i, j = positions[frame]

    patch = image[i:i + 3, j:j + 3]
    value = np.sum(patch * kernel)
    output[i, j] = value

    draw_matrix(ax1, image, "Input Image (6×6)", input_cmap, show_grid=True)
    draw_matrix(
        ax2,
        kernel_display,
        "Kernel (3×3)\nscaled by 1/9",
        kernel_cmap,
        show_grid=True,
        decimals=False
    )
    draw_matrix(ax3, output, "Output (4×4)", output_cmap, show_grid=True)

    # Visible sliding window on input
    input_window = Rectangle(
        (j - 0.5, i - 0.5),
        3,
        3,
        fill=False,
        edgecolor="#FF2D55",
        linewidth=7
    )
    ax1.add_patch(input_window)

    # Highlight current output cell
    output_window = Rectangle(
        (j - 0.5, i - 0.5),
        1,
        1,
        fill=False,
        edgecolor="#7B2CBF",
        linewidth=6
    )
    ax3.add_patch(output_window)

    fig.suptitle(
        f"Step {frame + 1}: weighted sum of highlighted region → Output[{i},{j}] = {value:.2f}",
        fontsize=24,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        y=0.98
    )

# -----------------------------
# Build and save animation
# -----------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=len(positions),
    interval=2500,
    repeat=True
)

writer = FFMpegWriter(fps=0.4)
ani.save("convolution_animation.mp4", writer=writer)

plt.show()