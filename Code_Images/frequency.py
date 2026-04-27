"""
Spatial vs Frequency Domain Grid Visualization
Author: Manasvi Khandelwal

Creates an 8x8 image in the spatial domain and shows its
corresponding 8x8 frequency-domain representation.

Output:
spatial_frequency_grid.png
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Create simple 8x8 image
# -----------------------------
N = 8

# Left half: slow gradient
left = np.tile(np.array([70, 95, 120, 145]), (N, 1))

# Right half: rapid alternating pattern
right = np.tile(np.array([230, 30, 230, 30]), (N, 1))

image = np.hstack([left, right]).astype(float)

# Add small bright structure
image[3, 3] = 255
image[4, 4] = 255

# -----------------------------
# Compute frequency domain
# -----------------------------
fft_img = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft_img)

# FFT values are complex, so use magnitude
magnitude = np.abs(fft_shifted)

# Log scale makes values easier to visualize
freq_display = np.log1p(magnitude)
freq_values = np.round(freq_display).astype(int)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1, ax2 = axes

def draw_grid(ax, data, title, cmap):
    ax.imshow(data, cmap=cmap)

    ax.set_title(title, fontsize=24, fontweight="bold", pad=20)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels([f"c{i}" for i in range(data.shape[1])], fontsize=9)
    ax.set_yticklabels([f"r{i}" for i in range(data.shape[0])], fontsize=9)

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)

    ax.tick_params(which="minor", bottom=False, left=False)

    max_val = np.max(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]

            # choose readable text color
            if val > max_val * 0.55:
                color = "white"
            else:
                color = "black"

            ax.text(
                j, i, f"{int(val)}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color
            )

# Spatial domain grid
draw_grid(ax1, image, "Spatial Domain", cmap="Blues")

# Frequency domain grid
draw_grid(ax2, freq_values, "Frequency Domain", cmap="hot")

# Mark center low-frequency / DC component
center = N // 2
ax2.scatter(
    center,
    center,
    s=1000,
    facecolors="none",
    edgecolors="cyan",
    linewidths=4
)

ax2.text(
    center,
    center + 0.85,
    "low frequency\n/ average brightness",
    color="cyan",
    fontsize=11,
    fontweight="bold",
    ha="center"
)

plt.suptitle(
    "Same 8×8 Image — Two Representations",
    fontsize=26,
    fontweight="bold",
    y=1.02
)

plt.tight_layout()

plt.savefig("spatial_frequency_grid.png", dpi=300, bbox_inches="tight")
plt.show()