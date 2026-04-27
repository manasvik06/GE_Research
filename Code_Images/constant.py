"""
Constant Image -> Frequency Domain using FFT
Author: Manasvi Khandelwal

Shows that a constant image has only one frequency component:
the DC component, which represents average brightness.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------
# Create constant image
# -----------------------------
N = 8
value = 150
image = np.full((N, N), value, dtype=float)

# -----------------------------
# Compute 2D FFT
# -----------------------------
fft_img = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft_img)
magnitude = np.abs(fft_shifted).astype(int)

# For an 8x8 image filled with 150:
# center value = 150 * 8 * 8 = 9600

# -----------------------------
# Drawing helper
# -----------------------------
def draw_grid(ax, data, title, cmap, text_color="black"):
    ax.imshow(data, cmap=cmap)

    ax.set_title(title, fontsize=22, fontweight="bold", pad=18)
    ax.set_xticks([])
    ax.set_yticks([])

    # Grid lines
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(N):
        for j in range(N):
            ax.text(
                j, i, str(int(data[i, j])),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color
            )

# -----------------------------
# Frequency display colors
# -----------------------------
# Make all zero cells dark gray and center cell bright yellow
freq_display = np.zeros((N, N))
freq_display[N // 2, N // 2] = 1

freq_cmap = ListedColormap(["#111111", "#FFD84D"])

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

draw_grid(axes[0], image, "Spatial Domain", cmap="Blues", text_color="black")

axes[1].imshow(freq_display, cmap=freq_cmap, vmin=0, vmax=1)
axes[1].set_title("Frequency Domain", fontsize=22, fontweight="bold", pad=18)
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[1].set_xticks(np.arange(-0.5, N, 1), minor=True)
axes[1].set_yticks(np.arange(-0.5, N, 1), minor=True)
axes[1].grid(which="minor", color="white", linewidth=2)
axes[1].tick_params(which="minor", bottom=False, left=False)

for i in range(N):
    for j in range(N):
        val = int(magnitude[i, j])
        color = "black" if val != 0 else "white"
        axes[1].text(
            j, i, str(val),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color
        )

# Highlight DC component
center = N // 2
axes[1].scatter(
    center,
    center,
    s=1000,
    facecolors="none",
    edgecolors="#00E5FF",
    linewidths=4
)

plt.suptitle(
    "Constant Image: Only the DC Component Exists",
    fontsize=24,
    fontweight="bold",
    y=0.98
)

plt.tight_layout()
plt.savefig("constant_image_fft_readable.png", dpi=300, bbox_inches="tight")
plt.show()