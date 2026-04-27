import numpy as np
import matplotlib.pyplot as plt

def generate_frequency_examples():
    """
    Creates 3 panels:
    Low frequency, Mid frequency, High frequency
    """

    size = 64
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # -----------------------------------
    # LOW FREQUENCY = smooth gradient
    # -----------------------------------
    low_freq = np.zeros((size, size))

    for j in range(size):
        low_freq[:, j] = j / (size - 1)

    axes[0].imshow(low_freq, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Low Frequency", fontsize=16, fontweight="bold")
    axes[0].axis("off")

    # -----------------------------------
    # MID FREQUENCY = edge
    # -----------------------------------
    mid_freq = np.zeros((size, size))
    mid_freq[:, :size // 2] = 0.15
    mid_freq[:, size // 2:] = 0.85

    axes[1].imshow(mid_freq, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mid Frequency", fontsize=16, fontweight="bold")
    axes[1].axis("off")

    # -----------------------------------
    # HIGH FREQUENCY = checkerboard
    # -----------------------------------
    high_freq = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            high_freq[i, j] = 1 if (i + j) % 2 == 0 else 0

    axes[2].imshow(high_freq, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("High Frequency", fontsize=16, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        "What Frequency Means in Images",
        fontsize=22,
        fontweight="bold"
    )

    plt.tight_layout()

    # Save locally in same folder
    plt.savefig("frequency_examples.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    generate_frequency_examples()