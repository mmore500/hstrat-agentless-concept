import imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_gif_frames(gif_path):
    """
    Opens a GIF file, samples 32 frames evenly across its duration,
    and plots them in an 8 wide by 4 tall grid using matplotlib.

    Parameters:
    gif_path (str): Path to the GIF file.
    """
    # Read all frames from the GIF
    frames = imageio.mimread(gif_path)
    n_frames = len(frames)

    # Create 32 evenly spaced indices across the total number of frames
    # If there are fewer than 32 frames, some indices will repeat
    indices = np.linspace(0, n_frames - 1, 32, dtype=int)

    # Create an 8x4 subplot grid
    fig, axes = plt.subplots(4, 8, figsize=(8, 4))

    # Loop over the grid and plot each sampled frame
    for ax, idx in zip(axes.flatten(), indices):
        ax.imshow(frames[idx])
        ax.axis("off")  # Hide axes for a cleaner look

    plt.tight_layout()
    plt.show()
