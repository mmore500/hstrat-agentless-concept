from IPython.display import Image as IPImage
from IPython.display import display
import imageio
import matplotlib.cm as cm
import numpy as np


class VideoWriter:
    def __init__(self, filename, fps=30.0, fourcc="mp4v"):
        """
        Drop-in replacement for the original VideoWriter, writing a GIF instead.

        Parameters:
            filename (str): Output filename for the GIF.
            fps (float): Frames per second for the GIF.
            fourcc (str): Unused parameter kept for compatibility.
        """
        self.filename = filename
        self.fps = fps
        self.frames = []
        # Use the "jet" colormap by default for grayscale images.
        self.colormap = cm.get_cmap("jet")

    def add(self, img):
        """
        Add a frame to the GIF.

        Converts the image to a NumPy array. If the image is of float type, it
        scales it to the 0-255 range and converts to uint8. If the image is
        grayscale (either a 2D array or a single channel image), the image is
        normalized and the colormap is applied.
        """
        img = np.asarray(img)
        # Convert from float to uint8 if needed.
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(np.clip(img, 0, 1) * 255)
        # Check if the image is grayscale or has a single channel.
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Convert image to float32 for normalization.
            img = img.astype(np.float32)
            norm_img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Apply colormap (ignore alpha) to obtain an RGB image.
            img = (self.colormap(norm_img)[:, :, :3] * 255).astype(np.uint8)
        self.frames.append(img)

    def close(self):
        """
        Save all accumulated frames as a GIF file and clear the frame list.
        """
        if self.frames:
            imageio.mimsave(self.filename, self.frames, fps=self.fps, loop=0)
            self.frames = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def show(self, width=640, height=480):
        """
        Close the writer (if not already closed) and display the saved GIF
        inline.

        The width and height parameters are maintained for compatibility with
        the original API.
        """
        self.close()
        display(IPImage(filename=self.filename))
