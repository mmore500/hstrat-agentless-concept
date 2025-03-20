import cv2
import numpy as np


class VideoWriter:
    def __init__(self, filename, fps=30.0, fourcc="mp4v"):
        self.filename = filename
        self.fps = fps
        self.writer = None
        # Define the codec using OpenCV's VideoWriter_fourcc function.
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

    def add(self, img):
        img = np.asarray(img)
        # Convert image from float to uint8 if needed.
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(np.clip(img, 0, 1) * 255)
        # If the image is grayscale, convert it to a 3-channel BGR image.
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Initialize the writer when the first frame is added.
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(
                self.filename, self.fourcc, self.fps, (w, h)
            )
        self.writer.write(img)

    def close(self):
        if self.writer:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def show(self, width=640, height=480):
        self.close()
        # Display the saved video in the notebook.
        print("bypassing output")
