import cv2


class Camera:
    """Handles webcam input for ADAPTIVE-DMS."""

    def __init__(self, source=0, width=960, height=540):
        self.source = source
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera with source {self.source}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Read one frame from the camera."""
        success, frame = self.cap.read()

        if not success:
            return None

        # Mirror the webcam for a natural user experience.
        frame = cv2.flip(frame, 1)

        return frame

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()