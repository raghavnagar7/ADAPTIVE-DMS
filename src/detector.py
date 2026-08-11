import cv2
import mediapipe as mp


class FaceDetector:
    """MediaPipe Face Mesh detector for ADAPTIVE-DMS."""

    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        """
        Process a BGR OpenCV frame.

        Returns:
            MediaPipe results object.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        return results

    def draw_landmarks(self, frame, results):
        """Draw detected face landmarks on the frame."""

        if not results.multi_face_landmarks:
            return frame

        for face_landmarks in results.multi_face_landmarks:
            self.mp_face_mesh.FACEMESH_TESSELATION

            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style()
                ),
            )

        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()