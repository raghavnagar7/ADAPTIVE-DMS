import cv2

from src.camera import Camera
from src.detector import FaceDetector


def main():
    print("=" * 60)
    print("ADAPTIVE-DMS")
    print("Adaptive Multimodal Driver State Monitoring System")
    print("=" * 60)
    print("Starting camera...")
    print("Press Q to quit.")

    camera = Camera(
        source=0,
        width=960,
        height=540,
    )

    detector = FaceDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            frame = camera.read()

            if frame is None:
                print("Unable to read frame from camera.")
                break

            results = detector.process(frame)

            # Draw face landmarks.
            frame = detector.draw_landmarks(frame, results)

            # Display status.
            if results.multi_face_landmarks:
                status = "FACE DETECTED"
            else:
                status = "NO FACE"

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "ADAPTIVE-DMS | Press Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("ADAPTIVE-DMS", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()

        print("Camera released.")
        print("ADAPTIVE-DMS stopped.")


if __name__ == "__main__":
    main()