import time

import cv2

from src.camera import Camera
from src.detector import FaceDetector
from src.metrics import DriverMetrics


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

    metrics = DriverMetrics(
        ear_threshold=0.21,
        mar_threshold=0.60,
        perclos_window_seconds=30,
    )

    try:
        while True:
            frame = camera.read()

            if frame is None:
                print("Unable to read frame from camera.")
                break

            results = detector.process(frame)

            face_detected = bool(
                results.multi_face_landmarks
            )

            if face_detected:

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                landmarks = face_landmarks.landmark

                # Calculate driver-state metrics.
                values = metrics.calculate(
                    landmarks,
                    timestamp=time.time(),
                )

                # Draw face landmarks.
                frame = detector.draw_landmarks(
                    frame,
                    results,
                )

                # Display metrics.
                cv2.putText(
                    frame,
                    f"EAR: {values['ear']:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"MAR: {values['mar']:.3f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"PERCLOS: {values['perclos']:.3f}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Blinks: {values['blink_count']}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                status = "EYES CLOSED" if values["eyes_closed"] else "EYES OPEN"

                cv2.putText(
                    frame,
                    status,
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if values["yawning"]:
                    cv2.putText(
                        frame,
                        "YAWNING",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

            else:
                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
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

            cv2.imshow(
                "ADAPTIVE-DMS",
                frame,
            )

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