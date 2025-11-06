from ultralytics import YOLO
import cv2
import numpy as np
import time

class EnvironmentDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            raise RuntimeError("Error: Camera not accessible")
        self.last_time = time.time()
        print("🔍 Environment detector initialized.")

    def detect(self):
        ret, frame = self.cam.read()
        if not ret:
            return "unknown"

        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean()

        # Detect diagonal lines for stairs
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
        stair_detected = False
        if lines is not None:
            angles = [theta for rho, theta in lines[:, 0]]
            avg_angle = np.mean(angles)
            if 0.4 < avg_angle < 1.2:
                stair_detected = True

        # Determine context
        if stair_detected:
            context = "stairs"
            color = (0, 0, 255)
            label = "🪜 STAIRS DETECTED"
        elif edge_density < 20:
            context = "flat"
            color = (0, 255, 0)
            label = "🟩 FLAT SURFACE"
        else:
            context = "obstacle"
            color = (0, 255, 255)
            label = "⚠️ OBSTACLE DETECTED"

        # Overlay label
        cv2.putText(annotated_frame, label, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Show the frame
        cv2.imshow("PROSTHEMIND+ Context View", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        return context

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()
