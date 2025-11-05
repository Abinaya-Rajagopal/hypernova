from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("Error: Camera not accessible")
    exit()

print("🔍 Starting Stair vs Flat Surface Detection... Press 'q' to quit")
last_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ Frame grab failed, retrying...")
        continue

    # Run YOLO inference
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    # Detect dominant line angles (for stairs detection)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    stair_detected = False
    if lines is not None:
        angles = [theta for rho, theta in lines[:, 0]]
        avg_angle = np.mean(angles)
        # If many diagonal lines, likely stairs
        if 0.4 < avg_angle < 1.2:  
            stair_detected = True

    # Display every 3 seconds
    if time.time() - last_time > 3:
        if stair_detected:
            print("Detected Environment: 🪜 STAIRS — Adapting to climb mode.")
            cv2.putText(annotated_frame, "STAIRS DETECTED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif edge_density < 20:
            print("Detected Environment: 🟩 FLAT SURFACE — Normal gait mode.")
            cv2.putText(annotated_frame, "FLAT SURFACE", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            print("Detected Environment: ⚠️ OBSTACLE — Adjusting balance mode.")
            cv2.putText(annotated_frame, "OBSTACLE", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        last_time = time.time()

    # Show annotated feed
    cv2.imshow("Stair vs Flat Surface Detection", annotated_frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
