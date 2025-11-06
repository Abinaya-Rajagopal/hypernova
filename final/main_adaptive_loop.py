import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO

# -----------------------------
# 1. MOTION MODEL
# -----------------------------
class MotionModel(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=3):
        super(MotionModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# -----------------------------
# 2. ADAPTIVE CONTROLLER
# -----------------------------
class AdaptiveController:
    def __init__(self, gain=0.3):
        self.gain = gain
        self.current_state = np.zeros(3)

    def apply(self, predicted):
        self.current_state += self.gain * (predicted - self.current_state)
        return self.current_state


# -----------------------------
# 3. ENVIRONMENT DETECTOR (YOLO + Edges)
# -----------------------------
class EnvironmentDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.last_time = time.time()

    def detect_context(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean()

        # Detect line angles (for stairs)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
        stair_detected = False
        if lines is not None:
            angles = [theta for rho, theta in lines[:, 0]]
            avg_angle = np.mean(angles)
            if 0.4 < avg_angle < 1.2:
                stair_detected = True

        if stair_detected:
            context = "STAIRS"
        elif edge_density < 20:
            context = "FLAT"
        else:
            context = "OBSTACLE"

        return context


# -----------------------------
# 4. DRAW 2D LIMB SIMULATOR
# -----------------------------
def draw_limb(joints, environment):
    # joints = [shoulder_angle, elbow_angle, wrist_angle]
    base = (250, 400)
    upper_len, fore_len, hand_len = 120, 100, 60

    # Calculate limb positions
    shoulder = base
    elbow = (
        int(shoulder[0] + upper_len * np.cos(joints[0])),
        int(shoulder[1] - upper_len * np.sin(joints[0]))
    )
    wrist = (
        int(elbow[0] + fore_len * np.cos(joints[1])),
        int(elbow[1] - fore_len * np.sin(joints[1]))
    )
    hand = (
        int(wrist[0] + hand_len * np.cos(joints[2])),
        int(wrist[1] - hand_len * np.sin(joints[2]))
    )

    limb_canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
    color = (0, 255, 0) if environment == "FLAT" else (0, 0, 255) if environment == "STAIRS" else (0, 255, 255)

    cv2.line(limb_canvas, shoulder, elbow, color, 8)
    cv2.line(limb_canvas, elbow, wrist, color, 6)
    cv2.line(limb_canvas, wrist, hand, color, 4)
    cv2.circle(limb_canvas, shoulder, 8, (100, 100, 100), -1)
    cv2.circle(limb_canvas, elbow, 8, (100, 100, 100), -1)
    cv2.circle(limb_canvas, wrist, 8, (100, 100, 100), -1)
    cv2.circle(limb_canvas, hand, 8, (0, 0, 0), -1)

    cv2.putText(limb_canvas, f"Environment: {environment}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return limb_canvas


# -----------------------------
# 5. MAIN ADAPTIVE LOOP
# -----------------------------
def main():
    motion_model = MotionModel()
    adaptive_controller = AdaptiveController()
    env_detector = EnvironmentDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("🧠 Starting ProstheMind+ Adaptive Loop — Press 'q' to quit.")
    sensor_window = torch.zeros((1, 10, 6))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect environment
        context = env_detector.detect_context(frame)

        # Context influence vectors
        context_vec = {
            "FLAT": np.array([0.0, 0.0, 0.0]),
            "STAIRS": np.array([0.6, 0.9, 0.3]),
            "OBSTACLE": np.array([-0.4, 0.2, -0.3])
        }[context]

        # Change gain adaptively
        adaptive_controller.gain = {"FLAT": 0.3, "STAIRS": 0.5, "OBSTACLE": 0.2}[context]

        # Generate fake sensor input
        new_sensor = torch.tensor(np.random.randn(1, 1, 6), dtype=torch.float32)
        sensor_window = torch.cat([sensor_window[:, 1:, :], new_sensor], dim=1)

        with torch.no_grad():
            predicted = motion_model(sensor_window).numpy().flatten()

        # Adjust with environment influence
        predicted += context_vec

        # Apply adaptive control
        applied = adaptive_controller.apply(predicted)

        # Print live feedback
        print(f"\nEnvironment: {context}")
        print(f"Predicted: {predicted.round(5)}")
        print(f"Applied:   {applied.round(5)}")

        # Draw limb
        limb_canvas = draw_limb(applied, context)

        # Combine camera feed and limb simulator side by side
        frame_resized = cv2.resize(frame, (500, 500))
        combined = np.hstack((frame_resized, limb_canvas))

        cv2.imshow("ProstheMind+ — Context + Adaptive Limb", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 System stopped safely.")
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
if __name__ == "__main__":
    main()
