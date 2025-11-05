"""
PROSTHEMIND+ Touch-to-Sense AI Module (YOLO-Enhanced)
Real-time object & material classification using YOLOv8 + texture analysis
"""

import cv2
import numpy as np
from collections import deque
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not installed. Install with: pip install ultralytics")

# Material properties database
MATERIAL_DATABASE = {
    'fabric': {
        'grip_strength': 30,
        'color': (100, 200, 255),
        'description': 'Soft, flexible material',
    },
    'metal': {
        'grip_strength': 70,
        'color': (180, 180, 180),
        'description': 'Hard, smooth, conductive',
    },
    'glass': {
        'grip_strength': 85,
        'color': (255, 200, 150),
        'description': 'Fragile, transparent, slippery',
    },
    'wood': {
        'grip_strength': 50,
        'color': (50, 150, 200),
        'description': 'Natural, textured, firm',
    },
    'plastic': {
        'grip_strength': 60,
        'color': (100, 255, 100),
        'description': 'Synthetic, lightweight, varied',
    },
    'paper': {
        'grip_strength': 25,
        'color': (200, 200, 255),
        'description': 'Thin, delicate, crushable',
    },
    'ceramic': {
        'grip_strength': 75,
        'color': (255, 180, 200),
        'description': 'Hard, breakable, smooth',
    }
}

# Object-to-material mapping (based on common object types)
OBJECT_MATERIAL_MAP = {
    # Fabric/cloth items
    'backpack': 'fabric', 'handbag': 'fabric', 'tie': 'fabric', 
    'umbrella': 'fabric', 'suitcase': 'fabric', 'teddy bear': 'fabric',
    
    # Metal items
    'knife': 'metal', 'fork': 'metal', 'spoon': 'metal', 
    'scissors': 'metal', 'refrigerator': 'metal', 'oven': 'metal',
    'sink': 'metal', 'toaster': 'metal',
    
    # Glass items
    'wine glass': 'glass', 'cup': 'glass', 'bottle': 'glass',
    'vase': 'glass', 'mirror': 'glass',
    
    # Wood items
    'chair': 'wood', 'dining table': 'wood', 'bench': 'wood',
    
    # Plastic items
    'cell phone': 'plastic', 'remote': 'plastic', 'keyboard': 'plastic',
    'mouse': 'plastic', 'laptop': 'plastic', 'toothbrush': 'plastic',
    'bottle': 'plastic', 'bowl': 'plastic',
    
    # Paper items
    'book': 'paper', 'newspaper': 'paper',
    
    # Ceramic items
    'potted plant': 'ceramic', 'vase': 'ceramic', 'bowl': 'ceramic',
}

class TouchToSenseAI:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Load YOLO model
        if YOLO_AVAILABLE:
            print("🔄 Loading YOLOv8 model...")
            self.yolo = YOLO('yolov8n.pt')  # Nano model for speed
            print("✅ YOLOv8 loaded successfully!")
        else:
            self.yolo = None
        
        self.current_material = None
        self.current_object = None
        self.current_grip = 0
        self.target_grip = 0
        self.confidence = 0.0
        
        # Smoothing buffer
        self.grip_history = deque(maxlen=15)
        self.material_history = deque(maxlen=10)
        
        # Detection zone
        self.detect_zone = None
        self.detection_box = None
        
    def analyze_advanced_texture(self, roi):
        """Advanced texture analysis for material classification"""
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            
            # 1. Edge-based features
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 2. Texture energy using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_energy = np.var(laplacian)
            
            # 3. Brightness and contrast
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            
            # 4. Color features
            saturation_mean = np.mean(hsv[:, :, 1])
            saturation_std = np.std(hsv[:, :, 1])
            value_mean = np.mean(hsv[:, :, 2])
            
            # 5. Local Binary Pattern approximation
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(gray, kernel, iterations=1)
            dilated = cv2.dilate(gray, kernel, iterations=1)
            lbp_approx = np.std(dilated - eroded)
            
            # 6. Gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))
            
            # 7. Specular highlights (for shiny surfaces)
            _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            highlight_ratio = np.sum(highlights > 0) / highlights.size
            
            return {
                'edge_density': edge_density,
                'texture_energy': texture_energy,
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'saturation_mean': saturation_mean,
                'saturation_std': saturation_std,
                'value_mean': value_mean,
                'lbp_approx': lbp_approx,
                'gradient_mag': gradient_mag,
                'highlight_ratio': highlight_ratio
            }
        except Exception as e:
            print(f"Error in texture analysis: {e}")
            return None
    
    def classify_material_advanced(self, features, detected_object=None):
        """Advanced material classification with better heuristics"""
        if features is None:
            return None, 0.0
        
        # If YOLO detected an object, use the mapping first
        if detected_object and detected_object in OBJECT_MATERIAL_MAP:
            return OBJECT_MATERIAL_MAP[detected_object], 0.85
        
        scores = {}
        
        # FABRIC: Medium edges, low highlights, soft texture
        scores['fabric'] = (
            features['edge_density'] * 3.0 +
            features['lbp_approx'] / 30.0 +
            (1.0 - features['highlight_ratio']) * 2.0 +
            (1.0 - features['brightness_std'] / 100.0) * 0.5
        )
        
        # METAL: High highlights, smooth, high brightness variance
        scores['metal'] = (
            features['highlight_ratio'] * 5.0 +
            features['brightness_std'] / 50.0 +
            (1.0 - features['edge_density']) * 2.0 +
            (1.0 - features['saturation_mean'] / 255.0) * 1.0
        )
        
        # GLASS: Very high highlights, very smooth, low saturation
        scores['glass'] = (
            features['highlight_ratio'] * 6.0 +
            (1.0 - features['edge_density']) * 3.0 +
            (1.0 - features['saturation_mean'] / 255.0) * 1.5 +
            features['value_mean'] / 200.0
        )
        
        # WOOD: High texture, medium edges, natural color variance
        scores['wood'] = (
            features['texture_energy'] / 300.0 +
            features['edge_density'] * 2.0 +
            features['lbp_approx'] / 25.0 +
            (1.0 - features['highlight_ratio']) * 1.5
        )
        
        # PLASTIC: Smooth but not shiny, medium saturation
        scores['plastic'] = (
            features['saturation_mean'] / 150.0 +
            (1.0 - features['texture_energy'] / 300.0) * 1.5 +
            (1.0 - features['edge_density']) * 1.0 +
            features['brightness_mean'] / 200.0
        )
        
        # PAPER: Very high edges, low brightness, rough texture
        scores['paper'] = (
            features['edge_density'] * 4.0 +
            features['lbp_approx'] / 20.0 +
            (1.0 - features['highlight_ratio']) * 2.0 +
            (1.0 - features['brightness_mean'] / 255.0) * 0.5
        )
        
        # CERAMIC: Smooth, medium highlights, uniform
        scores['ceramic'] = (
            (1.0 - features['edge_density']) * 2.0 +
            features['highlight_ratio'] * 2.0 +
            (1.0 - features['texture_energy'] / 300.0) * 1.0 +
            features['brightness_mean'] / 180.0
        )
        
        if not scores:
            return None, 0.0
        
        best_material = max(scores, key=scores.get)
        max_score = scores[best_material]
        
        # Normalize confidence
        total_score = sum(scores.values())
        confidence = (max_score / total_score) if total_score > 0 else 0.0
        
        return best_material, min(confidence, 1.0)
    
    def smooth_material_detection(self, material):
        """Smooth material detection to avoid flickering"""
        if material:
            self.material_history.append(material)
        
        if len(self.material_history) >= 5:
            # Return most common material in recent history
            return max(set(self.material_history), key=self.material_history.count)
        return material
    
    def detect_with_yolo(self, frame):
        """Detect objects using YOLO"""
        if not self.yolo:
            return None, None, None
        
        try:
            results = self.yolo(frame, verbose=False, conf=0.3)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the most confident detection in center region
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                best_box = None
                best_dist = float('inf')
                best_conf = 0
                best_class = None
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Calculate distance from center
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    dist = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
                    
                    if dist < best_dist and conf > 0.3:
                        best_dist = dist
                        best_box = (int(x1), int(y1), int(x2), int(y2))
                        best_conf = float(conf)
                        best_class = results[0].names[cls]
                
                return best_box, best_class, best_conf
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return None, None, None
    
    def smooth_grip_transition(self, target):
        """Smooth grip strength transitions"""
        self.grip_history.append(target)
        return int(np.mean(self.grip_history))
    
    def draw_ui(self, frame):
        """Draw the user interface overlay"""
        h, w = frame.shape[:2]
        
        # Draw detection box if available
        if self.detection_box:
            x1, y1, x2, y2 = self.detection_box
            color = (0, 255, 0) if self.current_material else (255, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw corner markers
            marker_len = 20
            cv2.line(frame, (x1, y1), (x1 + marker_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + marker_len), color, 4)
            cv2.line(frame, (x2, y1), (x2 - marker_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + marker_len), color, 4)
            cv2.line(frame, (x1, y2), (x1 + marker_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - marker_len), color, 4)
            cv2.line(frame, (x2, y2), (x2 - marker_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - marker_len), color, 4)
        
        # Info panel (left side)
        panel_w = 380
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Title
        cv2.putText(frame, "PROSTHEMIND+", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 200, 255), 3)
        cv2.putText(frame, "Touch-to-Sense AI (YOLO)", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        # Status indicator
        status_color = (0, 255, 0) if self.yolo else (0, 0, 255)
        status_text = "ACTIVE" if self.yolo else "NO YOLO"
        cv2.circle(frame, (panel_w - 30, 50), 8, status_color, -1)
        cv2.putText(frame, status_text, (panel_w - 90, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Object detection
        y_offset = 120
        if self.current_object:
            cv2.putText(frame, "OBJECT DETECTED:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(frame, self.current_object.upper(), (20, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            y_offset += 70
        
        # Material detection
        cv2.putText(frame, "MATERIAL DETECTED:", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        if self.current_material:
            mat_info = MATERIAL_DATABASE[self.current_material]
            
            # Material with color indicator
            cv2.rectangle(frame, (20, y_offset + 10), (65, y_offset + 45), 
                         mat_info['color'], -1)
            cv2.rectangle(frame, (20, y_offset + 10), (65, y_offset + 45), 
                         (255, 255, 255), 2)
            cv2.putText(frame, self.current_material.upper(), (75, y_offset + 38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Confidence bar
            y_offset += 65
            cv2.putText(frame, f"Confidence: {int(self.confidence * 100)}%", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (200, 200, 200), 1)
            bar_w = int(330 * self.confidence)
            bar_color = (0, 255, 0) if self.confidence > 0.6 else (0, 200, 255)
            cv2.rectangle(frame, (20, y_offset + 10), (20 + bar_w, y_offset + 28), 
                         bar_color, -1)
            cv2.rectangle(frame, (20, y_offset + 10), (350, y_offset + 28), 
                         (100, 100, 100), 2)
            
            # Grip strength
            y_offset += 55
            cv2.putText(frame, "ADAPTIVE GRIP STRENGTH:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Grip gauge
            gauge_h = 180
            gauge_y = y_offset + 20
            cv2.rectangle(frame, (20, gauge_y), (110, gauge_y + gauge_h), 
                         (40, 40, 40), -1)
            cv2.rectangle(frame, (20, gauge_y), (110, gauge_y + gauge_h), 
                         (100, 100, 100), 2)
            
            # Graduated marks
            for i in range(0, 101, 20):
                mark_y = gauge_y + int(gauge_h * (1 - i/100))
                cv2.line(frame, (110, mark_y), (120, mark_y), (150, 150, 150), 1)
                cv2.putText(frame, str(i), (125, mark_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            grip_h = int(gauge_h * self.current_grip / 100)
            grip_color = (0, 255, 0) if self.current_grip < 50 else (0, 200, 255) if self.current_grip < 80 else (50, 100, 255)
            cv2.rectangle(frame, (20, gauge_y + gauge_h - grip_h), 
                         (110, gauge_y + gauge_h), grip_color, -1)
            
            # Grip percentage
            cv2.putText(frame, f"{self.current_grip}%", (35, gauge_y + gauge_h + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Material properties
            y_offset = gauge_y + gauge_h + 60
            cv2.putText(frame, "PROPERTIES:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(frame, mat_info['description'], (20, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Safety indicator
            y_offset += 50
            if self.current_grip > 80:
                cv2.putText(frame, "⚠ HIGH FORCE", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)
            elif self.current_grip < 40:
                cv2.putText(frame, "✓ GENTLE GRIP", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "✓ BALANCED", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        else:
            cv2.putText(frame, "No object detected", (20, y_offset + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)
            cv2.putText(frame, "Show object to camera", (20, y_offset + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        
        # Instructions
        cv2.putText(frame, "Controls: Q=Quit | R=Reset | S=Screenshot", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.45, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("🤖 PROSTHEMIND+ Touch-to-Sense AI (YOLO-Enhanced)")
        print("="*60)
        if self.yolo:
            print("✅ YOLOv8 loaded - Object detection enabled")
        else:
            print("⚠️  YOLOv8 not available - Using texture analysis only")
        print("📹 Show objects to the camera for material detection")
        print("⚡ Controls: Q=Quit | R=Reset | S=Screenshot")
        print("="*60 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Run detection every 3 frames for performance
            if frame_count % 3 == 0:
                # YOLO detection
                detection_box, detected_object, obj_conf = self.detect_with_yolo(frame)
                
                if detection_box:
                    self.detection_box = detection_box
                    self.current_object = detected_object
                    
                    # Extract ROI for texture analysis
                    x1, y1, x2, y2 = detection_box
                    roi = frame[y1:y2, x1:x2]
                    
                    # Analyze texture
                    features = self.analyze_advanced_texture(roi)
                    material, confidence = self.classify_material_advanced(features, detected_object)
                    
                    if material and confidence > 0.4:
                        # Smooth material detection
                        material = self.smooth_material_detection(material)
                        
                        self.current_material = material
                        self.confidence = confidence
                        self.target_grip = MATERIAL_DATABASE[material]['grip_strength']
                        self.current_grip = self.smooth_grip_transition(self.target_grip)
                    else:
                        self.confidence = max(0, self.confidence - 0.03)
                        if self.confidence < 0.2:
                            self.current_material = None
                else:
                    # Fade out
                    self.confidence = max(0, self.confidence - 0.05)
                    if self.confidence < 0.1:
                        self.current_material = None
                        self.current_object = None
                        self.detection_box = None
                        self.current_grip = self.smooth_grip_transition(0)
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            # Display
            cv2.imshow('PROSTHEMIND+ Touch-to-Sense AI', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_material = None
                self.current_object = None
                self.detection_box = None
                self.current_grip = 0
                self.confidence = 0.0
                self.grip_history.clear()
                self.material_history.clear()
                print("🔄 Reset completed")
            elif key == ord('s'):
                filename = f"prosthemind_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Touch-to-Sense AI terminated\n")

if __name__ == "__main__":
    # Check for YOLO
    if not YOLO_AVAILABLE:
        print("\n" + "="*60)
        print("⚠️  WARNING: YOLOv8 not installed!")
        print("="*60)
        print("Install it with:")
        print("   pip install ultralytics")
        print("\nThe program will work with texture analysis only.")
        print("="*60 + "\n")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    ai = TouchToSenseAI()
    ai.run()
