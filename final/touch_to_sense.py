"""
PROSTHEMIND+ Integrated System - ENHANCED WITH OBJECT & MATERIAL INSIGHTS
Touch-to-Sense AI + Pain Feedback + Haptic Alerts + Smart Object & Material Insights
Real-time object detection with contextual feedback and handling recommendations
Intelligent material-specific insights (e.g., Granite, Marble, Glass, Metal, etc.)
"""

import cv2
import numpy as np
from collections import deque
import time
import threading
import math
import random

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not installed. Install with: pip install ultralytics")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  Pygame not installed. Install with: pip install pygame")

# Material properties database with pain factors
MATERIAL_DATABASE = {
    'fabric': {
        'grip_strength': 30,
        'color': (100, 200, 255),
        'description': 'Soft, flexible material',
        'haptic_intensity': 'LOW',
        'sound_freq': 300,
        'pain_factor': 0.2
    },
    'metal': {
        'grip_strength': 70,
        'color': (180, 180, 180),
        'description': 'Hard, smooth, conductive',
        'haptic_intensity': 'HIGH',
        'sound_freq': 600,
        'pain_factor': 0.7
    },
    'glass': {
        'grip_strength': 85,
        'color': (255, 200, 150),
        'description': 'Fragile, transparent, slippery',
        'haptic_intensity': 'VERY HIGH',
        'sound_freq': 800,
        'pain_factor': 0.9
    },
    'wood': {
        'grip_strength': 50,
        'color': (50, 150, 200),
        'description': 'Natural, textured, firm',
        'haptic_intensity': 'MEDIUM',
        'sound_freq': 400,
        'pain_factor': 0.4
    },
    'plastic': {
        'grip_strength': 60,
        'color': (100, 255, 100),
        'description': 'Synthetic, lightweight, varied',
        'haptic_intensity': 'MEDIUM',
        'sound_freq': 450,
        'pain_factor': 0.5
    },
    'paper': {
        'grip_strength': 25,
        'color': (200, 200, 255),
        'description': 'Thin, delicate, crushable',
        'haptic_intensity': 'VERY LOW',
        'sound_freq': 250,
        'pain_factor': 0.1
    },
    'ceramic': {
        'grip_strength': 75,
        'color': (255, 180, 200),
        'description': 'Hard, breakable, smooth',
        'haptic_intensity': 'HIGH',
        'sound_freq': 700,
        'pain_factor': 0.8
    },
    'stone': {
        'grip_strength': 80,
        'color': (150, 150, 150),
        'description': 'Very hard, heavy, durable (granite, marble)',
        'haptic_intensity': 'VERY HIGH',
        'sound_freq': 750,
        'pain_factor': 0.85
    }
}

# Object-to-material mapping
OBJECT_MATERIAL_MAP = {
    'backpack': 'fabric', 'handbag': 'fabric', 'tie': 'fabric', 
    'umbrella': 'fabric', 'suitcase': 'fabric', 'teddy bear': 'fabric',
    'knife': 'metal', 'fork': 'metal', 'spoon': 'metal', 
    'scissors': 'metal', 'refrigerator': 'metal', 'oven': 'metal',
    'sink': 'metal', 'toaster': 'metal',
    'wine glass': 'glass', 'cup': 'glass', 'bottle': 'glass',
    'vase': 'glass', 'mirror': 'glass',
    'chair': 'wood', 'dining table': 'wood', 'bench': 'wood',
    'cell phone': 'plastic', 'remote': 'plastic', 'keyboard': 'plastic',
    'mouse': 'plastic', 'laptop': 'plastic', 'toothbrush': 'plastic',
    'bowl': 'plastic',
    'book': 'paper', 'newspaper': 'paper',
    'potted plant': 'ceramic', 'vase': 'ceramic',
    # Stone/rock objects would map to stone
}

# Object-specific insights and recommendations
OBJECT_INSIGHTS = {
    # Fragile items
    'wine glass': {
        'insight': '⚠️ Fragile - Handle with extreme care to avoid breakage',
        'warning_level': 'high',
        'icon': '🍷',
        'tips': ['Use gentle grip', 'Support from bottom', 'Avoid stem pressure']
    },
    'cup': {
        'insight': '☕ Check if hot - May contain beverages, test temperature first',
        'warning_level': 'medium',
        'icon': '☕',
        'tips': ['Test temperature', 'Grip handle if available', 'Moderate pressure']
    },
    'bottle': {
        'insight': '💧 Check cap status - Ensure lid is secure before lifting',
        'warning_level': 'medium',
        'icon': '🧴',
        'tips': ['Verify cap is closed', 'Grip at center', 'Steady lift']
    },
    'vase': {
        'insight': '🏺 Decorative & fragile - High risk of dropping, use two points',
        'warning_level': 'high',
        'icon': '🏺',
        'tips': ['Support base and sides', 'Slow movements', 'Check for water inside']
    },
    
    # Sharp objects
    'knife': {
        'insight': '⚠️ SHARP EDGE - Grip handle only, blade poses cut risk',
        'warning_level': 'critical',
        'icon': '🔪',
        'tips': ['Handle grip only', 'Blade away from body', 'Secure grip needed']
    },
    'scissors': {
        'insight': '✂️ Sharp points - Handle with care, check blade position',
        'warning_level': 'high',
        'icon': '✂️',
        'tips': ['Closed position best', 'Grip handles', 'Point awareness']
    },
    'fork': {
        'insight': '🍴 Pointed tines - Be aware of prong direction',
        'warning_level': 'medium',
        'icon': '🍴',
        'tips': ['Tines down when moving', 'Firm but gentle', 'Watch orientation']
    },
    
    # Electronic devices
    'cell phone': {
        'insight': '📱 Sensitive screen - Avoid excessive pressure on display',
        'warning_level': 'medium',
        'icon': '📱',
        'tips': ['Edge grip preferred', 'Avoid screen pressure', 'Check for case']
    },
    'laptop': {
        'insight': '💻 Heavy & valuable - Use two contact points for stability',
        'warning_level': 'medium',
        'icon': '💻',
        'tips': ['Support with both sides', 'Avoid screen area', 'Check if closed']
    },
    'remote': {
        'insight': '📺 Electronic device - Gentle grip to avoid button presses',
        'warning_level': 'low',
        'icon': '📺',
        'tips': ['Side grip', 'Light pressure', 'Avoid button area']
    },
    'keyboard': {
        'insight': '⌨️ Delicate keys - Lift from edges, not key surface',
        'warning_level': 'medium',
        'icon': '⌨️',
        'tips': ['Edge grip only', 'Avoid key pressure', 'Check for cables']
    },
    'mouse': {
        'insight': '🖱️ Small & light - Gentle grip, easy to over-squeeze',
        'warning_level': 'low',
        'icon': '🖱️',
        'tips': ['Light touch', 'Palm support', 'Avoid click buttons']
    },
    
    # Food items
    'apple': {
        'insight': '🍎 Bruises easily - Moderate grip to avoid damaging fruit',
        'warning_level': 'low',
        'icon': '🍎',
        'tips': ['Gentle pressure', 'Avoid fingernail marks', 'Even distribution']
    },
    'banana': {
        'insight': '🍌 Very soft - Minimal pressure needed, bruises instantly',
        'warning_level': 'low',
        'icon': '🍌',
        'tips': ['Very gentle', 'Support full length', 'Minimal squeeze']
    },
    'orange': {
        'insight': '🍊 Firm but can burst - Balanced grip to avoid juice leakage',
        'warning_level': 'low',
        'icon': '🍊',
        'tips': ['Medium pressure', 'Even grip', 'Watch for soft spots']
    },
    'sandwich': {
        'insight': '🥪 Compressible - Light grip to maintain structure',
        'warning_level': 'low',
        'icon': '🥪',
        'tips': ['Gentle hold', 'Support from bottom', 'Avoid crushing']
    },
    'hot dog': {
        'insight': '🌭 Soft & messy - May have condiments, gentle handling',
        'warning_level': 'low',
        'icon': '🌭',
        'tips': ['Light grip', 'Horizontal hold', 'Mind the filling']
    },
    'pizza': {
        'insight': '🍕 Hot & floppy - Support underneath, check temperature',
        'warning_level': 'medium',
        'icon': '🍕',
        'tips': ['Bottom support', 'Test heat', 'Fold technique']
    },
    'cake': {
        'insight': '🍰 Very delicate - Extremely gentle or use utensil',
        'warning_level': 'high',
        'icon': '🍰',
        'tips': ['Plate grip preferred', 'Minimal contact', 'Support base']
    },
    
    # Tools & utensils
    'spoon': {
        'insight': '🥄 May contain liquid - Keep level to avoid spills',
        'warning_level': 'low',
        'icon': '🥄',
        'tips': ['Handle grip', 'Keep level', 'Smooth movements']
    },
    'toothbrush': {
        'insight': '🪥 Hygiene item - Clean grip area, bristles are delicate',
        'warning_level': 'low',
        'icon': '🪥',
        'tips': ['Handle only', 'Light grip', 'Avoid bristle area']
    },
    'umbrella': {
        'insight': '☂️ Extendable - Check if open/closed, mind the mechanism',
        'warning_level': 'medium',
        'icon': '☂️',
        'tips': ['Handle grip', 'Check state', 'Avoid trigger area']
    },
    
    # Containers
    'bowl': {
        'insight': '🥣 May contain items - Check for contents before lifting',
        'warning_level': 'medium',
        'icon': '🥣',
        'tips': ['Bottom support', 'Check contents', 'Level movement']
    },
    'backpack': {
        'insight': '🎒 Variable weight - Test weight first, may be heavy',
        'warning_level': 'medium',
        'icon': '🎒',
        'tips': ['Test weight', 'Strap grip', 'Lift preparation']
    },
    'handbag': {
        'insight': '👜 Contains valuables - Secure grip, check strap strength',
        'warning_level': 'medium',
        'icon': '👜',
        'tips': ['Handle secure', 'Test weight', 'Strap check']
    },
    'suitcase': {
        'insight': '🧳 Potentially heavy - Use handle, test weight before lifting',
        'warning_level': 'high',
        'icon': '🧳',
        'tips': ['Handle only', 'Test weight first', 'Consider wheels']
    },
    
    # Reading materials
    'book': {
        'insight': '📚 Pages are delicate - Grip cover edges, avoid page damage',
        'warning_level': 'low',
        'icon': '📚',
        'tips': ['Edge grip', 'Support spine', 'Gentle handling']
    },
    'newspaper': {
        'insight': '📰 Very thin paper - Extremely light touch required',
        'warning_level': 'low',
        'icon': '📰',
        'tips': ['Minimal pressure', 'Flat support', 'Edge handling']
    },
    
    # Furniture
    'chair': {
        'insight': '🪑 Heavy furniture - Check stability before moving',
        'warning_level': 'high',
        'icon': '🪑',
        'tips': ['Two-point grip', 'Test weight', 'Lift with care']
    },
    'potted plant': {
        'insight': '🪴 Living organism + soil - May be wet, check weight',
        'warning_level': 'medium',
        'icon': '🪴',
        'tips': ['Base support', 'Check moisture', 'Soil awareness']
    },
    
    # Sports equipment
    'sports ball': {
        'insight': '⚽ Inflatable - Check pressure, adjust grip accordingly',
        'warning_level': 'low',
        'icon': '⚽',
        'tips': ['Firm grip OK', 'Check inflation', 'Bounce test']
    },
    'baseball bat': {
        'insight': '⚾ Hard wood/metal - Grip handle firmly, mind the weight',
        'warning_level': 'medium',
        'icon': '⚾',
        'tips': ['Handle grip', 'Weight awareness', 'Secure hold']
    },
    'tennis racket': {
        'insight': '🎾 Stringed surface - Avoid string contact, grip handle',
        'warning_level': 'medium',
        'icon': '🎾',
        'tips': ['Handle only', 'Avoid strings', 'Check head weight']
    },
    
    # Default for unspecified objects
    'default': {
        'insight': '👁️ Analyzing object - Adjusting grip based on detection',
        'warning_level': 'low',
        'icon': '🔍',
        'tips': ['Assess carefully', 'Test weight', 'Start gentle']
    }
}

# Material-specific insights and recommendations
MATERIAL_INSIGHTS = {
    'granite': {
        'insight': '🪨 Granite - Extremely hard, heavy material. Very durable but requires strong grip',
        'warning_level': 'high',
        'icon': '🪨',
        'properties': ['Hardness: 9/10', 'Weight: Heavy', 'Surface: Rough/Polished'],
        'tips': ['Use firm grip (70-85%)', 'Check weight before lifting', 'Watch for sharp edges'],
        'handling': 'Granite is extremely dense and heavy. Requires substantial grip strength but beware of sharp corners. Surface may be polished (slippery) or rough (better grip).',
        'safety': 'Risk of crushing injury if dropped. Sharp edges can cause cuts. Support from bottom when lifting large pieces.'
    },
    'marble': {
        'insight': '🗿 Marble - Smooth, polished surface. Heavy but can be fragile',
        'warning_level': 'high',
        'icon': '🗿',
        'properties': ['Hardness: 7/10', 'Weight: Heavy', 'Surface: Polished/Smooth'],
        'tips': ['Moderate-firm grip (60-75%)', 'Slippery when wet', 'Handle with care - can crack'],
        'handling': 'Marble is heavy with a smooth surface that can be slippery. Requires balanced grip - not too tight to avoid strain, but firm enough to prevent drops.',
        'safety': 'High weight risk. Polished surface is slippery. May have natural fractures that weaken structure.'
    },
    'metal': {
        'insight': '⚙️ Metal - Conductive, may be hot or cold. Sharp edges possible',
        'warning_level': 'medium',
        'icon': '⚙️',
        'properties': ['Hardness: 8/10', 'Weight: Variable', 'Surface: Smooth/Textured'],
        'tips': ['Test temperature first', 'Check for sharp edges', 'Moderate grip (50-70%)'],
        'handling': 'Metal objects can be hot, cold, or have sharp edges. Test temperature before full grip. Adjust grip strength based on object size and weight.',
        'safety': 'Temperature extremes possible. Sharp edges can cut. Electrical conductivity risk if powered.'
    },
    'glass': {
        'insight': '💎 Glass - Extremely fragile, sharp when broken. Requires gentle precision',
        'warning_level': 'critical',
        'icon': '💎',
        'properties': ['Hardness: 6/10', 'Weight: Light-Medium', 'Surface: Smooth/Slippery'],
        'tips': ['Very gentle grip (30-50%)', 'Support from bottom', 'Avoid pressure points'],
        'handling': 'Glass requires extremely careful handling. Use gentle, distributed pressure. Support from multiple points. Avoid concentrated pressure that can cause breakage.',
        'safety': 'CRITICAL: Breaks into sharp fragments. High risk of injury. Always handle with extreme care.'
    },
    'wood': {
        'insight': '🪵 Wood - Natural texture provides grip. Watch for splinters',
        'warning_level': 'low',
        'icon': '🪵',
        'properties': ['Hardness: 4/10', 'Weight: Light-Medium', 'Surface: Textured'],
        'tips': ['Moderate grip (45-60%)', 'Check for splinters', 'Natural grip advantage'],
        'handling': 'Wood provides natural grip due to texture. Moderate grip strength usually sufficient. Check for rough edges or splinters before handling.',
        'safety': 'Splinter risk on rough surfaces. May have sharp edges from cuts. Generally safe to handle.'
    },
    'fabric': {
        'insight': '🧵 Fabric - Soft, flexible. Low grip requirement but can slip',
        'warning_level': 'low',
        'icon': '🧵',
        'properties': ['Hardness: 1/10', 'Weight: Light', 'Surface: Soft/Textured'],
        'tips': ['Light grip (25-40%)', 'May bunch up', 'Check for contents'],
        'handling': 'Fabric requires minimal grip strength. However, smooth fabrics can slip. Check for contents inside bags or containers.',
        'safety': 'Low risk. May contain items inside. Watch for zippers or sharp hardware.'
    },
    'plastic': {
        'insight': '🪣 Plastic - Lightweight, varies in hardness. Can be slippery',
        'warning_level': 'low',
        'icon': '🪣',
        'properties': ['Hardness: 3-5/10', 'Weight: Light', 'Surface: Smooth/Textured'],
        'tips': ['Light-moderate grip (40-60%)', 'Check for cracks', 'May be slippery'],
        'handling': 'Plastic is generally lightweight but can be slippery. Adjust grip based on object size. Check for cracks or weak points.',
        'safety': 'Low to medium risk. May have sharp edges from molding. Can crack under pressure.'
    },
    'ceramic': {
        'insight': '🏺 Ceramic - Hard but brittle. Can break from impact or pressure',
        'warning_level': 'high',
        'icon': '🏺',
        'properties': ['Hardness: 7/10', 'Weight: Medium', 'Surface: Smooth/Glazed'],
        'tips': ['Firm but careful grip (60-75%)', 'Support from bottom', 'Avoid impacts'],
        'handling': 'Ceramic requires firm grip to prevent drops, but avoid excessive pressure that can cause cracks. Support from base is crucial.',
        'safety': 'High breakage risk. Sharp fragments when broken. Glazed surfaces can be slippery.'
    },
    'paper': {
        'insight': '📄 Paper - Extremely delicate. Minimal pressure required',
        'warning_level': 'low',
        'icon': '📄',
        'properties': ['Hardness: 1/10', 'Weight: Very Light', 'Surface: Smooth'],
        'tips': ['Very light grip (20-35%)', 'Avoid creasing', 'Support flat surface'],
        'handling': 'Paper requires minimal grip strength. Use light, distributed pressure. Avoid bending or creasing. Support from flat surface.',
        'safety': 'Very low risk. Can tear easily. Sharp edges from cuts possible.'
    },
    'stone': {
        'insight': '🪨 Stone/Granite - Extremely hard, heavy material. Very durable but requires strong grip',
        'warning_level': 'high',
        'icon': '🪨',
        'properties': ['Hardness: 9/10', 'Weight: Heavy', 'Surface: Rough/Polished'],
        'tips': ['Use firm grip (70-85%)', 'Check weight before lifting', 'Watch for sharp edges'],
        'handling': 'Stone (granite, marble, etc.) is extremely dense and heavy. Requires substantial grip strength but beware of sharp corners. Surface may be polished (slippery) or rough (better grip).',
        'safety': 'Risk of crushing injury if dropped. Sharp edges can cause cuts. Support from bottom when lifting large pieces.'
    }
}

class HapticFeedbackSimulator:
    """Simulates haptic feedback through audio and visual cues"""
    
    def __init__(self):
        self.audio_enabled = PYGAME_AVAILABLE
        self.last_beep_time = 0
        self.beep_cooldown = 0.2
        self.proximity_alert_active = False
        self.contact_feedback_active = False
        self.feedback_animation = 0
        self.pain_alert_active = False
        
        if self.audio_enabled:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.set_num_channels(8)
                print("✅ Haptic audio feedback initialized")
            except Exception as e:
                print(f"⚠️  Audio init failed: {e}")
                self.audio_enabled = False
    
    def generate_beep(self, frequency=440, duration=0.1, volume=0.5):
        """Generate a beep sound at specified frequency"""
        if not self.audio_enabled:
            return
        
        try:
            sample_rate = 44100
            n_samples = int(duration * sample_rate)
            
            t = np.linspace(0, duration, n_samples, False)
            wave = np.sin(frequency * 2 * np.pi * t)
            
            envelope = np.ones(n_samples)
            fade_samples = int(0.005 * sample_rate)
            if fade_samples > 0:
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            wave = wave * envelope
            
            wave = (wave * volume * 32767).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))
            
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
        except Exception as e:
            pass
    
    def object_insight_alert(self, warning_level):
        """Play different sounds based on warning level"""
        if warning_level == 'critical':
            self.generate_beep(900, 0.15, 0.7)
            threading.Timer(0.1, lambda: self.generate_beep(900, 0.15, 0.7)).start()
        elif warning_level == 'high':
            self.generate_beep(700, 0.2, 0.6)
        elif warning_level == 'medium':
            self.generate_beep(500, 0.15, 0.5)
        else:  # low
            self.generate_beep(400, 0.1, 0.4)
    
    def proximity_alert(self, distance_cm):
        """Alert when object is nearby"""
        current_time = time.time()
        
        if distance_cm <= 10 and (current_time - self.last_beep_time) > self.beep_cooldown:
            freq = 400 + int((10 - distance_cm) * 60)
            duration = 0.12
            volume = 0.4 + (10 - distance_cm) * 0.04
            volume = min(volume, 0.8)
            
            self.generate_beep(freq, duration, volume)
            self.last_beep_time = current_time
            self.proximity_alert_active = True
            self.feedback_animation = 25
            return True
        return False
    
    def contact_feedback(self, material, grip_strength):
        """Provide feedback when contact is made"""
        if material and material in MATERIAL_DATABASE:
            mat_info = MATERIAL_DATABASE[material]
            
            freq = mat_info['sound_freq']
            duration = 0.2 + (grip_strength / 400)
            volume = 0.4 + (grip_strength / 250)
            volume = min(volume, 0.8)
            
            self.generate_beep(freq, duration, volume)
            
            if grip_strength > 70 and material in ['glass', 'ceramic', 'paper']:
                threading.Timer(0.15, lambda: self.generate_beep(freq * 1.3, 0.1, volume * 0.7)).start()
            
            self.contact_feedback_active = True
            self.feedback_animation = 35
    
    def pain_alert(self, pain_level):
        """Alert for high pain levels"""
        if pain_level > 70:
            self.generate_beep(800, 0.2, 0.7)
            threading.Timer(0.12, lambda: self.generate_beep(900, 0.15, 0.6)).start()
            self.pain_alert_active = True
        elif pain_level > 50:
            self.generate_beep(600, 0.15, 0.5)
            self.pain_alert_active = True

class PainFeedbackSystem:
    """Monitors grip strength and provides adaptive pain feedback"""
    
    def __init__(self):
        self.pain_level = 0
        self.pain_history = deque(maxlen=20)
        self.strain_accumulation = 0
        self.relief_mode = False
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
    
    def calculate_pain_from_grip(self, grip_strength, material):
        """Calculate pain level based on grip strength and material"""
        if material and material in MATERIAL_DATABASE:
            pain_factor = MATERIAL_DATABASE[material]['pain_factor']
        else:
            pain_factor = 0.5
        
        base_pain = max(0, (grip_strength - 40) * 1.5)
        material_pain = base_pain * pain_factor
        self.strain_accumulation += (grip_strength / 100) * 0.5
        self.strain_accumulation = min(self.strain_accumulation, 50)
        
        total_pain = material_pain + self.strain_accumulation
        
        if self.relief_mode:
            total_pain *= 0.6
        
        return min(100, max(0, int(total_pain)))
    
    def update_pain_level(self, grip_strength, material, is_gripping):
        """Update pain level based on current state"""
        if is_gripping:
            new_pain = self.calculate_pain_from_grip(grip_strength, material)
        else:
            new_pain = max(0, self.pain_level - 2)
            self.strain_accumulation = max(0, self.strain_accumulation - 0.5)
        
        self.pain_history.append(new_pain)
        self.pain_level = int(np.mean(self.pain_history)) if self.pain_history else 0
        
        if self.pain_level > 75:
            self.relief_mode = True
        elif self.pain_level < 40:
            self.relief_mode = False
    
    def get_pain_status(self):
        """Get current pain status message"""
        if self.pain_level > 80:
            return "🔥 CRITICAL", (0, 0, 255)
        elif self.pain_level > 60:
            return "⚠️ HIGH", (0, 100, 255)
        elif self.pain_level > 40:
            return "⚡ MODERATE", (0, 200, 255)
        elif self.pain_level > 20:
            return "○ MILD", (100, 255, 200)
        else:
            return "✓ NORMAL", (0, 255, 0)

class IntegratedProstheMind:
    """Main integrated system combining all components"""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if YOLO_AVAILABLE:
            print("🔄 Loading YOLOv8 model...")
            self.yolo = YOLO('yolov8n.pt')
            print("✅ YOLOv8 loaded!")
        else:
            self.yolo = None
        
        self.haptic = HapticFeedbackSimulator()
        self.pain_system = PainFeedbackSystem()
        
        self.current_material = None
        self.current_object = None
        self.current_grip = 0
        self.target_grip = 0
        self.confidence = 0.0
        self.object_distance_cm = None
        self.current_insight = None
        self.current_material_insight = None
        self.insight_animation = 0
        self.last_insight_sound = 0
        
        self.grip_history = deque(maxlen=15)
        self.material_history = deque(maxlen=10)
        
        self.detection_box = None
        self.was_in_contact = False
        self.contact_frame_count = 0
        self.last_pain_alert = 0
    
    def get_object_insight(self, object_name):
        """Get insight for detected object"""
        if object_name in OBJECT_INSIGHTS:
            return OBJECT_INSIGHTS[object_name]
        return OBJECT_INSIGHTS['default']
    
    def get_material_insight(self, material_name):
        """Get insight for detected material"""
        if not material_name:
            return None
        
        # If stone is detected, prefer granite insight (more specific)
        if material_name == 'stone':
            return MATERIAL_INSIGHTS.get('granite', MATERIAL_INSIGHTS.get('stone'))
        
        if material_name in MATERIAL_INSIGHTS:
            return MATERIAL_INSIGHTS[material_name]
        
        return None
    
    def estimate_distance(self, box_area, frame_area):
        """Estimate distance based on object size"""
        size_ratio = box_area / frame_area
        
        if size_ratio > 0.3:
            return 5
        elif size_ratio > 0.2:
            return 8
        elif size_ratio > 0.15:
            return 12
        elif size_ratio > 0.1:
            return 20
        elif size_ratio > 0.05:
            return 35
        else:
            return 50
    
    def analyze_texture(self, roi):
        """Analyze texture for material classification"""
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_energy = np.var(laplacian)
            
            brightness_std = np.std(gray)
            saturation_mean = np.mean(hsv[:, :, 1])
            
            _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            highlight_ratio = np.sum(highlights > 0) / highlights.size
            
            return {
                'edge_density': edge_density,
                'texture_energy': texture_energy,
                'brightness_std': brightness_std,
                'saturation_mean': saturation_mean,
                'highlight_ratio': highlight_ratio
            }
        except:
            return None
    
    def classify_material(self, features, detected_object=None):
        """Classify material from features"""
        if features is None:
            return None, 0.0
        
        if detected_object and detected_object in OBJECT_MATERIAL_MAP:
            return OBJECT_MATERIAL_MAP[detected_object], 0.85
        
        scores = {
            'fabric': features['edge_density'] * 3.0 + (1.0 - features['highlight_ratio']) * 2.0,
            'metal': features['highlight_ratio'] * 5.0 + (1.0 - features['edge_density']) * 2.0,
            'glass': features['highlight_ratio'] * 6.0 + (1.0 - features['edge_density']) * 3.0,
            'wood': features['texture_energy'] / 300.0 + features['edge_density'] * 2.0,
            'plastic': features['saturation_mean'] / 150.0 + (1.0 - features['texture_energy'] / 300.0) * 1.5,
            'paper': features['edge_density'] * 4.0 + (1.0 - features['highlight_ratio']) * 2.0,
            'ceramic': (1.0 - features['edge_density']) * 2.0 + features['highlight_ratio'] * 2.0,
            'stone': features['texture_energy'] / 250.0 + features['highlight_ratio'] * 3.0 + (1.0 - features['edge_density']) * 2.5
        }
        
        best_material = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = (scores[best_material] / total_score) if total_score > 0 else 0.0
        
        return best_material, min(confidence, 1.0)
    
    def detect_with_yolo(self, frame):
        """Detect objects using YOLO"""
        if not self.yolo:
            return None, None, None
        
        try:
            results = self.yolo(frame, verbose=False, conf=0.3)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                best_box = None
                best_dist = float('inf')
                best_class = None
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    dist = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
                    
                    if dist < best_dist and conf > 0.3:
                        best_dist = dist
                        best_box = (int(x1), int(y1), int(x2), int(y2))
                        best_class = results[0].names[cls]
                
                return best_box, best_class, 0.0
        except:
            pass
        
        return None, None, None
    
    def draw_insight_panel(self, frame):
        """Draw object and material insight panels with recommendations"""
        h, w = frame.shape[:2]
        panel_y = 20
        
        # Determine if we should show object or material insight (or both)
        show_object = self.current_insight is not None
        show_material = self.current_material_insight is not None
        
        # Object Insight Panel
        if show_object:
            panel_w = 900
            panel_h = 140
            panel_x = (w - panel_w) // 2
            
            # Determine color based on warning level
            warning_colors = {
                'critical': (0, 0, 255),
                'high': (0, 100, 255),
                'medium': (0, 200, 255),
                'low': (100, 255, 100)
            }
            border_color = warning_colors.get(self.current_insight['warning_level'], (100, 255, 100))
            
            # Draw background with animation
            overlay = frame.copy()
            alpha = 0.92
            
            # Pulse effect for critical warnings
            if self.current_insight['warning_level'] == 'critical':
                pulse = abs(math.sin(time.time() * 4)) * 0.1
                alpha = 0.92 + pulse
            
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                         (30, 30, 50), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Animated border
            thickness = 3
            if self.insight_animation > 0:
                thickness = 3 + int(2 * (self.insight_animation / 20))
                self.insight_animation -= 1
            
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                         border_color, thickness)
            
            # Icon
            icon_text = self.current_insight['icon']
            cv2.putText(frame, icon_text, (panel_x + 20, panel_y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)
            
            # Main insight message
            cv2.putText(frame, "OBJECT INSIGHT:", (panel_x + 100, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Truncate long insights
            insight_text = self.current_insight['insight']
            if len(insight_text) > 80:
                insight_text = insight_text[:77] + "..."
            cv2.putText(frame, insight_text, (panel_x + 100, panel_y + 58),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            # Tips section
            cv2.putText(frame, "HANDLING TIPS:", (panel_x + 100, panel_y + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            
            tips_text = " • ".join(self.current_insight['tips'][:3])
            if len(tips_text) > 100:
                tips_text = tips_text[:97] + "..."
            cv2.putText(frame, tips_text, (panel_x + 100, panel_y + 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            
            # Warning level indicator
            level_text = f"[{self.current_insight['warning_level'].upper()}]"
            cv2.putText(frame, level_text, (panel_x + panel_w - 120, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)
            
            panel_y += panel_h + 15
        
        # Material Insight Panel (e.g., Granite Insight)
        if show_material:
            panel_w = 900
            panel_h = 160
            panel_x = (w - panel_w) // 2
            
            # Determine color based on warning level
            warning_colors = {
                'critical': (0, 0, 255),
                'high': (0, 100, 255),
                'medium': (0, 200, 255),
                'low': (100, 255, 100)
            }
            border_color = warning_colors.get(self.current_material_insight['warning_level'], (100, 255, 100))
            
            # Draw background
            overlay = frame.copy()
            alpha = 0.92
            
            # Pulse effect for critical/high warnings
            if self.current_material_insight['warning_level'] in ['critical', 'high']:
                pulse = abs(math.sin(time.time() * 3)) * 0.08
                alpha = 0.92 + pulse
            
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                         (40, 30, 30), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Border
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                         border_color, 3)
            
            # Icon
            icon_text = self.current_material_insight['icon']
            cv2.putText(frame, icon_text, (panel_x + 20, panel_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)
            
            # Header - Show "GRANITE INSIGHT" if stone is detected
            material_name = self.current_material if self.current_material else ""
            header_text = "GRANITE INSIGHT:" if material_name == 'stone' else "MATERIAL INSIGHT:"
            cv2.putText(frame, header_text, (panel_x + 100, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Main insight message
            insight_text = self.current_material_insight['insight']
            cv2.putText(frame, insight_text, (panel_x + 100, panel_y + 58),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 200), 2)
            
            # Properties
            if 'properties' in self.current_material_insight:
                props_text = " | ".join(self.current_material_insight['properties'][:3])
                cv2.putText(frame, props_text, (panel_x + 100, panel_y + 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
            
            # Handling tips
            cv2.putText(frame, "HANDLING:", (panel_x + 100, panel_y + 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            
            tips_text = " • ".join(self.current_material_insight['tips'][:3])
            if len(tips_text) > 110:
                tips_text = tips_text[:107] + "..."
            cv2.putText(frame, tips_text, (panel_x + 100, panel_y + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            
            # Warning level
            level_text = f"[{self.current_material_insight['warning_level'].upper()}]"
            cv2.putText(frame, level_text, (panel_x + panel_w - 120, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)
            
            # Safety info (if available)
            if 'safety' in self.current_material_insight and panel_y + panel_h < h - 200:
                safety_text = self.current_material_insight['safety']
                if len(safety_text) > 100:
                    safety_text = safety_text[:97] + "..."
                cv2.putText(frame, f"⚠ {safety_text}", (panel_x + 100, panel_y + 155),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 150), 1)
    
    def draw_pain_monitor(self, frame):
        """Draw pain monitoring panel"""
        h, w = frame.shape[:2]
        
        panel_x = w - 350
        panel_y = h - 280
        panel_w = 330
        panel_h = 260
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        pain_status, status_color = self.pain_system.get_pain_status()
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     status_color, 3)
        
        cv2.putText(frame, "PAIN MONITORING", (panel_x + 15, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        y_off = panel_y + 60
        
        cv2.putText(frame, f"Pain Level: {self.pain_system.pain_level}%",
                   (panel_x + 15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_off += 25
        bar_w = int(300 * (self.pain_system.pain_level / 100))
        
        if self.pain_system.pain_level > 70:
            bar_color = (0, 0, 255)
        elif self.pain_system.pain_level > 40:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 255, 100)
        
        cv2.rectangle(frame, (panel_x + 15, y_off), (panel_x + 15 + bar_w, y_off + 20),
                     bar_color, -1)
        cv2.rectangle(frame, (panel_x + 15, y_off), (panel_x + 315, y_off + 20),
                     (80, 80, 80), 2)
        
        y_off += 40
        
        cv2.putText(frame, f"Status: {pain_status}", (panel_x + 15, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        
        y_off += 35
        
        strain_pct = int(self.pain_system.strain_accumulation * 2)
        cv2.putText(frame, f"Strain: {strain_pct}%", (panel_x + 15, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        y_off += 20
        strain_bar_w = int(300 * (strain_pct / 100))
        cv2.rectangle(frame, (panel_x + 15, y_off), (panel_x + 15 + strain_bar_w, y_off + 12),
                     (255, 150, 0), -1)
        cv2.rectangle(frame, (panel_x + 15, y_off), (panel_x + 315, y_off + 12),
                     (80, 80, 80), 2)
        
        y_off += 30
        
        if self.pain_system.relief_mode:
            cv2.putText(frame, "RELIEF MODE ACTIVE", (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            
            pulse_size = int(8 + 4 * abs(math.sin(time.time() * 3)))
            cv2.circle(frame, (panel_x + panel_w - 30, y_off - 5), pulse_size, (100, 255, 100), -1)
        else:
            cv2.putText(frame, "Normal Operation", (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        y_off += 35
        
        if self.pain_system.pain_level > 70:
            cv2.putText(frame, "! Reduce grip strength!", (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
        elif self.pain_system.pain_level > 50:
            cv2.putText(frame, "> Consider rest period", (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    
    def draw_haptic_panel(self, frame):
        """Draw haptic feedback panel"""
        h, w = frame.shape[:2]
        
        panel_x = w - 320
        panel_y = 160
        panel_w = 300
        panel_h = 250
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (100, 255, 100), 2)
        
        cv2.putText(frame, "HAPTIC FEEDBACK", (panel_x + 15, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        y_off = panel_y + 60
        
        if self.object_distance_cm is not None:
            cv2.putText(frame, f"Distance: {self.object_distance_cm:.1f} cm",
                       (panel_x + 15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            y_off += 25
            bar_max_w = 270
            
            if self.object_distance_cm <= 50:
                proximity_ratio = 1.0 - (self.object_distance_cm / 50.0)
                bar_w = int(bar_max_w * proximity_ratio)
                
                if self.object_distance_cm <= 10:
                    bar_color = (0, 0, 255)
                    status = "! VERY CLOSE"
                elif self.object_distance_cm <= 20:
                    bar_color = (0, 165, 255)
                    status = "> APPROACHING"
                else:
                    bar_color = (0, 255, 255)
                    status = "  Detected"
                
                cv2.rectangle(frame, (panel_x + 15, y_off), (panel_x + 15 + bar_w, y_off + 18),
                             bar_color, -1)
                cv2.rectangle(frame, (panel_x + 15, y_off), (panel_x + 15 + bar_max_w, y_off + 18),
                             (80, 80, 80), 2)
                
                cv2.putText(frame, status, (panel_x + 15, y_off + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)
                
                if self.object_distance_cm <= 10 and self.haptic.feedback_animation > 0:
                    pulse = int(abs(10 * math.sin(time.time() * 15)))
                    radius = max(5, 8 + pulse)
                    cv2.circle(frame, (panel_x + panel_w - 30, y_off + 9), radius, (0, 0, 255), -1)
                    cv2.putText(frame, "BEEP", (panel_x + panel_w - 55, y_off + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_off += 55
        
        if self.current_material:
            mat_info = MATERIAL_DATABASE[self.current_material]
            
            cv2.line(frame, (panel_x + 15, y_off), (panel_x + panel_w - 15, y_off), (80, 80, 80), 1)
            y_off += 20
            
            cv2.putText(frame, "CONTACT DETECTED", (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_off += 25
            
            intensity = mat_info['haptic_intensity']
            if intensity == 'VERY LOW':
                intensity_color = (100, 255, 100)
                bars = 1
            elif intensity == 'LOW':
                intensity_color = (150, 255, 100)
                bars = 2
            elif intensity == 'MEDIUM':
                intensity_color = (200, 200, 100)
                bars = 3
            elif intensity == 'HIGH':
                intensity_color = (255, 150, 50)
                bars = 4
            else:
                intensity_color = (50, 50, 255)
                bars = 5
            
            cv2.putText(frame, f"Intensity: {intensity}", (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_off += 20
            for i in range(5):
                bar_color = intensity_color if i < bars else (40, 40, 40)
                cv2.rectangle(frame, (panel_x + 15 + i * 55, y_off),
                             (panel_x + 15 + i * 55 + 50, y_off + 12), bar_color, -1)
                cv2.rectangle(frame, (panel_x + 15 + i * 55, y_off),
                             (panel_x + 15 + i * 55 + 50, y_off + 12), (100, 100, 100), 1)
            
            y_off += 30
            
            if self.current_grip > 80:
                feedback = "! STRONG GRIP"
                feedback_color = (0, 100, 255)
            elif self.current_grip > 60:
                feedback = "v FIRM GRIP"
                feedback_color = (0, 200, 255)
            elif self.current_grip > 40:
                feedback = "v BALANCED"
                feedback_color = (100, 255, 100)
            else:
                feedback = "o GENTLE"
                feedback_color = (150, 255, 150)
            
            cv2.putText(frame, feedback, (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, feedback_color, 2)
            
            if self.haptic.feedback_animation > 0:
                pulse_size = int(15 * (self.haptic.feedback_animation / 30))
                cv2.circle(frame, (panel_x + panel_w - 30, y_off - 7), pulse_size, feedback_color, 2)
                self.haptic.feedback_animation -= 1
        else:
            cv2.line(frame, (panel_x + 15, y_off), (panel_x + panel_w - 15, y_off), (80, 80, 80), 1)
            y_off += 25
            cv2.putText(frame, "No contact", (panel_x + 15, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        y_off = panel_y + panel_h - 25
        audio_status = "Audio: ON" if self.haptic.audio_enabled else "Audio: OFF"
        audio_color = (100, 255, 100) if self.haptic.audio_enabled else (100, 100, 100)
        cv2.putText(frame, audio_status, (panel_x + 15, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, audio_color, 1)
    
    def draw_main_ui(self, frame):
        """Draw main UI panel"""
        h, w = frame.shape[:2]
        
        if self.detection_box:
            x1, y1, x2, y2 = self.detection_box
            color = (0, 255, 0) if self.current_material else (255, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            marker_len = 20
            cv2.line(frame, (x1, y1), (x1 + marker_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + marker_len), color, 4)
            cv2.line(frame, (x2, y1), (x2 - marker_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + marker_len), color, 4)
            cv2.line(frame, (x1, y2), (x1 + marker_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - marker_len), color, 4)
            cv2.line(frame, (x2, y2), (x2 - marker_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - marker_len), color, 4)
        
        panel_w = 380
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, "PROSTHEMIND+", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 200, 255), 3)
        cv2.putText(frame, "Smart Object Detection", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        status_color = (0, 255, 0) if self.yolo else (0, 0, 255)
        status_text = "ACTIVE" if self.yolo else "NO YOLO"
        cv2.circle(frame, (panel_w - 30, 50), 8, status_color, -1)
        cv2.putText(frame, status_text, (panel_w - 90, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y_offset = 120
        
        if self.current_object:
            cv2.putText(frame, "OBJECT DETECTED:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(frame, self.current_object.upper(), (20, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            y_offset += 70
        
        cv2.putText(frame, "MATERIAL DETECTED:", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        if self.current_material:
            mat_info = MATERIAL_DATABASE[self.current_material]
            
            cv2.rectangle(frame, (20, y_offset + 10), (65, y_offset + 45),
                         mat_info['color'], -1)
            cv2.rectangle(frame, (20, y_offset + 10), (65, y_offset + 45),
                         (255, 255, 255), 2)
            cv2.putText(frame, self.current_material.upper(), (75, y_offset + 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            y_offset += 65
            cv2.putText(frame, f"Confidence: {int(self.confidence * 100)}%",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            bar_w = int(330 * self.confidence)
            bar_color = (0, 255, 0) if self.confidence > 0.6 else (0, 200, 255)
            cv2.rectangle(frame, (20, y_offset + 10), (20 + bar_w, y_offset + 28),
                         bar_color, -1)
            cv2.rectangle(frame, (20, y_offset + 10), (350, y_offset + 28),
                         (100, 100, 100), 2)
            
            y_offset += 55
            cv2.putText(frame, "ADAPTIVE GRIP STRENGTH:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            gauge_h = 180
            gauge_y = y_offset + 20
            cv2.rectangle(frame, (20, gauge_y), (110, gauge_y + gauge_h),
                         (40, 40, 40), -1)
            cv2.rectangle(frame, (20, gauge_y), (110, gauge_y + gauge_h),
                         (100, 100, 100), 2)
            
            for i in range(0, 101, 20):
                mark_y = gauge_y + int(gauge_h * (1 - i/100))
                cv2.line(frame, (110, mark_y), (120, mark_y), (150, 150, 150), 1)
                cv2.putText(frame, str(i), (125, mark_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            grip_h = int(gauge_h * self.current_grip / 100)
            if self.current_grip < 50:
                grip_color = (0, 255, 0)
            elif self.current_grip < 80:
                grip_color = (0, 200, 255)
            else:
                grip_color = (50, 100, 255)
            
            cv2.rectangle(frame, (20, gauge_y + gauge_h - grip_h),
                         (110, gauge_y + gauge_h), grip_color, -1)
            
            cv2.putText(frame, f"{self.current_grip}%", (35, gauge_y + gauge_h + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            y_offset = gauge_y + gauge_h + 60
            cv2.putText(frame, "PROPERTIES:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(frame, mat_info['description'], (20, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            y_offset += 50
            if self.current_grip > 80:
                cv2.putText(frame, "! HIGH FORCE", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)
            elif self.current_grip < 40:
                cv2.putText(frame, "v GENTLE GRIP", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "v BALANCED", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        else:
            cv2.putText(frame, "No object detected", (20, y_offset + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)
            cv2.putText(frame, "Show object to camera", (20, y_offset + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        
        cv2.putText(frame, "Controls: Q=Quit | R=Reset | S=Screenshot",
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main system loop"""
        print("\n" + "="*70)
        print("           PROSTHEMIND+ WITH OBJECT INSIGHTS")
        print("="*70)
        print("\nComponents Status:")
        if self.yolo:
            print("  [OK] YOLOv8 Object Detection")
        else:
            print("  [!!] YOLOv8 unavailable - texture analysis only")
        
        if self.haptic.audio_enabled:
            print("  [OK] Haptic Audio Feedback")
        else:
            print("  [!!] Audio feedback unavailable")
        
        print("  [OK] Pain Monitoring System")
        print("  [OK] Adaptive Grip Control")
        print("  [OK] Smart Object Insights")
        print("  [OK] Material-Specific Insights (Granite, Marble, Glass, etc.)")
        
        print("\nActive Features:")
        print("  > Real-time object & material detection")
        print("  > Context-aware handling recommendations")
        print("  > Object-specific safety warnings")
        print("  > Material-specific insights (e.g., Granite, Marble, Metal)")
        print("  > Detailed material properties & handling guides")
        print("  > Proximity alerts (< 10cm)")
        print("  > Contact haptic feedback")
        print("  > Pain level monitoring")
        print("  > Adaptive relief mode")
        
        print("\nControls:")
        print("  Q - Quit system")
        print("  R - Reset all systems")
        print("  S - Save screenshot")
        print("="*70 + "\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Cannot read from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                h, w = frame.shape[:2]
                frame_area = h * w
                
                if frame_count % 3 == 0:
                    detection_box, detected_object, _ = self.detect_with_yolo(frame)
                    
                    if detection_box:
                        self.detection_box = detection_box
                        
                        # New object detected - show insight
                        if detected_object != self.current_object:
                            self.current_object = detected_object
                            self.current_insight = self.get_object_insight(detected_object)
                            self.insight_animation = 20
                            
                            # Play insight sound
                            current_time = time.time()
                            if (current_time - self.last_insight_sound) > 1.5:
                                self.haptic.object_insight_alert(self.current_insight['warning_level'])
                                self.last_insight_sound = current_time
                                
                                print(f"\n[INSIGHT] {detected_object.upper()}")
                                print(f"  {self.current_insight['insight']}")
                                print(f"  Tips: {', '.join(self.current_insight['tips'])}")
                        
                        x1, y1, x2, y2 = detection_box
                        box_area = (x2 - x1) * (y2 - y1)
                        self.object_distance_cm = self.estimate_distance(box_area, frame_area)
                        
                        if self.object_distance_cm <= 10:
                            self.haptic.proximity_alert(self.object_distance_cm)
                        
                        roi = frame[y1:y2, x1:x2]
                        features = self.analyze_texture(roi)
                        material, confidence = self.classify_material(features, detected_object)
                        
                        if material and confidence > 0.4:
                            self.material_history.append(material)
                            if len(self.material_history) >= 5:
                                material = max(set(self.material_history), key=self.material_history.count)
                            
                            is_new_contact = (not self.was_in_contact and material is not None)
                            is_new_material = (material != self.current_material)
                            
                            self.current_material = material
                            self.confidence = confidence
                            self.target_grip = MATERIAL_DATABASE[material]['grip_strength']
                            
                            # Update material insight
                            if is_new_material:
                                self.current_material_insight = self.get_material_insight(material)
                                if self.current_material_insight:
                                    current_time = time.time()
                                    if (current_time - self.last_insight_sound) > 1.5:
                                        self.haptic.object_insight_alert(self.current_material_insight['warning_level'])
                                        self.last_insight_sound = current_time
                                    
                                    # Print material insight to console (e.g., "granite insight")
                                    # Show "GRANITE INSIGHT" if stone is detected (uses granite insight)
                                    display_name = "GRANITE" if material == 'stone' else material.upper()
                                    print(f"\n{'='*70}")
                                    print(f"[MATERIAL INSIGHT] {display_name}")
                                    print(f"{'='*70}")
                                    print(f"  {self.current_material_insight['insight']}")
                                    if 'properties' in self.current_material_insight:
                                        print(f"  Properties: {', '.join(self.current_material_insight['properties'])}")
                                    print(f"  Tips: {', '.join(self.current_material_insight['tips'])}")
                                    if 'handling' in self.current_material_insight:
                                        print(f"  Handling: {self.current_material_insight['handling']}")
                                    if 'safety' in self.current_material_insight:
                                        print(f"  Safety: {self.current_material_insight['safety']}")
                                    print(f"{'='*70}")
                            
                            self.grip_history.append(self.target_grip)
                            self.current_grip = int(np.mean(self.grip_history))
                            
                            if is_new_contact or self.contact_frame_count == 0:
                                self.haptic.contact_feedback(material, self.current_grip)
                                self.contact_frame_count = 30
                            
                            self.was_in_contact = True
                            
                            if self.contact_frame_count > 0:
                                self.contact_frame_count -= 1
                            
                            self.pain_system.update_pain_level(self.current_grip, material, True)
                            
                            current_time = time.time()
                            if self.pain_system.pain_level > 60 and \
                               (current_time - self.last_pain_alert) > 3.0:
                                self.haptic.pain_alert(self.pain_system.pain_level)
                                self.last_pain_alert = current_time
                                print(f"[ALERT] Pain: {self.pain_system.pain_level}% | "
                                      f"Material: {material} | Grip: {self.current_grip}%")
                        else:
                            self.confidence = max(0, self.confidence - 0.03)
                            if self.confidence < 0.2:
                                self.current_material = None
                                self.current_material_insight = None
                                self.was_in_contact = False
                                self.pain_system.update_pain_level(0, None, False)
                    else:
                        self.confidence = max(0, self.confidence - 0.05)
                        if self.confidence < 0.1:
                            self.current_material = None
                            self.current_object = None
                            self.current_insight = None
                            self.current_material_insight = None
                            self.detection_box = None
                            self.object_distance_cm = None
                            
                            self.grip_history.append(0)
                            self.current_grip = int(np.mean(self.grip_history))
                            
                            self.was_in_contact = False
                            self.haptic.proximity_alert_active = False
                            
                            self.pain_system.update_pain_level(0, None, False)
                
                frame = self.draw_main_ui(frame)
                
                # Draw insight panel at top (highest priority)
                if self.current_insight:
                    self.draw_insight_panel(frame)
                
                self.draw_haptic_panel(frame)
                self.draw_pain_monitor(frame)
                
                cv2.imshow('PROSTHEMIND+ with Object Insights', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.current_material = None
                    self.current_object = None
                    self.current_insight = None
                    self.current_material_insight = None
                    self.current_grip = 0
                    self.confidence = 0.0
                    self.grip_history.clear()
                    self.material_history.clear()
                    self.pain_system.pain_level = 0
                    self.pain_system.strain_accumulation = 0
                    print("[RESET] All systems reset")
                elif key == ord('s') or key == ord('S'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"prosthemind_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[SCREENSHOT] Saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Shutting down...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n[SHUTDOWN] PROSTHEMIND+ system closed")

def main():
    """Main entry point"""
    system = IntegratedProstheMind()
    system.run()

if __name__ == "__main__":
    main()