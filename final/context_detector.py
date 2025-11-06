import random

class ContextDetector:
    def detect(self):
        # Simulated classifier: randomly detect terrain type
        return random.choice(["flat", "stairs", "incline"])
