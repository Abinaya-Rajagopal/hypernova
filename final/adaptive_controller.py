import numpy as np

class AdaptiveController:
    def __init__(self):
        self.last_command = np.zeros(3)
    
    def apply(self, prediction, context):
        adjustment = np.array(prediction)
        if context == "stairs":
            adjustment[1] += 0.3  # increase lift
        elif context == "incline":
            adjustment[0] += 0.2  # lean forward
        smoothed = 0.7 * self.last_command + 0.3 * adjustment
        self.last_command = smoothed
        return smoothed
