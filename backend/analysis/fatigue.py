import numpy as np
from collections import deque

class FatigueEstimator:
    def __init__(self, window_size=30):
        self.rep_speeds = deque(maxlen=window_size)
        self.rom_history = deque(maxlen=window_size)
        self.sway_history = deque(maxlen=window_size)

    def update(self, landmarks, exercise):
        # Simplified: measure wrist velocity and hip sway
        # Returns fatigue score 0-100
        # Placeholder – implement real logic
        speed = np.random.uniform(0.5, 1.5)
        self.rep_speeds.append(speed)
        if len(self.rep_speeds) > 5:
            speed_degradation = max(0, (self.rep_speeds[0] - self.rep_speeds[-1]) / self.rep_speeds[0])
        else:
            speed_degradation = 0
        fatigue = min(100, speed_degradation * 100)
        return fatigue

    def get_intensity_factor(self):
        return max(0.5, 1.0 - (np.mean(self.rep_speeds) if self.rep_speeds else 0))