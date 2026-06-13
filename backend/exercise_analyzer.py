# exercise_analyzer.py
from utils import calculate_angle

class ExerciseAnalyzer:
    def __init__(self, exercise_config):
        self.config = exercise_config
        self.current_exercise = None
        
    def analyze_pushup(self, landmarks):
        # Get coordinates for shoulders, elbows, wrists
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        
        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder[1:3], left_elbow[1:3], left_wrist[1:3])
        right_elbow_angle = calculate_angle(right_shoulder[1:3], right_elbow[1:3], right_wrist[1:3])
        
        # Calculate back angle (shoulders to hips)
        left_hip = landmarks[23]
        back_angle = calculate_angle(left_shoulder[1:3], left_hip[1:3], left_elbow[1:3])
        
        # Form feedback
        feedback = []
        if left_elbow_angle < 90 or right_elbow_angle < 90:
            feedback.append("DOWN position")
        elif left_elbow_angle > 160 or right_elbow_angle > 160:
            feedback.append("UP position")
        
        if back_angle < 160:
            feedback.append("Keep your back straight!")
        
        return {
            "exercise": "pushup",
            "elbow_angle": (left_elbow_angle + right_elbow_angle) / 2,
            "back_angle": back_angle,
            "feedback": feedback,
            "is_valid": back_angle >= 160
        }
    
    def analyze_squat(self, landmarks):
        # Get coordinates for hips, knees, ankles
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]
        right_hip = landmarks[24]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip[1:3], left_knee[1:3], left_ankle[1:3])
        right_knee_angle = calculate_angle(right_hip[1:3], right_knee[1:3], right_ankle[1:3])
        
        # Calculate torso angle
        left_shoulder = landmarks[11]
        torso_angle = calculate_angle(left_shoulder[1:3], left_hip[1:3], left_knee[1:3])
        
        feedback = []
        if left_knee_angle < 90 or right_knee_angle < 90:
            feedback.append("DOWN position")
        elif left_knee_angle > 160 or right_knee_angle > 160:
            feedback.append("UP position")
        
        if torso_angle < 80:
            feedback.append("Leaning too far forward!")
        
        return {
            "exercise": "squat",
            "knee_angle": (left_knee_angle + right_knee_angle) / 2,
            "torso_angle": torso_angle,
            "feedback": feedback,
            "is_valid": torso_angle >= 80
        }