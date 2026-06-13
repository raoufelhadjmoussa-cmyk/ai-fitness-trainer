import numpy as np
from backend.utils import calculate_angle, compute_velocity, compute_symmetry

class FeedbackGenerator:
    def __init__(self, exercise_config):
        self.config = exercise_config
        # Store previous frame landmarks for velocity
        self.prev_landmarks = None

    def analyze_form(self, exercise, landmarks):
        """Return (feedback_msgs, form_score) with advanced metrics"""
        if exercise not in self.config:
            return [], 100.0

        issues = []
        form_errors = 0
        total_checks = 0

        # --- Exercise‑specific analysis ---
        if exercise == "pushup":
            # Rounded back: angle shoulders-hips-knees
            sh = (landmarks[11][1], landmarks[11][2])
            hip = (landmarks[23][1], landmarks[23][2])
            knee = (landmarks[25][1], landmarks[25][2])
            back_angle = calculate_angle(sh, hip, knee)
            total_checks += 1
            if back_angle < 160:
                issues.append("Rounded back – keep spine neutral")
                form_errors += 1

            # Elbow flare: angle shoulder-elbow-wrist
            left_elbow = calculate_angle(
                (landmarks[11][1], landmarks[11][2]),
                (landmarks[13][1], landmarks[13][2]),
                (landmarks[15][1], landmarks[15][2])
            )
            right_elbow = calculate_angle(
                (landmarks[12][1], landmarks[12][2]),
                (landmarks[14][1], landmarks[14][2]),
                (landmarks[16][1], landmarks[16][2])
            )
            total_checks += 1
            if left_elbow < 70 or right_elbow < 70:
                issues.append("Elbows too wide – tuck them in")
                form_errors += 1

            # Sagging hips: hip height relative to shoulders
            hip_y = (landmarks[23][2] + landmarks[24][2]) / 2
            sh_y = (landmarks[11][2] + landmarks[12][2]) / 2
            total_checks += 1
            if hip_y > sh_y + 30:  # hips lower than shoulders = sagging
                issues.append("Hips sagging – engage core")
                form_errors += 1

        elif exercise == "squat":
            # Knee valgus: lateral distance between knees and ankles
            left_knee = (landmarks[25][1], landmarks[25][2])
            left_ankle = (landmarks[27][1], landmarks[27][2])
            right_knee = (landmarks[26][1], landmarks[26][2])
            right_ankle = (landmarks[28][1], landmarks[28][2])
            valgus = abs(left_knee[0] - left_ankle[0]) + abs(right_knee[0] - right_ankle[0])
            total_checks += 1
            if valgus > 50:
                issues.append("Knees caving inward – push knees out")
                form_errors += 1

            # Depth: knee angle < 90?
            left_knee_angle = calculate_angle(
                (landmarks[23][1], landmarks[23][2]),
                (landmarks[25][1], landmarks[25][2]),
                (landmarks[27][1], landmarks[27][2])
            )
            total_checks += 1
            if left_knee_angle > 100:
                issues.append("Not deep enough – go lower")
                form_errors += 1

            # Forward lean: torso angle relative to vertical
            torso = calculate_angle(
                (landmarks[11][1], landmarks[11][2]),
                (landmarks[23][1], landmarks[23][2]),
                (landmarks[25][1], landmarks[25][2])
            )
            total_checks += 1
            if torso < 70:
                issues.append("Leaning too far forward – chest up")
                form_errors += 1

        elif exercise == "deadlift":
            # Lumbar flexion: angle between shoulders, hips, knees
            sh = (landmarks[11][1], landmarks[11][2])
            hip = (landmarks[23][1], landmarks[23][2])
            knee = (landmarks[25][1], landmarks[25][2])
            back_angle = calculate_angle(sh, hip, knee)
            total_checks += 1
            if back_angle < 140:
                issues.append("Rounded back – keep spine neutral")
                form_errors += 1

            # Bar path deviation: wrist relative to ankle
            wrist_y = (landmarks[15][2] + landmarks[16][2]) / 2
            ankle_y = (landmarks[27][2] + landmarks[28][2]) / 2
            if wrist_y > ankle_y + 50:
                issues.append("Bar too far from body")
                form_errors += 1

        # --- Motion quality (velocity, ROM, symmetry) ---
        if self.prev_landmarks is not None:
            # Compute velocity of wrist
            wrist_curr = ((landmarks[15][1]+landmarks[16][1])/2, (landmarks[15][2]+landmarks[16][2])/2)
            wrist_prev = ((self.prev_landmarks[15][1]+self.prev_landmarks[16][1])/2,
                          (self.prev_landmarks[15][2]+self.prev_landmarks[16][2])/2)
            speed = compute_velocity(wrist_prev, wrist_curr)
            if speed > 1500:  # pixels/sec ~ too fast
                issues.append("Slow down – control the movement")
                form_errors += 1

            # Asymmetry between left and right elbows
            left_elbow_angle = calculate_angle(
                (landmarks[11][1], landmarks[11][2]), (landmarks[13][1], landmarks[13][2]), (landmarks[15][1], landmarks[15][2])
            )
            right_elbow_angle = calculate_angle(
                (landmarks[12][1], landmarks[12][2]), (landmarks[14][1], landmarks[14][2]), (landmarks[16][1], landmarks[16][2])
            )
            symmetry = compute_symmetry(left_elbow_angle, right_elbow_angle)
            if symmetry < 0.7:
                issues.append("Asymmetrical movement – check both sides")
                form_errors += 1

        self.prev_landmarks = landmarks

        # Compute form score (0-100)
        if total_checks == 0:
            form_score = 100
        else:
            form_score = max(0, 100 - (form_errors / total_checks * 100))

        return issues, form_score