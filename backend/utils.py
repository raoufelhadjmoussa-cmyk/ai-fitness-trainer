import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# New: velocity from landmark history
def compute_velocity(prev_pt, curr_pt, time_delta=1/30):
    if prev_pt is None: return 0
    dist = calculate_distance(prev_pt, curr_pt)
    return dist / time_delta  # pixels per second

# New: range of motion over sequence
def compute_rom(values):
    return max(values) - min(values)

# New: symmetry between left and right
def compute_symmetry(left_angle, right_angle):
    return 1 - abs(left_angle - right_angle) / (left_angle + right_angle + 1e-6)