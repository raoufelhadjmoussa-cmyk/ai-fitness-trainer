import cv2
import numpy as np
from abc import ABC, abstractmethod
from mediapipe.python.solutions.pose import Pose, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks

class PoseDetectorBase(ABC):
    @abstractmethod
    def find_pose(self, img, draw=True): pass
    @abstractmethod
    def get_landmarks(self, img): pass

class MediaPipeDetector(PoseDetectorBase):
    def __init__(self, static_mode=False, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = Pose(static_image_mode=static_mode, model_complexity=model_complexity,
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)
        self.landmarks = None

    def find_pose(self, img, draw=True):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        self.landmarks = None
        if results.pose_landmarks:
            self.landmarks = [[i, lm.x, lm.y, lm.z, lm.visibility]
                              for i, lm in enumerate(results.pose_landmarks.landmark)]
            if draw:
                draw_landmarks(img, results.pose_landmarks, POSE_CONNECTIONS)
        return img

    def get_landmarks(self, img):
        return self.landmarks

class YOLOPoseDetector(PoseDetectorBase):
    def __init__(self, model_path="yolo11n-pose.pt", device="cpu"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.landmarks = None

    def find_pose(self, img, draw=True):
        results = self.model.predict(img, device=self.device, verbose=False, conf=0.5)
        if results and results[0].keypoints is not None:
            kpts = results[0].keypoints.data[0].cpu().numpy()
            full = np.zeros((33, 3))
            for y, m in {5:11,6:12,7:13,8:14,9:15,10:16,11:23,12:24,13:25,14:26,15:27,16:28}.items():
                if y < len(kpts): full[m] = kpts[y]
            self.landmarks = [[i,full[i,0],full[i,1],full[i,2],1.0] for i in range(33)]
        else:
            self.landmarks = None
        return img

    def get_landmarks(self, img):
        return self.landmarks

def get_pose_detector(detector_type="mediapipe", **kwargs):
    if detector_type == "mediapipe": return MediaPipeDetector(**kwargs)
    elif detector_type == "yolo": return YOLOPoseDetector(**kwargs)
    else: raise ValueError(f"Unknown detector: {detector_type}")