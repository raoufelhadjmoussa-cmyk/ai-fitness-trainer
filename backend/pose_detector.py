import cv2
import numpy as np
import os
import urllib.request
from abc import ABC, abstractmethod

# New MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------------------------------------------
# Base class
# -----------------------------------------------------------------
class PoseDetectorBase(ABC):
    @abstractmethod
    def find_pose(self, img, draw=True):
        pass
    @abstractmethod
    def get_landmarks(self, img):
        pass

# -----------------------------------------------------------------
# MediaPipe Tasks implementation
# -----------------------------------------------------------------
class MediaPipeDetector(PoseDetectorBase):
    def __init__(self, static_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Download pose landmarker model if not present
        self.model_path = 'pose_landmarker_full.task'
        if not os.path.exists(self.model_path):
            print("Downloading pose model (14MB) ...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.landmarks = None
        self.timestamp = 0  # microseconds

    def find_pose(self, img, draw=True):
        # Convert BGR to RGB and to MediaPipe Image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Increment timestamp (microseconds) assuming ~30 fps
        self.timestamp += 33333
        detection_result = self.detector.detect_for_video(mp_image, self.timestamp)
        self.landmarks = []
        if detection_result.pose_landmarks:
            # Convert landmarks to list of [id, x, y, z, visibility]
            for i, landmark in enumerate(detection_result.pose_landmarks[0]):
                self.landmarks.append([i, landmark.x, landmark.y, landmark.z, 1.0])
            if draw:
                h, w, _ = img.shape
                # Draw circles and connections
                for lm in self.landmarks:
                    cx, cy = int(lm[1] * w), int(lm[2] * h)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
                # Define key connections (simplified skeleton)
                connections = [
                    (11,12), (11,13), (13,15), (12,14), (14,16),  # arms
                    (11,23), (12,24), (23,24), (23,25), (24,26),  # torso
                    (25,27), (26,28), (27,28)                     # legs
                ]
                for (a,b) in connections:
                    if a < len(self.landmarks) and b < len(self.landmarks):
                        pt1 = (int(self.landmarks[a][1]*w), int(self.landmarks[a][2]*h))
                        pt2 = (int(self.landmarks[b][1]*w), int(self.landmarks[b][2]*h))
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        else:
            self.landmarks = None
        return img

    def get_landmarks(self, img):
        return self.landmarks

    def __del__(self):
        if hasattr(self, 'detector'):
            self.detector.close()

# -----------------------------------------------------------------
# YOLO11 Pose implementation
# -----------------------------------------------------------------
class YOLOPoseDetector(PoseDetectorBase):
    def __init__(self, model_path="yolo11n-pose.pt", device="cpu"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")
        self.device = device
        self.landmarks = None

    def find_pose(self, img, draw=True):
        results = self.model.predict(img, device=self.device, verbose=False, conf=0.5)
        if results and results[0].keypoints is not None:
            kpts = results[0].keypoints.data[0].cpu().numpy()  # (17,3) x,y,conf
            full = np.zeros((33, 3))
            map_yolo_to_mp = {5:11,6:12,7:13,8:14,9:15,10:16,11:23,12:24,13:25,14:26,15:27,16:28}
            for y_idx, mp_idx in map_yolo_to_mp.items():
                if y_idx < len(kpts):
                    full[mp_idx,0] = kpts[y_idx,0]
                    full[mp_idx,1] = kpts[y_idx,1]
                    full[mp_idx,2] = kpts[y_idx,2]
            self.landmarks = [[i, full[i,0], full[i,1], full[i,2], 1.0] for i in range(33)]
            if draw:
                h,w,_ = img.shape
                for (x,y) in full[:,:2]:
                    if x>0 and y>0:
                        cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
        else:
            self.landmarks = None
        return img

    def get_landmarks(self, img):
        return self.landmarks

# -----------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------
def get_pose_detector(detector_type="mediapipe", **kwargs):
    if detector_type == "mediapipe":
        return MediaPipeDetector(**kwargs)
    elif detector_type == "yolo":
        return YOLOPoseDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector: {detector_type}")