import cv2
import numpy as np
import os
import argparse
import urllib.request
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_pose_sequence(image_folder, output_folder, seq_len=30, stride=10):
    # Load model (download once)
    model_path = 'pose_landmarker_full.task'
    if not os.path.exists(model_path):
        print("Downloading pose model (14MB) ...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    image_files = sorted([f for f in os.listdir(image_folder) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if len(image_files) < seq_len:
        print(f"Skipping {image_folder}: only {len(image_files)} images")
        detector.close()
        return

    sequences = []
    total_windows = len(image_files) - seq_len + 1
    # Process windows with progress bar
    for i in tqdm(range(0, total_windows, stride), desc=f"Processing {os.path.basename(image_folder)}"):
        seq = []
        ok = True
        for j in range(i, i + seq_len):
            img_path = os.path.join(image_folder, image_files[j])
            img = cv2.imread(img_path)
            if img is None:
                ok = False
                break
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                features = []
                for lm in landmarks:
                    features.extend([lm.x, lm.y, lm.z, 1.0])
                seq.append(features)
            else:
                seq.append([0.0] * (33 * 4))
        if ok and len(seq) == seq_len:
            sequences.append(np.array(seq, dtype=np.float32))

    if sequences:
        os.makedirs(output_folder, exist_ok=True)
        for idx, seq in enumerate(sequences):
            np.save(os.path.join(output_folder, f"{os.path.basename(image_folder)}_{idx}.npy"), seq)
        print(f"Saved {len(sequences)} sequences to {output_folder}")
    else:
        print(f"No valid sequences from {image_folder}")
    
    detector.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", default="data/pose_sequences")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    
    exercise_map = {
        "push up": "pushup",
        "pull up": "pullup",
        "squat": "squat",
        "deadlift": "deadlift",
        "tricep dips": "dip",
        "barbell biceps curl": "bicep_curl",
        "shoulder press": "shoulder_press",
        "bench press": "bench_press",
        "plank": "plank",
    }
    
    for folder_name, ex_name in exercise_map.items():
        src = os.path.join(args.dataset_root, folder_name)
        if not os.path.isdir(src):
            print(f"Warning: {src} not found – skipping")
            continue
        dst = os.path.join(args.output_root, ex_name)
        print(f"\nProcessing {folder_name} -> {ex_name}")
        extract_pose_sequence(src, dst, args.seq_len, args.stride)
    
    print("\n✅ Preprocessing complete!")

if __name__ == "__main__":
    main()