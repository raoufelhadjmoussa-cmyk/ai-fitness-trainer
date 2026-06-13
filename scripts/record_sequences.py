"""
Record pose sequences from webcam for training the LSTM classifier.
Saves .npy files organized by exercise folder.
Usage: python scripts/record_sequences.py --exercise pushup --samples 100
"""

import cv2
import numpy as np
import os
import argparse
import sys
sys.path.append(".")
from backend.pose_detector import MediaPipeDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", type=str, required=True, help="Exercise name (pushup, squat, ...)")
    parser.add_argument("--samples", type=int, default=100, help="Number of sequences to record")
    parser.add_argument("--seq_len", type=int, default=30, help="Frames per sequence")
    args = parser.parse_args()

    # Create output directory
    out_dir = f"data/pose_sequences/{args.exercise}"
    os.makedirs(out_dir, exist_ok=True)

    detector = MediaPipeDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print(f"Recording {args.samples} sequences for '{args.exercise}'")
    print("Press SPACE to start recording a sequence, ESC to quit")

    sequence = []
    recording = False
    recorded = 0

    while recorded < args.samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks(frame)

        if landmarks and len(landmarks) >= 33:
            # Extract features: x,y,z,visibility for 33 points
            features = []
            for lm in landmarks:
                features.extend([lm[1], lm[2], lm[3], lm[4]])  # x,y,z,vis
            if recording:
                sequence.append(features)
                if len(sequence) >= args.seq_len:
                    # Save sequence
                    seq_array = np.array(sequence[:args.seq_len], dtype=np.float32)
                    filename = f"{out_dir}/{args.exercise}_{recorded+1}.npy"
                    np.save(filename, seq_array)
                    print(f"Saved {filename}")
                    recorded += 1
                    recording = False
                    sequence = []
            # Show status on frame
            status = "RECORDING" if recording else "READY"
            cv2.putText(frame, f"{args.exercise} | {status} | {recorded}/{args.samples}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "No pose detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Record Pose Sequences", frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            if not recording and landmarks:
                recording = True
                sequence = []
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recording finished.")

if __name__ == "__main__":
    main()