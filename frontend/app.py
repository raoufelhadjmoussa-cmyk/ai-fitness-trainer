import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pose_detector import get_pose_detector
from backend.classifier import ExerciseClassifier
from backend.rep_counter import RepCounter
from backend.exercise_config import EXERCISE_THRESHOLDS
from backend.feedback import FeedbackGenerator
from backend.voice import VoiceCoach
from backend.session import WorkoutSession
from backend.analysis.fatigue import FatigueEstimator
from backend.analysis.motion_quality import MotionAnalyzer
from backend.utils import calculate_angle

# ------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------
st.set_page_config(page_title="My Trainer", layout="wide")
st.title("ðŸ‹ï¸ My Trainer")
st.markdown("*Your personal AI fitness coach*")

# ------------------------------------------------------------------
# Initialize session state keys
# ------------------------------------------------------------------
if "detector" not in st.session_state:
    st.session_state["detector"] = None
if "classifier" not in st.session_state:
    st.session_state["classifier"] = None
if "feedback_gen" not in st.session_state:
    st.session_state["feedback_gen"] = None
if "voice" not in st.session_state:
    st.session_state["voice"] = None
if "rep_counters" not in st.session_state:
    st.session_state["rep_counters"] = {}
if "frame_count" not in st.session_state:
    st.session_state["frame_count"] = 0
if "running" not in st.session_state:
    st.session_state["running"] = False
if "session" not in st.session_state:
    st.session_state["session"] = None
if "fatigue" not in st.session_state:
    st.session_state["fatigue"] = None
if "motion" not in st.session_state:
    st.session_state["motion"] = None
if "last_exercise" not in st.session_state:
    st.session_state["last_exercise"] = None
if "video_mode" not in st.session_state:
    st.session_state["video_mode"] = "camera"
if "video_path" not in st.session_state:
    st.session_state["video_path"] = None
if "video_cap" not in st.session_state:
    st.session_state["video_cap"] = None
if "stop_camera" not in st.session_state:
    st.session_state["stop_camera"] = False

# ------------------------------------------------------------------
# Sidebar â€“ user profile and controls
# ------------------------------------------------------------------
st.sidebar.header("ðŸ‘¤ User Profile")
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
age = st.sidebar.number_input("Age", 15, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
goal = st.sidebar.selectbox("Goal", ["Muscle Gain", "Weight Loss", "Strength", "Endurance"])
detector_type = st.sidebar.selectbox("Pose Detector", ["mediapipe", "yolo"], index=0)
mode = st.sidebar.radio("Exercise Mode", ["Auto-detect", "Manual"])
manual_exercise = st.sidebar.selectbox("Choose exercise", list(EXERCISE_THRESHOLDS.keys())) if mode == "Manual" else None

# Input source selection
input_source = st.sidebar.radio("Input Source", ["ðŸ“· Live Camera", "ðŸŽ¥ Upload Video"])
if input_source == "ðŸ“· Live Camera":
    st.session_state["video_mode"] = "camera"
else:
    st.session_state["video_mode"] = "file"
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        st.session_state["video_path"] = tfile.name
        st.sidebar.success("âœ… Video loaded")
        if st.session_state["video_cap"] is not None:
            st.session_state["video_cap"].release()
        st.session_state["video_cap"] = cv2.VideoCapture(st.session_state["video_path"])
    else:
        st.sidebar.info("Please upload a video file.")

# Session controls
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("â–¶ï¸ Start Session"):
        st.session_state["session"] = WorkoutSession(user_id="1", user_weight=weight)
        st.session_state["running"] = True
        st.session_state["fatigue"] = FatigueEstimator()
        st.session_state["motion"] = MotionAnalyzer()
        st.session_state["rep_counters"] = {}
        st.session_state["last_exercise"] = None
        st.success("Session started!")
with col_btn2:
    if st.button("â¹ï¸ End Session"):
        if st.session_state.get("session") is not None:
            summary = st.session_state["session"].end_session()
            st.success(f"Session saved! Calories: {summary['total_calories']:.0f}")
            st.session_state["running"] = False

# Camera stop button (only when camera mode)
if st.session_state["video_mode"] == "camera":
    if st.sidebar.button("ðŸ›‘ Stop Camera"):
        st.session_state["stop_camera"] = True

# ------------------------------------------------------------------
# Lazy initialization of heavy components
# ------------------------------------------------------------------
if st.session_state["detector"] is None:
    st.session_state["detector"] = get_pose_detector(detector_type)
if st.session_state["classifier"] is None:
    st.session_state["classifier"] = ExerciseClassifier()
if st.session_state["feedback_gen"] is None:
    st.session_state["feedback_gen"] = FeedbackGenerator(EXERCISE_THRESHOLDS)
if st.session_state["voice"] is None:
    st.session_state["voice"] = VoiceCoach()

# ------------------------------------------------------------------
# Layout columns for video and stats
# ------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
stats_placeholder = col2.empty()
feedback_placeholder = col2.empty()

# ------------------------------------------------------------------
# Helper function to process a single frame
# ------------------------------------------------------------------
def process_frame(frame, frame_num):
    frame = cv2.flip(frame, 1)
    frame = st.session_state["detector"].find_pose(frame, draw=True)
    landmarks = st.session_state["detector"].get_landmarks(frame)

    if landmarks and len(landmarks) >= 33:
        if mode == "Auto-detect":
            exercise = st.session_state["classifier"].classify(landmarks)
        else:
            exercise = manual_exercise

        if exercise != "unknown":
            # Reset rep counter when exercise changes
            if st.session_state["last_exercise"] != exercise:
                st.session_state["rep_counters"][exercise] = RepCounter()
                st.session_state["last_exercise"] = exercise

            if exercise not in st.session_state["rep_counters"]:
                st.session_state["rep_counters"][exercise] = RepCounter()
            rep_counter = st.session_state["rep_counters"][exercise]

            thresholds = EXERCISE_THRESHOLDS.get(exercise, {})
            key_angle_name = thresholds.get("key_angle", "elbow")

            def get_angle_by_name(name):
                if name == "elbow":
                    left = calculate_angle(
                        (landmarks[11][1], landmarks[11][2]),
                        (landmarks[13][1], landmarks[13][2]),
                        (landmarks[15][1], landmarks[15][2])
                    )
                    right = calculate_angle(
                        (landmarks[12][1], landmarks[12][2]),
                        (landmarks[14][1], landmarks[14][2]),
                        (landmarks[16][1], landmarks[16][2])
                    )
                    return (left + right) / 2
                elif name == "knee":
                    left = calculate_angle(
                        (landmarks[23][1], landmarks[23][2]),
                        (landmarks[25][1], landmarks[25][2]),
                        (landmarks[27][1], landmarks[27][2])
                    )
                    right = calculate_angle(
                        (landmarks[24][1], landmarks[24][2]),
                        (landmarks[26][1], landmarks[26][2]),
                        (landmarks[28][1], landmarks[28][2])
                    )
                    return (left + right) / 2
                elif name == "hip":
                    left = calculate_angle(
                        (landmarks[11][1], landmarks[11][2]),
                        (landmarks[23][1], landmarks[23][2]),
                        (landmarks[25][1], landmarks[25][2])
                    )
                    return left
                return 90

            angle = get_angle_by_name(key_angle_name)
            down_th = thresholds.get("down_threshold", 90)
            up_th = thresholds.get("up_threshold", 160)

            new_rep, count, state = rep_counter.update(angle, down_th, up_th)
            if new_rep:
                st.session_state["voice"].speak(f"Rep {count}")
                if st.session_state.get("running") and st.session_state.get("session") is not None:
                    intensity = st.session_state["fatigue"].get_intensity_factor() if st.session_state["fatigue"] else 1.0
                    st.session_state["session"].add_rep(exercise, intensity)
                cv2.putText(frame, "REP!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            feedback_msgs, form_score = st.session_state["feedback_gen"].analyze_form(exercise, landmarks)
            if st.session_state["fatigue"] is not None:
                fatigue_score = st.session_state["fatigue"].update(landmarks, exercise)
            else:
                fatigue_score = 0
            if st.session_state["motion"] is not None:
                motion_score = st.session_state["motion"].update(landmarks)
            else:
                motion_score = 100

            if fatigue_score > 80:
                st.session_state["voice"].speak("High fatigue â€“ consider resting")

            cv2.putText(frame, f"{exercise} | Reps: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Form: {form_score:.0f}  Motion: {motion_score:.0f}  Fatigue: {fatigue_score:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if st.session_state.get("running") and st.session_state.get("session") is not None:
                stats = st.session_state["session"].get_stats()
                stats_placeholder.metric("â±ï¸ Duration", f"{stats['duration']:.0f}s")
                stats_placeholder.metric("ðŸ”¥ Calories", f"{stats['calories']:.0f}")
                stats_placeholder.metric("ðŸ˜“ Fatigue", f"{fatigue_score:.0f}")
                stats_placeholder.write("**ðŸ“Š Reps per exercise:**")
                for ex, r in stats["reps"].items():
                    stats_placeholder.write(f"- {ex.capitalize()}: {r}")
            else:
                stats_placeholder.info("Start a session to track your workout")

            feedback_placeholder.write("\n".join(feedback_msgs[:2]) if feedback_msgs else "âœ… Form looks good!")

    return frame

# ------------------------------------------------------------------
# Main execution based on input source
# ------------------------------------------------------------------
if st.session_state["video_mode"] == "camera":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        st.stop()

    run = st.checkbox("ðŸŽ¥ Start Camera", value=True)
    if run:
        st.session_state["stop_camera"] = False

    while run and cap.isOpened() and not st.session_state["stop_camera"]:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, 0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        time.sleep(0.03)

    cap.release()
    if st.session_state["stop_camera"]:
        st.session_state["stop_camera"] = False
        st.info("Camera stopped.")
else:
    if st.session_state["video_cap"] is None or not st.session_state["video_cap"].isOpened():
        st.warning("Please upload a video file first.")
    else:
        play_video = st.button("â–¶ï¸ Process Video")
        stop_video = st.button("â¹ï¸ Stop")
        if play_video:
            cap = st.session_state["video_cap"]
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 0.03
            frame_num = 0
            while cap.isOpened() and not stop_video:
                ret, frame = cap.read()
                if not ret:
                    st.info("End of video")
                    break
                frame = process_frame(frame, frame_num)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                time.sleep(frame_delay)
                frame_num += 1
            cap.release()
            if st.session_state["video_path"]:
                st.session_state["video_cap"] = cv2.VideoCapture(st.session_state["video_path"])

cv2.destroyAllWindows()
