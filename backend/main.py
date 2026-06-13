# main.py
import cv2
import streamlit as st
from pose_detector import PoseDetector
from exercise_analyzer import ExerciseAnalyzer
from rep_counter import RepCounter
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

def provide_voice_feedback(message):
    engine.say(message)
    engine.runAndWait()

def main():
    st.title("AI Fitness Trainer")
    
    # Sidebar for exercise selection
    exercise_options = ["auto", "pushup", "squat", "deadlift", "pullup", "dip", "lunge"]
    selected_exercise = st.sidebar.selectbox("Select Exercise (or auto-detect)", exercise_options)
    
    # Initialize components
    detector = PoseDetector()
    analyzer = ExerciseAnalyzer(exercise_config)
    rep_counter = RepCounter()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    # Create video placeholder
    frame_placeholder = st.empty()
    
    # Stats display
    col1, col2, col3 = st.columns(3)
    rep_display = col1.empty()
    feedback_display = col2.empty()
    form_score = col3.empty()
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        
        # Detect pose
        img = detector.find_pose(img)
        landmarks = detector.get_landmarks(img)
        
        if landmarks:
            # Analyze based on selected exercise
            if selected_exercise == "pushup":
                analysis = analyzer.analyze_pushup(landmarks)
            elif selected_exercise == "squat":
                analysis = analyzer.analyze_squat(landmarks)
            else:
                # Auto-detection logic
                analysis = auto_detect_exercise(landmarks, analyzer)
            
            # Update rep counter
            if analysis["exercise"] == "pushup":
                rep_update = rep_counter.update(analysis["elbow_angle"])
            elif analysis["exercise"] == "squat":
                rep_update = rep_counter.update(analysis["knee_angle"])
            
            # Display feedback
            feedback_text = " | ".join(analysis["feedback"])
            if rep_update["new_rep"]:
                provide_voice_feedback(f"Rep {rep_update['count']}")
            
            # Update UI
            rep_display.metric("Reps", rep_update["count"])
            feedback_display.write(f"💡 {feedback_text}")
            form_score.progress(int(analysis.get("is_valid", 0.5) * 100))
        
        # Show video feed
        frame_placeholder.image(img, channels="BGR", use_container_width=True)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()