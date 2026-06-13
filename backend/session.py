import time
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.core.config import DATABASE_URL

# SQLAlchemy setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    age = Column(Integer)
    weight_kg = Column(Float)
    height_cm = Column(Float)
    sex = Column(String)
    goal = Column(String)

class WorkoutSessionDB(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_sec = Column(Float)
    total_calories = Column(Float)
    exercises_json = Column(JSON)  # stores reps per exercise

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# --------------------------------------------------
# Calorie estimation using MET
# --------------------------------------------------
MET_VALUES = {
    "pushup": 3.8,
    "pullup": 5.0,
    "squat": 4.0,
    "deadlift": 4.5,
    "lunge": 4.0,
    "dip": 4.2,
    "bicep_curl": 3.0,
    "shoulder_press": 3.5,
    "bench_press": 3.5,
    "plank": 2.8,
}

def estimate_calories(exercise: str, weight_kg: float, duration_min: float, intensity_factor=1.0):
    """MET = metabolic equivalent of task. Calories per minute = MET * 3.5 * weight_kg / 200."""
    met = MET_VALUES.get(exercise, 3.5) * intensity_factor
    cal_per_min = met * 3.5 * weight_kg / 200
    return cal_per_min * duration_min

# --------------------------------------------------
# WorkoutSession class (original plus DB)
# --------------------------------------------------
class WorkoutSession:
    def __init__(self, user_id="default", user_weight=70):
        self.user_id = user_id
        self.user_weight = user_weight
        self.start_time = time.time()
        self.reps_per_exercise = {}
        self.current_exercise = None
        self.calories_burned = 0.0
        self.duration = 0.0
        self.history = []  # store (exercise, rep_count) for each set

    def start_exercise(self, exercise_name):
        self.current_exercise = exercise_name
        if exercise_name not in self.reps_per_exercise:
            self.reps_per_exercise[exercise_name] = 0

    def add_rep(self, exercise_name, intensity=1.0):
        if exercise_name in self.reps_per_exercise:
            self.reps_per_exercise[exercise_name] += 1
            # Estimate calories per rep (approx 0.4 kcal for average, but now MET‑based)
            # We'll sum at end, but for live display we approximate
            self.calories_burned += 0.4 * intensity

    def end_session(self):
        self.duration = time.time() - self.start_time
        # Recalculate calories using MET for each exercise
        total_cal = 0.0
        for ex, reps in self.reps_per_exercise.items():
            # Assume 2 seconds per rep
            duration_min = reps * 2 / 60
            total_cal += estimate_calories(ex, self.user_weight, duration_min)
        self.calories_burned = total_cal

        session_data = {
            "user_id": self.user_id,
            "date": datetime.now().isoformat(),
            "duration_sec": self.duration,
            "total_calories": self.calories_burned,
            "reps": self.reps_per_exercise
        }
        # Save to PostgreSQL
        db_session = SessionLocal()
        db_session.add(WorkoutSessionDB(
            user_id=int(self.user_id) if self.user_id.isdigit() else 1,
            start_time=datetime.fromtimestamp(self.start_time),
            end_time=datetime.now(),
            duration_sec=self.duration,
            total_calories=self.calories_burned,
            exercises_json=self.reps_per_exercise
        ))
        db_session.commit()
        db_session.close()

        # Also keep JSON backup
        os.makedirs("data", exist_ok=True)
        with open("data/sessions.json", "a") as f:
            f.write(json.dumps(session_data) + "\n")
        return session_data

    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {
            "duration": elapsed,
            "calories": self.calories_burned,
            "reps": self.reps_per_exercise.copy()
        }