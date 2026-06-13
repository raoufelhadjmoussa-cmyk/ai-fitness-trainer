import torch
import numpy as np
from pathlib import Path

# --- LSTM Model (same as before, but now inside classifier) ---
class ExerciseLSTM(torch.nn.Module):
    def __init__(self, input_dim=132, hidden_dim=128, num_layers=2, num_classes=11, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, num_classes)
        )
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

class ExerciseClassifier:
    EXERCISE_MAP = {
        "pushup": 0, "pullup": 1, "squat": 2, "deadlift": 3,
        "lunge": 4, "dip": 5, "bicep_curl": 6, "shoulder_press": 7,
        "bench_press": 8, "plank": 9, "unknown": 10
    }
    INV_MAP = {v: k for k, v in EXERCISE_MAP.items()}

    def __init__(self, model_path="models/exercise_lstm.pth", seq_len=30, device="cpu"):
        self.device = device
        self.seq_len = seq_len
        self.model = ExerciseLSTM(num_classes=11)
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.buffer = []  # stores flattened landmarks for last seq_len frames

    def classify(self, landmarks):
        """
        landmarks: list of 33 points [id, x, y, z, visibility]
        Returns exercise name string
        """
        if not landmarks or len(landmarks) < 33:
            return "unknown"
        # Convert to feature vector: 33 * 4 = 132 (x,y,z,visibility)
        features = []
        for lm in landmarks:
            features.extend([lm[1], lm[2], lm[3], lm[4]])  # x,y,z,vis
        self.buffer.append(features)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
        if len(self.buffer) == self.seq_len:
            seq = np.array(self.buffer, dtype=np.float32)  # (seq_len, 132)
            seq_tensor = torch.tensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(seq_tensor)
                pred = torch.argmax(logits, dim=1).item()
            return self.INV_MAP.get(pred, "unknown")
        return "unknown"