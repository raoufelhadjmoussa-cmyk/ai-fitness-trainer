import numpy as np
import torch
from torch.utils.data import Dataset
import os

class PoseSequenceDataset(Dataset):
    EXERCISE_MAP = {
        "pushup": 0, "pullup": 1, "squat": 2, "deadlift": 3,
        "lunge": 4, "dip": 5, "bicep_curl": 6, "shoulder_press": 7,
        "bench_press": 8, "plank": 9, "unknown": 10
    }
    INV_MAP = {v: k for k, v in EXERCISE_MAP.items()}
    
    def __init__(self, data_dir, seq_len=30):
        self.seq_len = seq_len
        self.sequences = []
        self.labels = []
        
        for ex_name, label_id in self.EXERCISE_MAP.items():
            folder = os.path.join(data_dir, ex_name)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if file.endswith('.npy'):
                    seq = np.load(os.path.join(folder, file))
                    if seq.shape[0] == seq_len:
                        self.sequences.append(seq)
                        self.labels.append(label_id)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)