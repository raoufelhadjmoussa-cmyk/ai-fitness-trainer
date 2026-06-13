"""
Evaluate LSTM model on test set.
Usage: python scripts/evaluate.py --model_path models/exercise_lstm.pth --data_dir data/pose_sequences
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import argparse
import sys
sys.path.append(".")
from backend.recognition.dataset import PoseSequenceDataset
from backend.recognition.models.lstm_model import ExerciseLSTM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PoseSequenceDataset(args.data_dir, seq_len=args.seq_len)
    # Use all data as test (or split if needed)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = ExerciseLSTM(input_dim=33*4, num_classes=len(dataset.EXERCISE_MAP))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    class_names = [dataset.INV_MAP[i] for i in range(len(dataset.EXERCISE_MAP))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Test Set")
    plt.savefig("models/eval_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/exercise_lstm.pth")
    parser.add_argument("--data_dir", type=str, default="data/pose_sequences")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    args = parser.parse_args()
    evaluate(args)