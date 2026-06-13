"""
Train LSTM exercise classifier from recorded .npy sequences.
Usage: python scripts/train.py --data_dir data/pose_sequences --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add backend to path
import sys
sys.path.append(".")
from backend.recognition.dataset import PoseSequenceDataset
from backend.recognition.models.lstm_model import ExerciseLSTM

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = PoseSequenceDataset(args.data_dir, seq_len=args.seq_len)
    print(f"Total samples: {len(dataset)}")
    if len(dataset) == 0:
        print("No data found. Run record_sequences.py first.")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = ExerciseLSTM(
        input_dim=33*4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(dataset.EXERCISE_MAP),
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # Early stopping & checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"Checkpoint saved to {args.save_path} (acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Final evaluation on validation set
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [dataset.INV_MAP[i] for i in range(len(dataset.EXERCISE_MAP))]
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.show()

    # Save training curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training Curves")
    plt.savefig("models/training_curves.png")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open("models/classification_report.txt", "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/pose_sequences", help="Folder with exercise subfolders")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="models/exercise_lstm.pth")
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    train(args)