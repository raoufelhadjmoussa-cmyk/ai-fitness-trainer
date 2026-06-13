"""
Export trained LSTM model to ONNX format.
Usage: python scripts/export_onnx.py --model_path models/exercise_lstm.pth --output models/exercise_lstm.onnx
"""

import torch
import argparse
import sys
sys.path.append(".")
from backend.recognition.models.lstm_model import ExerciseLSTM

def export(args):
    device = "cpu"
    model = ExerciseLSTM(input_dim=33*4, num_classes=11)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, args.seq_len, 33*4)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'},
                      'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/exercise_lstm.pth")
    parser.add_argument("--output", type=str, default="models/exercise_lstm.onnx")
    parser.add_argument("--seq_len", type=int, default=30)
    args = parser.parse_args()
    export(args)