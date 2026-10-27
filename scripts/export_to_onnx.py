"""
Model export script: Converts PyTorch ResNet-18 and YOLOv8n to ONNX format
for on-device offline inference in the Android app.

Outputs placed directly into android/app/src/main/assets/
- resnet18.onnx
- yolov8n.onnx
- waste_labels.txt
"""

import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"
YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

# Prefer expanded checkpoint, fallback to finetuned
RESNET_CHECKPOINT = MODELS_DIR / "best_expanded_resnet18.pth"
if not RESNET_CHECKPOINT.exists():
    RESNET_CHECKPOINT = MODELS_DIR / "best_finetuned_resnet18.pth"


class SoftmaxResNetWrapper(nn.Module):
    """Wraps ResNet18 to append Softmax, yielding direct probabilities in ONNX."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base_model(x)
        return self.softmax(logits)


def export_resnet(output_dir: Path):
    print(f"Loading ResNet checkpoint: {RESNET_CHECKPOINT}")
    checkpoint = torch.load(RESNET_CHECKPOINT, map_location="cpu")
    classes = checkpoint.get("classes", ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"])
    print(f"Classes ({len(classes)}): {classes}")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(classes))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapped_model = SoftmaxResNetWrapper(model)
    wrapped_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    output_onnx_path = output_dir / "resnet18.onnx"

    print(f"Exporting ResNet-18 to ONNX: {output_onnx_path}")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )
    print("ResNet-18 export complete.")

    labels_path = output_dir / "waste_labels.txt"
    with open(labels_path, "w") as f:
        for c in classes:
            f.write(f"{c}\n")
    print(f"Saved class labels to: {labels_path}")


def export_yolo(output_dir: Path):
    from ultralytics import YOLO

    print(f"Loading YOLO weights: {YOLO_WEIGHTS}")
    model = YOLO(str(YOLO_WEIGHTS))

    print("Exporting YOLOv8n to ONNX (imgsz=640)...")
    exported_path = model.export(format="onnx", imgsz=640, opset=12)
    exported_path = Path(exported_path)

    target_yolo_path = output_dir / "yolov8n.onnx"
    shutil.copyfile(exported_path, target_yolo_path)
    print(f"Copied YOLOv8n ONNX model to: {target_yolo_path}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target assets directory: {ASSETS_DIR}")

    export_resnet(ASSETS_DIR)

    try:
        export_yolo(ASSETS_DIR)
    except Exception as e:
        print(f"YOLO export failed or needs onnx package: {e}")
        print("Note: YOLO can also be exported once 'onnx' finishes installing.")

    print("\nSummary of files in assets folder:")
    for f in ASSETS_DIR.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f" - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
