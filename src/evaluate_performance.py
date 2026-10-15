"""
Waste Classification evaluation + performance benchmark.

Run from project root:
    python3 src/evaluate_performance.py

Optional:
    python3 src/evaluate_performance.py --eval-dir data/test --max-images 500
    python3 src/evaluate_performance.py --eval-dir data/raw --max-images 100

Important:
- For meaningful classification metrics, --eval-dir must contain labeled images in
  class folders and ideally be a held-out test set not used for training/validation.
- If data/raw contains training images, metrics measured on it are NOT test accuracy.
- YOLO detection precision/recall/mAP cannot be calculated without bounding-box
  annotations. This script reports detection counts/confidence and runtime instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import models, transforms
from ultralytics import YOLO

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing scikit-learn. Install dependencies with:\n"
        "python3 -m pip install scikit-learn ultralytics psutil"
    ) from exc

try:
    import psutil
except ImportError:
    psutil = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "best_expanded_resnet18.pth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device):
    # Synchronization is necessary for accurate timing on asynchronous accelerators.
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_classifier(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = list(checkpoint["classes"])
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names


def get_label_from_path(image_path: Path, eval_dir: Path, class_names):
    """Use the first directory below eval_dir matching a checkpoint class."""
    try:
        relative_parts = image_path.relative_to(eval_dir).parts[:-1]
    except ValueError:
        relative_parts = image_path.parts[:-1]
    for part in relative_parts:
        for class_name in class_names:
            if part.casefold() == class_name.casefold():
                return class_name
    return None


def discover_images(eval_dir: Path):
    return sorted(
        p for p in eval_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_to_tensor(pil_image, device):
    return TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(device)


def predict_classifier(model, class_names, pil_image, device):
    tensor = image_to_tensor(pil_image, device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    index = int(probs.argmax().item())
    return class_names[index], probs.detach().cpu().numpy()


def report_classification_metrics(y_true, y_pred, class_names, out_dir):
    labels = list(class_names)
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "=" * 68)
    print("CLASSIFICATION METRICS")
    print("=" * 68)
    print(f"Evaluated labeled images : {len(y_true)}")
    print(f"Accuracy                 : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro precision          : {p_macro:.4f}")
    print(f"Macro recall             : {r_macro:.4f}")
    print(f"Macro F1-score           : {f_macro:.4f}")
    print(f"Weighted precision       : {p_weight:.4f}")
    print(f"Weighted recall          : {r_weight:.4f}")
    print(f"Weighted F1-score        : {f_weight:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        y_true, y_pred, labels=labels, target_names=labels, zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted; class order below):")
    print(labels)
    print(cm)

    (out_dir / "classification_report.json").write_text(
        json.dumps({
            "evaluated_images": len(y_true),
            "accuracy": acc,
            "macro_precision": p_macro,
            "macro_recall": r_macro,
            "macro_f1": f_macro,
            "weighted_precision": p_weight,
            "weighted_recall": r_weight,
            "weighted_f1": f_weight,
            "classes": labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        }, indent=2)
    )
    with (out_dir / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\predicted"] + labels)
        for name, row in zip(labels, cm.tolist()):
            writer.writerow([name] + row)
    return acc



def save_plots(y_true, y_pred, class_names, prediction_rows, class_latencies_ms,
               detection_times_ms, detections_per_image, detection_confidences,
               output_dir):
    """Save Matplotlib plots using only metrics measured during this run."""
    # 1. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(class_names))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Number of images")
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Waste Classification — Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 2. Per-class precision, recall, and F1
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(class_names), zero_division=0
    )
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, p, width, label="Precision")
    ax.bar(x, r, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")
    ax.set_xticks(x, class_names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall and F1-Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_per_class_metrics.png", dpi=160)
    plt.close(fig)

    # 3. Classifier inference latency distribution
    if len(class_latencies_ms):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(class_latencies_ms, bins=25, edgecolor="black")
        ax.set_xlabel("Inference latency (ms/image)")
        ax.set_ylabel("Number of images")
        ax.set_title("ResNet-18 Inference Latency Distribution")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_classifier_latency.png", dpi=160)
        plt.close(fig)

    # 4. YOLO detection latency distribution
    if len(detection_times_ms):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(detection_times_ms, bins=20, edgecolor="black")
        ax.set_xlabel("Detection latency (ms/image)")
        ax.set_ylabel("Number of images")
        ax.set_title("YOLO Object Detection Latency Distribution")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_yolo_latency.png", dpi=160)
        plt.close(fig)

    # 5. Detection counts per image
    if len(detections_per_image):
        counts = Counter(detections_per_image)
        keys = sorted(counts)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([str(k) for k in keys], [counts[k] for k in keys], edgecolor="black")
        ax.set_xlabel("Objects detected in an image")
        ax.set_ylabel("Number of images")
        ax.set_title("YOLO Detection Count Distribution")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "plot_detection_counts.png", dpi=160)
        plt.close(fig)

    # 6. Mean YOLO confidence by predicted detector class
    if detection_confidences:
        # Detector labels and confidence values are paired in the main loop.
        # This aggregate is plotted there when paired data are available.
        pass

    print("\nMatplotlib plots saved:")
    for filename in (
        "plot_confusion_matrix.png",
        "plot_per_class_metrics.png",
        "plot_classifier_latency.png",
        "plot_yolo_latency.png",
        "plot_detection_counts.png",
    ):
        if (output_dir / filename).exists():
            print(f"- {filename}")

def open_saved_plots(output_dir):
    """Open the PNG plots in Matplotlib only after all outputs are saved."""
    pngs = sorted(output_dir.glob("*.png"))
    if not pngs:
        print("No saved PNG plots were found to display.")
        return
    columns = 2
    rows = (len(pngs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, png_path in zip(axes, pngs):
        ax.imshow(plt.imread(png_path))
        ax.set_title(png_path.name, fontsize=10)
        ax.axis("off")
    for ax in axes[len(pngs):]:
        ax.axis("off")
    fig.suptitle(f"PERFORMANCE EVALUATION — {output_dir.name}", fontsize=14)
    fig.tight_layout()
    print("All performance plots have been saved. Opening them in Matplotlib now...")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier and benchmark pipeline.")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "raw",
                        help="Labeled image root. Expected class-named parent folders.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-images", type=int, default=0,
                        help="Maximum images to evaluate; 0 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--detector-confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-detection-images", type=int, default=100,
                        help="Number of images used for object detection benchmarking.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.eval_dir = args.eval_dir.resolve()
    args.model = args.model.resolve()
    output_root = args.output_dir.resolve() / "PERFORMANCE"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    existing = sorted(output_root.glob("PERFORMANCE_*"))
    run_number = len(existing) + 1
    args.output_dir = output_root / f"PERFORMANCE_{run_number:03d}_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=False)

    device = choose_device()
    print(f"Project root : {ROOT}")
    print(f"Evaluation dir: {args.eval_dir}")
    print(f"Model        : {args.model}")
    print(f"Device       : {device}")
    print(f"Output dir   : {args.output_dir}")

    if not args.eval_dir.exists():
        raise SystemExit(f"Evaluation directory not found: {args.eval_dir}")

    model, class_names = load_classifier(args.model, device)
    print(f"Classes      : {class_names}")

    all_images = discover_images(args.eval_dir)
    labeled = [(p, get_label_from_path(p, args.eval_dir, class_names))
               for p in all_images]
    labeled = [(p, label) for p, label in labeled if label is not None]
    if args.max_images > 0 and len(labeled) > args.max_images:
        random.shuffle(labeled)
        labeled = labeled[:args.max_images]
    if not labeled:
        raise SystemExit(
            "No labeled images found. Put images under folders named exactly like "
            f"the checkpoint classes: {class_names}"
        )

    # Warm-up so first-use setup is not mixed into steady-state latency.
    warm_image = Image.open(labeled[0][0]).convert("RGB")
    warm_tensor = image_to_tensor(warm_image, device)
    with torch.inference_mode():
        _ = model(warm_tensor)
    sync_device(device)

    y_true, y_pred = [], []
    class_latencies_ms = []
    prediction_rows = []
    confidence_values = []
    total_start = time.perf_counter()

    for index, (image_path, true_label) in enumerate(labeled, start=1):
        try:
            pil_image = Image.open(image_path).convert("RGB")
            tensor = image_to_tensor(pil_image, device)
            sync_device(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = F.softmax(logits, dim=1)[0]
            sync_device(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            pred_index = int(probabilities.argmax().item())
            predicted_label = class_names[pred_index]
            probs = probabilities.detach().cpu().numpy()
            confidence = float(probs[pred_index])
            y_true.append(true_label)
            y_pred.append(predicted_label)
            class_latencies_ms.append(elapsed_ms)
            confidence_values.append(confidence)
            prediction_rows.append({
                "image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
                "confidence": confidence,
                "inference_ms": elapsed_ms,
            })
        except Exception as exc:
            print(f"Skipping {image_path}: {exc}")
        if index % 100 == 0 or index == len(labeled):
            print(f"Classifier evaluation: {index}/{len(labeled)} images processed")

    total_elapsed = time.perf_counter() - total_start
    if not y_true:
        raise SystemExit("No images could be evaluated successfully.")

    # Save individual predictions.
    with (args.output_dir / "image_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    accuracy = report_classification_metrics(y_true, y_pred, class_names, args.output_dir)

    latencies = np.array(class_latencies_ms)
    print("\n" + "=" * 68)
    print("CLASSIFIER PERFORMANCE")
    print("=" * 68)
    print(f"Device                         : {device}")
    print(f"Successfully evaluated         : {len(y_true)} images")
    print(f"Total evaluation wall time     : {total_elapsed:.2f} s")
    print(f"Mean model inference latency   : {latencies.mean():.2f} ms/image")
    print(f"Median model inference latency : {np.median(latencies):.2f} ms/image")
    print(f"P95 model inference latency    : {np.percentile(latencies, 95):.2f} ms/image")
    print(f"Approx. model throughput       : {1000 / latencies.mean():.2f} images/s")
    print(f"End-to-end eval throughput     : {len(y_true) / total_elapsed:.2f} images/s")
    print(f"Mean predicted confidence      : {np.mean(confidence_values) * 100:.2f}%")
    if psutil is not None:
        process = psutil.Process()
        print(f"Process RSS at report time     : {process.memory_info().rss / (1024**2):.1f} MB")
    else:
        print("Process RSS                    : unavailable (install psutil)")
    if device.type == "mps":
        try:
            print(f"MPS current allocated memory   : {torch.mps.current_allocated_memory() / (1024**2):.1f} MB")
        except Exception:
            pass
    elif device.type == "cuda":
        print(f"CUDA peak allocated memory     : {torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB")

    # YOLO timing/detection statistics. No ground-truth boxes are available,
    # so do not claim precision/recall/mAP for object detection.
    print("\nLoading YOLO for detection benchmark...")
    detector = YOLO(args.yolo_model)
    det_images = discover_images(args.eval_dir)
    if args.max_detection_images > 0 and len(det_images) > args.max_detection_images:
        random.shuffle(det_images)
        det_images = det_images[:args.max_detection_images]

    detection_times_ms = []
    detections_per_image = []
    detection_confidences = []
    detector_class_counts = Counter()
    detector_errors = 0

    for index, image_path in enumerate(det_images, start=1):
        try:
            with Image.open(image_path) as im:
                rgb = np.array(im.convert("RGB"))
            sync_device(device)
            t0 = time.perf_counter()
            result = detector.predict(
                source=rgb,
                conf=args.detector_confidence,
                imgsz=args.imgsz,
                max_det=100,
                verbose=False,
                device=str(device),
            )[0]
            sync_device(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            detection_times_ms.append(elapsed_ms)
            boxes = result.boxes
            n = 0 if boxes is None else len(boxes)
            detections_per_image.append(n)
            if n:
                for box in boxes:
                    detection_confidences.append(float(box.conf[0].item()))
                    detector_class_counts[result.names[int(box.cls[0].item())]] += 1
        except Exception as exc:
            detector_errors += 1
            print(f"YOLO skipped {image_path.name}: {exc}")
        if index % 25 == 0 or index == len(det_images):
            print(f"YOLO benchmark: {index}/{len(det_images)} images processed")

    if detection_times_ms:
        det_lat = np.array(detection_times_ms)
        print("\n" + "=" * 68)
        print("YOLO OBJECT-DETECTION PERFORMANCE (not detection accuracy)")
        print("=" * 68)
        print(f"Images benchmarked              : {len(detection_times_ms)}")
        print(f"Detection errors                : {detector_errors}")
        print(f"Mean detector latency           : {det_lat.mean():.2f} ms/image")
        print(f"Median detector latency         : {np.median(det_lat):.2f} ms/image")
        print(f"P95 detector latency            : {np.percentile(det_lat, 95):.2f} ms/image")
        print(f"Detector throughput              : {1000 / det_lat.mean():.2f} images/s")
        print(f"Images with >=1 detection       : {sum(n > 0 for n in detections_per_image)}")
        print(f"Images with zero detections     : {sum(n == 0 for n in detections_per_image)}")
        print(f"Total boxes predicted           : {sum(detections_per_image)}")
        print(f"Mean detections per image       : {np.mean(detections_per_image):.2f}")
        if detection_confidences:
            print(f"Mean YOLO detection confidence  : {np.mean(detection_confidences) * 100:.2f}%")
        print(f"Most common YOLO labels         : {detector_class_counts.most_common(10)}")
    else:
        det_lat = np.array([])
        print("YOLO benchmark produced no successful timings.")

    save_plots(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        prediction_rows=prediction_rows,
        class_latencies_ms=class_latencies_ms,
        detection_times_ms=detection_times_ms,
        detections_per_image=detections_per_image,
        detection_confidences=detection_confidences,
        output_dir=args.output_dir,
    )

    summary = {
        "evaluation_directory": str(args.eval_dir),
        "model_checkpoint": str(args.model),
        "device": str(device),
        "classifier": {
            "evaluated_images": len(y_true),
            "accuracy": accuracy,
            "mean_inference_ms": float(latencies.mean()),
            "median_inference_ms": float(np.median(latencies)),
            "p95_inference_ms": float(np.percentile(latencies, 95)),
            "model_throughput_images_per_second": float(1000 / latencies.mean()),
            "end_to_end_eval_images_per_second": float(len(y_true) / total_elapsed),
        },
        "yolo_detection_benchmark": {
            "images_benchmarked": len(detection_times_ms),
            "mean_detector_latency_ms": float(np.mean(detection_times_ms)) if detection_times_ms else None,
            "median_detector_latency_ms": float(np.median(detection_times_ms)) if detection_times_ms else None,
            "p95_detector_latency_ms": float(np.percentile(detection_times_ms, 95)) if detection_times_ms else None,
            "total_predicted_boxes": int(sum(detections_per_image)),
            "images_with_detections": int(sum(n > 0 for n in detections_per_image)),
            "images_with_zero_detections": int(sum(n == 0 for n in detections_per_image)),
            "mean_detections_per_image": float(np.mean(detections_per_image)) if detections_per_image else None,
            "mean_detection_confidence": float(np.mean(detection_confidences)) if detection_confidences else None,
            "detector_label_counts": dict(detector_class_counts),
            "note": "Precision, recall and mAP require ground-truth bounding-box annotations.",
        },
        "warning": (
            "Metrics are only valid as held-out test metrics if eval-dir is a separate "
            "test split not used during training or model selection."
        ),
    }
    (args.output_dir / "performance_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nReports saved to: {args.output_dir}")
    print("- classification_report.json")
    print("- confusion_matrix.csv")
    print("- image_predictions.csv")
    print("- performance_summary.json")
    print("\nIMPORTANT: If you evaluated data/raw and it contains training/validation images, "
          "do not report this accuracy as held-out test accuracy.")
    print("\nOpening the saved performance plots in Matplotlib...")
    open_saved_plots(args.output_dir)


if __name__ == "__main__":
    main()
