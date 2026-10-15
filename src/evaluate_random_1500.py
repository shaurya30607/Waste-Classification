"""
Random 1,500-image evaluation with timestamped report folder and saved plots.

Run from the project root:
    python3 src/evaluate_random_1500.py

Optional:
    python3 src/evaluate_random_1500.py --sample-size 1500
    python3 src/evaluate_random_1500.py --eval-dir data/raw
    python3 src/evaluate_random_1500.py --seed 42

Important:
- Images must be inside class-named folders matching the model checkpoint classes.
- If you sample from data/raw and those images were used for training/model selection,
  the classification metrics are not unbiased held-out test metrics.
- Detection precision/recall/mAP require bounding-box ground truth and are not claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

try:
    import psutil
except ImportError:
    psutil = None

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing scikit-learn. Install with:\n"
        "python3 -m pip install scikit-learn ultralytics matplotlib psutil"
    ) from exc


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
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_classifier(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = list(checkpoint["classes"])
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names


def find_label(image_path, root, class_names):
    relative_dirs = image_path.relative_to(root).parts[:-1]
    # Use the first matching class directory from the path.
    for part in relative_dirs:
        for class_name in class_names:
            if part.casefold() == class_name.casefold():
                return class_name
    return None


def discover_labeled_images(root, class_names):
    items = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = find_label(path, root, class_names)
            if label is not None:
                items.append((path, label))
    return items


def classifier_predict(model, class_names, image, device):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]
    pred_index = int(probabilities.argmax().item())
    return class_names[pred_index], probabilities.detach().cpu().numpy()


def save_plots(run_dir, class_names, y_true, y_pred, classifier_latencies,
               yolo_latencies, boxes_per_image, detector_labels, detector_confidences,
               overall_summary):
    # 1. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Image count")
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Classification Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(run_dir / "01_confusion_matrix.png", dpi=180)
    plt.close(fig)

    # 2. Per-class precision/recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    x = np.arange(len(class_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")
    ax.set_xticks(x, class_names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall and F1-Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "02_classification_metrics.png", dpi=180)
    plt.close(fig)

    # 3. Classifier latency histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(classifier_latencies, bins=30, edgecolor="black")
    ax.set_xlabel("Latency (milliseconds per image)")
    ax.set_ylabel("Number of images")
    ax.set_title("ResNet-18 Inference Latency")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "03_classifier_latency.png", dpi=180)
    plt.close(fig)

    # 4. YOLO latency histogram
    if len(yolo_latencies):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(yolo_latencies, bins=25, edgecolor="black")
        ax.set_xlabel("Latency (milliseconds per image)")
        ax.set_ylabel("Number of images")
        ax.set_title("YOLO Object Detection Latency")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_dir / "04_yolo_latency.png", dpi=180)
        plt.close(fig)

    # 5. Number of detections per image
    if len(boxes_per_image):
        counts = Counter(boxes_per_image)
        keys = sorted(counts)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([str(k) for k in keys], [counts[k] for k in keys], edgecolor="black")
        ax.set_xlabel("Objects detected in an image")
        ax.set_ylabel("Number of images")
        ax.set_title("YOLO Detection Count Distribution")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_dir / "05_detection_counts.png", dpi=180)
        plt.close(fig)

    # 6. YOLO detector label frequencies (general object labels, not waste categories)
    if detector_labels:
        labels = sorted(detector_labels.items(), key=lambda item: item[1], reverse=True)[:12]
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [item[0] for item in labels]
        values = [item[1] for item in labels]
        ax.barh(names[::-1], values[::-1], edgecolor="black")
        ax.set_xlabel("Number of predicted boxes")
        ax.set_title("Most Frequent YOLO Detector Labels")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_dir / "06_yolo_label_counts.png", dpi=180)
        plt.close(fig)

    # 7. Key performance summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        ["Mean", "Median", "P95"],
        [
            overall_summary["classifier_performance"]["mean_latency_ms"],
            overall_summary["classifier_performance"]["median_latency_ms"],
            overall_summary["classifier_performance"]["p95_latency_ms"],
        ],
        edgecolor="black",
    )
    axes[0].set_title("Classifier Latency")
    axes[0].set_ylabel("Milliseconds / image")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        ["Mean", "Median", "P95"],
        [
            overall_summary["yolo"]["mean_latency_ms"],
            overall_summary["yolo"]["median_latency_ms"],
            overall_summary["yolo"]["p95_latency_ms"],
        ],
        edgecolor="black",
    )
    axes[1].set_title("YOLO Latency")
    axes[1].set_ylabel("Milliseconds / image")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Inference Performance Summary")
    fig.tight_layout()
    fig.savefig(run_dir / "07_performance_summary.png", dpi=180)
    plt.close(fig)


def open_plots(run_dir):
    """Display the already-saved PNG plots in a Matplotlib window."""
    pngs = sorted(run_dir.glob("*.png"))
    if not pngs:
        return

    # Display after every PNG has been saved. Use a grid so all figures are
    # visible together in one Matplotlib window.
    count = len(pngs)
    columns = 2
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, png_path in zip(axes, pngs):
        ax.imshow(plt.imread(png_path))
        ax.set_title(png_path.name, fontsize=10)
        ax.axis("off")

    for ax in axes[len(pngs):]:
        ax.axis("off")

    fig.suptitle(f"Evaluation plots — {run_dir.name}", fontsize=14)
    fig.tight_layout()
    print("\nAll plots have been saved as PNG files.")
    print("Opening saved plots in a Matplotlib window now...")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for repeatable random sampling.")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--detector-confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-detection-images", type=int, default=1500,
                        help="Max sampled images used for YOLO timing; default uses the whole sample.")
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    model_path = args.model.resolve()
    if not eval_dir.exists():
        raise SystemExit(f"Evaluation directory not found: {eval_dir}")
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be >= 1")

    # Unique run folder: sequence number + local timestamp.
    output_root = ROOT / "outputs" / "evaluation" / "RANDOM_1500"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    existing = sorted(p.name for p in output_root.glob("RANDOM1500_*"))
    run_number = len(existing) + 1
    run_dir = output_root / f"RANDOM1500_{run_number:03d}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    device = choose_device()
    print("=" * 72)
    print("WASTE CLASSIFICATION — RANDOM 1500 EVALUATION")
    print("=" * 72)
    print(f"Run folder       : {run_dir}")
    print(f"Evaluation data  : {eval_dir}")
    print(f"Checkpoint       : {model_path}")
    print(f"Device           : {device}")
    print(f"Requested sample : {args.sample_size} images")

    model, class_names = load_classifier(model_path, device)
    print(f"Classes          : {class_names}")

    candidates = discover_labeled_images(eval_dir, class_names)
    if not candidates:
        raise SystemExit(
            "No labeled images found under the evaluation directory. Expected folders "
            f"matching model classes: {class_names}"
        )
    sample_size = min(args.sample_size, len(candidates))
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, sample_size)
    print(f"Available labeled images: {len(candidates)}")
    print(f"Randomly sampled images : {len(selected)}")

    # Save sample manifest so the run can be reproduced/audited.
    with (run_dir / "sample_manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label"])
        for image_path, label in selected:
            writer.writerow([str(image_path), label])

    # Warm-up before measuring latency.
    try:
        warm_image = Image.open(selected[0][0]).convert("RGB")
        warm_tensor = TRANSFORM(warm_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            _ = model(warm_tensor)
        sync_device(device)
    except Exception:
        pass

    y_true, y_pred = [], []
    classifier_latencies = []
    confidence_scores = []
    rows = []
    class_start = time.perf_counter()

    for i, (image_path, true_label) in enumerate(selected, start=1):
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = TRANSFORM(image).unsqueeze(0).to(device)
            sync_device(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
            sync_device(device)
            latency_ms = (time.perf_counter() - t0) * 1000
            pred_index = int(probs.argmax().item())
            pred_label = class_names[pred_index]
            confidence = float(probs[pred_index].detach().cpu().item())
            y_true.append(true_label)
            y_pred.append(pred_label)
            classifier_latencies.append(latency_ms)
            confidence_scores.append(confidence)
            rows.append({
                "image_path": str(image_path),
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": true_label == pred_label,
                "confidence": confidence,
                "inference_latency_ms": latency_ms,
            })
        except Exception as exc:
            print(f"Skipping unreadable image {image_path}: {exc}")
        if i % 100 == 0 or i == len(selected):
            print(f"Classification: {i}/{len(selected)}")

    classifier_wall_s = time.perf_counter() - class_start
    if not y_true:
        raise SystemExit("Could not evaluate any sampled images.")

    labels = list(class_names)
    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    classifier_latency_arr = np.array(classifier_latencies)

    with (run_dir / "image_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\predicted"] + labels)
        for label, row in zip(labels, cm.tolist()):
            writer.writerow([label] + row)

    # YOLO timing and detections on the sampled images.
    print("\nLoading YOLO detector...")
    detector = YOLO(args.yolo_model)
    detection_images = selected[:min(args.max_detection_images, len(selected))]
    yolo_latencies = []
    boxes_per_image = []
    detector_labels = Counter()
    detector_confidences = []
    yolo_errors = 0
    yolo_start = time.perf_counter()

    for i, (image_path, _) in enumerate(detection_images, start=1):
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
            yolo_latencies.append((time.perf_counter() - t0) * 1000)
            boxes = result.boxes
            n_boxes = 0 if boxes is None else len(boxes)
            boxes_per_image.append(n_boxes)
            if boxes is not None:
                for box in boxes:
                    detector_labels[result.names[int(box.cls[0].item())]] += 1
                    detector_confidences.append(float(box.conf[0].item()))
        except Exception as exc:
            yolo_errors += 1
            print(f"YOLO error for {image_path.name}: {exc}")
        if i % 50 == 0 or i == len(detection_images):
            print(f"YOLO: {i}/{len(detection_images)}")

    yolo_wall_s = time.perf_counter() - yolo_start
    yolo_arr = np.array(yolo_latencies)
    yolo_mean = float(yolo_arr.mean()) if len(yolo_arr) else 0.0
    yolo_median = float(np.median(yolo_arr)) if len(yolo_arr) else 0.0
    yolo_p95 = float(np.percentile(yolo_arr, 95)) if len(yolo_arr) else 0.0
    cls_mean = float(classifier_latency_arr.mean())
    cls_median = float(np.median(classifier_latency_arr))
    cls_p95 = float(np.percentile(classifier_latency_arr, 95))

    summary = {
        "evaluation_id": run_dir.name,
        "created_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_directory": str(eval_dir),
        "model_checkpoint": str(model_path),
        "device": str(device),
        "sampling": {
            "available_labeled_images": len(candidates),
            "requested_sample_size": args.sample_size,
            "sampled_images": len(selected),
            "successfully_classified_images": len(y_true),
            "random_seed": args.seed,
        },
        "classification": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "per_class_report": report,
            "classes": labels,
            "confusion_matrix": cm.tolist(),
            "mean_confidence": float(np.mean(confidence_scores)),
        },
        "classifier_performance": {
            "mean_latency_ms": cls_mean,
            "median_latency_ms": cls_median,
            "p95_latency_ms": cls_p95,
            "model_throughput_images_per_second": 1000.0 / cls_mean if cls_mean else None,
            "end_to_end_classification_wall_seconds": classifier_wall_s,
            "end_to_end_throughput_images_per_second": len(y_true) / classifier_wall_s if classifier_wall_s else None,
        },
        "yolo": {
            "images_attempted": len(detection_images),
            "images_successfully_processed": len(yolo_latencies),
            "errors": yolo_errors,
            "mean_latency_ms": yolo_mean if len(yolo_arr) else None,
            "median_latency_ms": yolo_median if len(yolo_arr) else None,
            "p95_latency_ms": yolo_p95 if len(yolo_arr) else None,
            "throughput_images_per_second": 1000.0 / yolo_mean if yolo_mean else None,
            "wall_seconds": yolo_wall_s,
            "images_with_detections": int(sum(n > 0 for n in boxes_per_image)),
            "images_with_zero_detections": int(sum(n == 0 for n in boxes_per_image)),
            "total_predicted_boxes": int(sum(boxes_per_image)),
            "mean_detections_per_image": float(np.mean(boxes_per_image)) if boxes_per_image else None,
            "mean_detection_confidence": float(np.mean(detector_confidences)) if detector_confidences else None,
            "detector_label_counts": dict(detector_labels),
            "note": "YOLO is a general-object detector. Detection precision, recall and mAP need ground-truth boxes.",
        },
        "system": {
            "cpu_logical_cores": __import__("os").cpu_count(),
            "process_rss_mb_at_end": (
                psutil.Process().memory_info().rss / (1024 ** 2) if psutil else None
            ),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "interpretation_warning": (
            "If this sample comes from training/validation data, classification metrics "
            "are not unbiased held-out test metrics. Use a separate test split for final reporting."
        ),
    }
    (run_dir / "report.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "classification_report.txt").write_text(
        "WASTE CLASSIFICATION EVALUATION REPORT\n"
        f"Evaluation ID: {run_dir.name}\n"
        f"Timestamp: {summary['created_at_local']}\n"
        f"Device: {device}\n"
        f"Sampled images: {len(selected)}; classified successfully: {len(y_true)}\n\n"
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
        f"Macro precision: {macro_p:.4f}\nMacro recall: {macro_r:.4f}\nMacro F1: {macro_f1:.4f}\n"
        f"Weighted precision: {weighted_p:.4f}\nWeighted recall: {weighted_r:.4f}\nWeighted F1: {weighted_f1:.4f}\n\n"
        f"CLASSIFICATION DETAILS\n{classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=0)}\n"
        f"CONFUSION MATRIX (rows=true, columns=predicted; order={labels})\n{cm}\n\n"
        f"CLASSIFIER LATENCY: mean={cls_mean:.2f} ms, median={cls_median:.2f} ms, P95={cls_p95:.2f} ms\n"
        f"CLASSIFIER THROUGHPUT: {1000 / cls_mean if cls_mean else 0:.2f} model inferences/s\n"
        f"CLASSIFICATION WALL TIME: {classifier_wall_s:.2f} s\n\n"
        f"YOLO images processed: {len(yolo_latencies)}/{len(detection_images)}; errors={yolo_errors}\n"
        f"YOLO latency: mean={yolo_mean:.2f} ms, median={yolo_median:.2f} ms, P95={yolo_p95:.2f} ms\n"
        f"YOLO throughput: {1000 / yolo_mean if yolo_mean else 0:.2f} images/s\n"
        f"Images with detections: {sum(n > 0 for n in boxes_per_image)}\n"
        f"Images with zero detections: {sum(n == 0 for n in boxes_per_image)}\n"
        f"Total predicted boxes: {sum(boxes_per_image)}\n"
        f"Mean detections/image: {np.mean(boxes_per_image) if boxes_per_image else 0:.2f}\n"
        f"Mean YOLO confidence: {np.mean(detector_confidences) * 100 if detector_confidences else 0:.2f}%\n\n"
        "CAVEATS\n"
        "- This evaluation uses images sampled randomly from the selected directory.\n"
        "- If those images were used for training/model selection, these are not held-out test metrics.\n"
        "- YOLO detection precision/recall/mAP require ground-truth bounding-box annotations.\n"
        "- YOLO's pretrained labels are general objects, not custom waste categories.\n"
    )

    save_plots(
        run_dir, labels, y_true, y_pred, classifier_latencies, yolo_latencies,
        boxes_per_image, detector_labels, detector_confidences, summary
    )

    print("\n" + "=" * 72)
    print("EVALUATION FINISHED")
    print("=" * 72)
    print(f"Evaluation ID       : {run_dir.name}")
    print(f"Images sampled      : {len(selected)}")
    print(f"Images classified   : {len(y_true)}")
    print(f"Accuracy            : {accuracy * 100:.2f}%")
    print(f"Macro F1            : {macro_f1:.4f}")
    print(f"Classifier mean     : {cls_mean:.2f} ms/image")
    print(f"YOLO mean           : {yolo_mean:.2f} ms/image")
    print(f"Report folder       : {run_dir}")
    print("Saved plots:")
    for png in sorted(run_dir.glob("*.png")):
        print(f" - {png.name}")

    # Open only after report and every plot has been written to disk.
    open_plots(run_dir)


if __name__ == "__main__":
    main()
