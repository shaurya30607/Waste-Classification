import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Anchor path relative to this script's parent folder (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_dataset():
    if not DATA_DIR.exists():
        print(f"Directory not found: {DATA_DIR}")
        print("Please ensure your dataset is placed inside the 'data/raw' folder.")
        return

    classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not classes:
        print(f"No class folders found inside: {DATA_DIR}")
        print("Expected folders like: 'Hazardous', 'Organic', 'Recyclable', etc.")
        return

    print(f"Found {len(classes)} classes: {classes}")

    class_counts = {}
    sample_images = {}

    for cls in classes:
        cls_folder = DATA_DIR / cls
        images = [f for f in cls_folder.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        class_counts[cls] = len(images)
        if images:
            sample_images[cls] = str(images[0])

    print("\n--- Image Count Per Class ---")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images")

    # Plot sample images
    # if sample_images:
    #     fig, axes = plt.subplots(1, len(sample_images), figsize=(4 * len(sample_images), 4))
    #     if len(sample_images) == 1:
    #         axes = [axes]
    #
    #     for ax, (cls, img_path) in zip(axes, sample_images.items()):
    #         img = cv2.imread(img_path)
    #         if img is not None:
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             ax.imshow(img)
    #             ax.set_title(f"{cls}\n({class_counts[cls]} imgs)")
    #         ax.axis("off")
    #
    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    inspect_dataset()