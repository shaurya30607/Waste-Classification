import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def demonstrate_ivp_pipeline(image_path: str):
    # Step 1: Matrix Representation & Color Conversion (CO1)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error loading image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Step 2: Grayscale Transform (CO1)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Step 3: Spatial Gaussian Filtering for Noise Reduction (CO2)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Step 4: Histogram Equalization via CLAHE (CO2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Step 5: Canny Edge Detection (CO3)
    edges = cv2.Canny(enhanced, 50, 150)

    # Step 6: Otsu Automatic Segmentation (CO4)
    _, otsu_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 7: Morphological Closing to Fill Holes (CO3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

    # Step 8: Foreground Waste Extraction
    segmented = cv2.bitwise_and(img_resized, img_resized, mask=morph_mask)

    # 8-Panel Pipeline Plot
    titles = [
        "1. Original RGB (CO1)",
        "2. Grayscale (CO1)",
        "3. Gaussian Blur (CO2)",
        "4. CLAHE Enhanced (CO2)",
        "5. Canny Edges (CO3)",
        "6. Otsu Mask (CO4)",
        "7. Morph. Closed (CO3)",
        "8. Segmented Foreground"
    ]
    images = [img_resized, gray, blurred, enhanced, edges, otsu_mask, morph_mask, segmented]

    plt.figure(figsize=(15, 7))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(2, 4, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title, fontsize=10, fontweight="bold")
        plt.axis("off")

    plt.suptitle(f"Sample: {Path(image_path).parent.name} / {Path(image_path).name}", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Collect all image files recursively across all class directories
    all_images = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png")) + list(DATA_DIR.rglob("*.jpeg"))

    if all_images:
        selected_image = str(random.choice(all_images))
        print(f"Randomly selected sample: {selected_image}")
        demonstrate_ivp_pipeline(selected_image)
    else:
        print("No image files found in data/raw.")