import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def demonstrate_ivp_pipeline(image_path: str):
    # Step 1: Matrix Representation & Resolution Standardisation (CO1)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error loading image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Step 2: Grayscale Transform (CO1)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Step 3: Spatial Filtering / Gaussian Blur (CO2)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Step 4: Histogram Equalization via CLAHE (CO2)        (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Step 5: Canny Edge Detection (CO3)
    edges = cv2.Canny(enhanced, 50, 150)

    # Step 6: Otsu Segmentation (CO4)
    _, otsu_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 7: Morphological Closing (CO3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    morph_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

    # Step 8: Foreground Waste Masking

    segmented = cv2.bitwise_and(img_resized, img_resized, mask=morph_mask)

    # 8-Panel Plot Display
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

    plt.suptitle(f"File: {Path(image_path).name} | Class: {Path(image_path).parent.name}", fontsize=12)
    plt.tight_layout()
    plt.show()


def select_image_interactive():
    classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not classes:
        print(f"No classes found in {DATA_DIR}")
        return None

    print("\n--- Available Waste Classes ---")
    for idx, cls_name in enumerate(classes):
        print(f"[{idx + 1}] {cls_name}")
    print("[R] Random Image from Entire Dataset")
    print("[M] Enter Custom Relative Path Manually")

    choice = input("\nSelect an option: ").strip()

    if choice.lower() == "r":
        all_images = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png")) + list(DATA_DIR.rglob("*.jpeg"))
        return str(random.choice(all_images)) if all_images else None

    elif choice.lower() == "m":
        custom_path = input("Enter path (e.g. data/raw/Organic/sample.jpg): ").strip()
        full_path = PROJECT_ROOT / custom_path
        return str(full_path) if full_path.exists() else None

    elif choice.isdigit() and 1 <= int(choice) <= len(classes):
        selected_class = classes[int(choice) - 1]
        class_folder = DATA_DIR / selected_class
        images = list(class_folder.rglob("*.jpg")) + list(class_folder.rglob("*.png")) + list(
            class_folder.rglob("*.jpeg"))

        if not images:
            print(f"No images found inside {selected_class}")
            return None

        # Picks a random sample from the chosen category
        return str(random.choice(images))

    else:
        print("Invalid selection.")
        return None


if __name__ == "__main__":
    image_to_process = select_image_interactive()
    if image_to_process:
        print(f"\nProcessing: {image_to_process}")
        demonstrate_ivp_pipeline(image_to_process)