import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resolve dynamic paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def demonstrate_ivp_pipeline(image_path: str):
    """
    Executes the 8-stage Image & Video Processing (IVP) pipeline on a given waste image
    and displays the comparative step-by-step grid for faculty presentation.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        print(f"Error: File not found at '{image_path}'")
        return

    # -------------------------------------------------------------
    # Stage 1: Matrix Representation & Color Space Conversion (CO1)
    # -------------------------------------------------------------
    img_bgr = cv2.imread(str(path_obj))
    if img_bgr is None:
        print(f"Error: OpenCV could not decode image at '{image_path}'")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------
    # Stage 2: Grayscale Intensity Matrix Transform (CO1)
    # -------------------------------------------------------------
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # -------------------------------------------------------------
    # Stage 3: Spatial Filtering via 2D Gaussian Kernel (CO2)
    # Suppresses high-frequency sensor noise and micro-textures
    # -------------------------------------------------------------
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)

    # -------------------------------------------------------------
    # Stage 4: Contrast Limited Adaptive Histogram Equalization (CO2)
    # Normalizes non-uniform lighting over 8x8 contextual tiles
    # -------------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # -------------------------------------------------------------
    # Stage 5: Gradient-Based Edge Detection via Canny Operator (CO3)
    # Applies Sobel derivatives, Non-Maximum Suppression, and Hysteresis
    # -------------------------------------------------------------
    edges = cv2.Canny(enhanced, threshold1=50, threshold2=150)

    # -------------------------------------------------------------
    # Stage 6: Optimal Global Thresholding via Otsu's Criterion (CO4)
    # Maximizes between-class variance between background and object
    # -------------------------------------------------------------
    _, otsu_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # -------------------------------------------------------------
    # Stage 7: Mathematical Morphology - Closing Operation (CO3)
    # Dilation followed by erosion using a 5x5 structuring element
    # Bridges fissures and seals internal reflection pores
    # -------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

    # Invert mask polarity if background was erroneously designated foreground
    if np.mean(morph_mask == 255) > 0.85:
        morph_mask = cv2.bitwise_not(morph_mask)

    # -------------------------------------------------------------
    # Stage 8: Background Suppression & Waste Isolation
    # Element-wise multiplication: I_seg = I * Mask
    # -------------------------------------------------------------
    segmented_object = cv2.bitwise_and(img_resized, img_resized, mask=morph_mask)

    # -------------------------------------------------------------
    # Matplotlib 8-Panel Visualization Grid
    # -------------------------------------------------------------
    titles = [
        "1. Original RGB (CO1)",
        "2. Grayscale (CO1)",
        "3. Gaussian Blur (CO2)",
        "4. CLAHE Enhanced (CO2)",
        "5. Canny Edges (CO3)",
        "6. Otsu Mask (CO4)",
        "7. Morph. Closed (CO3)",
        "8. Isolated Waste Object"
    ]

    stages = [
        img_resized, gray, blurred, enhanced,
        edges, otsu_mask, morph_mask, segmented_object
    ]

    plt.figure(figsize=(16, 8))
    for idx, (title, stage_img) in enumerate(zip(titles, stages)):
        plt.subplot(2, 4, idx + 1)
        if len(stage_img.shape) == 2:
            plt.imshow(stage_img, cmap='gray')
        else:
            plt.imshow(stage_img)
        plt.title(title, fontsize=10, fontweight="bold")
        plt.axis("off")

    class_name = path_obj.parent.name
    plt.suptitle(f"Sample: {path_obj.name} | Category: {class_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def select_image_interactive():
    """
    Provides an interactive terminal menu to select images by category,
    draw randomly, or specify a custom relative file path.
    """
    if not DATA_DIR.exists():
        print(f"Error: Dataset directory does not exist at '{DATA_DIR}'")
        return None

    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not classes:
        print(f"No category subdirectories found in '{DATA_DIR}'")
        return None

    print("\n" + "=" * 50)
    print("      IVP PIPELINE DEMONSTRATION SELECTOR")
    print("=" * 50)
    for idx, cls_name in enumerate(classes):
        print(f"  [{idx + 1}] Select from '{cls_name}'")
    print("  [R] Pick Random Image from Any Category")
    print("  [M] Enter Custom File Path Manually")
    print("=" * 50)

    choice = input("Enter selection: ").strip()

    if choice.lower() == "r":
        all_imgs = (
            list(DATA_DIR.rglob("*.jpg"))
            + list(DATA_DIR.rglob("*.jpeg"))
            + list(DATA_DIR.rglob("*.png"))
        )
        if not all_imgs:
            print("No images found in dataset.")
            return None
        return str(random.choice(all_imgs))

    elif choice.lower() == "m":
        user_input = input("Enter relative or absolute image path: ").strip()
        custom_path = Path(user_input)
        if not custom_path.is_absolute():
            custom_path = PROJECT_ROOT / custom_path
        if custom_path.exists():
            return str(custom_path)
        print(f"File not found: {custom_path}")
        return None

    elif choice.isdigit() and 1 <= int(choice) <= len(classes):
        chosen_class = classes[int(choice) - 1]
        class_folder = DATA_DIR / chosen_class
        cat_imgs = (
            list(class_folder.rglob("*.jpg"))
            + list(class_folder.rglob("*.jpeg"))
            + list(class_folder.rglob("*.png"))
        )
        if not cat_imgs:
            print(f"No images found inside '{chosen_class}'.")
            return None
        return str(random.choice(cat_imgs))

    else:
        print("Invalid choice selected.")
        return None


if __name__ == "__main__":
    selected_file = select_image_interactive()
    if selected_file:
        print(f"\nProcessing target image: {selected_file}")
        demonstrate_ivp_pipeline(selected_file)