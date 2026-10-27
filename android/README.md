# Waste Classifier - Android App ♻️

A modern, native Android application built with **Kotlin** and **Jetpack Compose** that performs **100% offline, on-device multi-object detection and waste classification**.

---

## Key Features

1. **On-Device Offline Inference**:
   * Uses **ONNX Runtime Mobile** (`com.microsoft.onnxruntime:onnxruntime-android`).
   * No network connection or server backend required.
2. **Hybrid Detection & Classification Pipeline**:
   * **Stage 1 (Object Proposals)**: Pretrained **YOLOv8n** detects objects and filters non-waste classes (e.g. people, animals, vehicles).
   * **Stage 2 (Waste Classification)**: Fine-tuned **ResNet-18** classifies each cropped item into one of the 4 project classes:
     * `Organic`
     * `Recyclable`
     * `Hazardous`
     * `Non-Recyclable`
   * **Single-Item Fallback**: If no discrete bounding boxes are found, the model classifies the full frame seamlessly.
3. **Interactive Visual Feedback**:
   * Color-coded bounding box overlays directly on top of the captured photo.
   * Confidence progress bars and category badges.
4. **Guardrails & Disposal Guidance**:
   * Predictions with confidence below **65%** are flagged with a prominent manual check alert.
   * Expandable step-by-step disposal instructions and environmental warnings tailored to each category.
5. **Modern Android Architecture**:
   * Clean Architecture (Data, Domain, Presentation).
   * MVVM with Coroutines & StateFlow.
   * Jetpack Compose + Material 3.

---

## Directory Structure

```text
android/
├── build.gradle.kts                      # Root Gradle configuration
├── settings.gradle.kts                   # Project module inclusion
├── gradle.properties                     # JVM & AndroidX memory flags
├── gradlew                               # Unix Gradle launcher
├── app/
│   ├── build.gradle.kts                  # App module dependencies & packaging
│   └── src/main/
│       ├── AndroidManifest.xml           # Permissions & FileProvider
│       ├── assets/                       # On-device model storage
│       │   ├── yolov8n.onnx              # YOLOv8n object detection model
│       │   ├── resnet18.onnx             # ResNet-18 waste classification model
│       │   └── waste_labels.txt          # Class labels (Hazardous, etc.)
│       ├── res/
│       │   ├── xml/file_paths.xml        # FileProvider camera photo paths
│       │   └── values/
│       │       ├── strings.xml
│       │       └── themes.xml
│       └── java/com/shaurya/wasteclassfication/
│           ├── WasteClassifierApp.kt     # Application class
│           ├── data/
│           │   ├── model/
│           │   │   ├── BoundingBox.kt     # Normalized bounding box math
│           │   │   ├── WasteDetection.kt  # Unified detection result model
│           │   │   └── DisposalGuidance.kt# Actionable disposal advice
│           │   └── ml/
│           │       ├── BitmapUtils.kt     # EXIF rotation, crop & NCHW FloatBuffer
│           │       ├── YoloV8Detector.kt  # YOLOv8 ONNX session + NMS
│           │       └── ResNetClassifier.kt# ResNet-18 ONNX crop classifier
│           ├── domain/
│           │   └── WasteRepository.kt     # Pipeline coordinator & fallback
│           └── ui/
│               ├── MainActivity.kt        # Camera, Gallery & permission logic
│               ├── MainViewModel.kt       # Async state management (StateFlow)
│               ├── MainScreen.kt          # Compose UI (Idle, Analyzing, Success)
│               ├── components/
│               │   ├── BoundingBoxOverlay.kt # Canvas bounding box renderer
│               │   └── WasteResultCard.kt    # Card with guidance & warning
│               └── theme/
│                   ├── Color.kt           # Eco-friendly Material 3 palette
│                   ├── Theme.kt           # Dynamic theme wrapper
│                   └── Type.kt            # Typography definitions
```

---

## How to Build & Run

### Step 1: Export Models to Android Assets
From the root of the repository, execute the automated export script:
```bash
python3 scripts/export_to_onnx.py
```
This converts the PyTorch checkpoints into:
* `android/app/src/main/assets/resnet18.onnx`
* `android/app/src/main/assets/yolov8n.onnx`
* `android/app/src/main/assets/waste_labels.txt`

### Step 2: Open in Android Studio
1. Launch **Android Studio**.
2. Select **Open an Existing Project** and choose the `/android` directory.
3. Allow Gradle to sync dependencies.
4. Select a physical device or emulator running **Android 8.0 (API 26)** or higher.
5. Click **Run** (`Shift + F10`).

### Step 3: Build from Command Line (Optional)
```bash
cd android
./gradlew assembleDebug
```
The generated APK will be located at:
`android/app/build/outputs/apk/debug/app-debug.apk`

To install directly to a connected device:
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## Gradle & JVM Compatibility Note

This project is configured with **Gradle 9.3+** and **Android Gradle Plugin (AGP) 9.1+**, which natively support **JVM 25** (including Android Studio's bundled JBR 25).

* If Android Studio displays a prompt asking to **"Use JVM 21"**, either:
  * Select **"Use JVM 21"** to let Android Studio automatically configure the Gradle Daemon to run on JDK 21 LTS, OR
  * Use **Gradle 9.3+** with **JVM 25** (configured by default in `gradle/wrapper/gradle-wrapper.properties`).

