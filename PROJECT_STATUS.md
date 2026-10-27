# Waste Classification Project Status

## Current Baseline

- Model: ResNet-18 using transfer learning and fine-tuning
- Classes: Hazardous, Non-Recyclable, Organic, Recyclable
- Reported held-out test accuracy: 84%
- Reported macro F1-score: 0.76
- Checkpoint: `models/best_finetuned_resnet18.pth`
- Application: `src/app.py`

## Completed

- Dataset cleaning and manual review
- Initial model training
- Fine-tuning and checkpoint saving
- Classification report and confusion matrix
- Streamlit application with image upload and webcam input
- Grad-CAM visualization
- IVP preprocessing visualization
- Basic disposal recommendations
- Confidence-based manual-inspection warning

## Remaining

1. Verify and document the final baseline checkpoint.
2. Collect a separate dataset of non-waste objects.
3. Create a fifth class: `Not_Waste`.
4. Train and evaluate a five-class model.
5. Compare the new model against the baseline.
6. Update the application to handle non-waste images.
7. Test the complete application and finalize documentation.

## Important Notes

- Keep the existing baseline checkpoint unchanged.
- Do not use the test set to select training settings.
- Recalculate metrics whenever the dataset or checkpoint changes.
- Confidence alone cannot reliably detect non-waste images.
