package com.shaurya.wasteclassfication.domain

import android.content.Context
import android.graphics.Bitmap
import com.shaurya.wasteclassfication.data.ml.BitmapUtils
import com.shaurya.wasteclassfication.data.ml.ResNetClassifier
import com.shaurya.wasteclassfication.data.ml.YoloV8Detector
import com.shaurya.wasteclassfication.data.model.BoundingBox
import com.shaurya.wasteclassfication.data.model.DisposalGuidanceProvider
import com.shaurya.wasteclassfication.data.model.WasteDetection
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class WasteRepository(context: Context) {

    private val detector = YoloV8Detector(context)
    private val classifier = ResNetClassifier(context)

    companion object {
        const val MIN_GUARDRAIL_CONFIDENCE = 65.0f

        val E_WASTE_LABELS = setOf(
            "remote", "cell phone", "laptop", "mouse", "keyboard", "tv",
            "microwave", "oven", "toaster", "hair drier", "clock"
        )

        val ORGANIC_FOOD_LABELS = setOf(
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake"
        )
    }

    suspend fun analyzeImage(
        bitmap: Bitmap,
        highSensitivity: Boolean = false,
        allowFullFrameFallback: Boolean = false
    ): List<WasteDetection> = withContext(Dispatchers.Default) {
        val threshold = if (highSensitivity) 0.12f else 0.25f
        val yoloResult = detector.detect(bitmap, confidenceThreshold = threshold, autoRetry = true)
        val wasteCandidates = yoloResult.wasteCandidates

        val results = mutableListOf<WasteDetection>()

        if (wasteCandidates.isNotEmpty()) {
            wasteCandidates.forEachIndexed { index, detection ->
                val crop = BitmapUtils.cropBoundingBox(bitmap, detection.box)
                val classification = classifier.classify(crop)

                val (finalCategory, finalConfidence, guidanceKey) = when {
                    E_WASTE_LABELS.contains(detection.label.lowercase()) -> {
                        val fusedConf = (detection.score * 100f).coerceIn(85f, 98f)
                        Triple("Hazardous", fusedConf, "Hazardous (E-Waste)")
                    }
                    ORGANIC_FOOD_LABELS.contains(detection.label.lowercase()) -> {
                        val fusedConf = (detection.score * 100f).coerceIn(85f, 98f)
                        Triple("Organic", fusedConf, "Organic")
                    }
                    else -> {
                        Triple(classification.category, classification.confidencePercent, classification.category)
                    }
                }

                val isLow = finalConfidence < MIN_GUARDRAIL_CONFIDENCE
                val guidance = DisposalGuidanceProvider.getGuidance(guidanceKey)

                results.add(
                    WasteDetection(
                        id = index + 1,
                        boundingBox = detection.box,
                        category = finalCategory,
                        confidence = finalConfidence,
                        yoloLabel = detection.label,
                        isLowConfidence = isLow,
                        disposalGuidance = guidance
                    )
                )
            }
        } else if (allowFullFrameFallback && !yoloResult.hasPerson) {
            // Only perform full frame classification if explicitly requested by the user
            val fullBox = BoundingBox(0.05f, 0.05f, 0.95f, 0.95f)
            val classification = classifier.classify(bitmap)
            val isLow = classification.confidencePercent < MIN_GUARDRAIL_CONFIDENCE
            val guidance = DisposalGuidanceProvider.getGuidance(classification.category)

            results.add(
                WasteDetection(
                    id = 1,
                    boundingBox = fullBox,
                    category = classification.category,
                    confidence = classification.confidencePercent,
                    yoloLabel = "Single Item (Full Frame)",
                    isLowConfidence = isLow,
                    disposalGuidance = guidance
                )
            )
        }

        results
    }

    fun isReady(): Boolean = detector.isReady() && classifier.isReady()

    fun release() {
        detector.close()
        classifier.close()
    }
}
