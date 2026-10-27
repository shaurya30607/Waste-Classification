package com.shaurya.wasteclassfication.ui.components

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import com.shaurya.wasteclassfication.data.model.WasteDetection
import com.shaurya.wasteclassfication.ui.theme.ConfidentGreen
import com.shaurya.wasteclassfication.ui.theme.HazardousRed
import com.shaurya.wasteclassfication.ui.theme.NonRecyclableBrown
import com.shaurya.wasteclassfication.ui.theme.OrganicGreen
import com.shaurya.wasteclassfication.ui.theme.RecyclableBlue
import com.shaurya.wasteclassfication.ui.theme.WarningAmber

@Composable
fun BoundingBoxOverlay(
    detections: List<WasteDetection>,
    selectedId: Int?,
    onSelectDetection: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Canvas(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(detections) {
                detectTapGestures { offset ->
                    val clicked = detections.lastOrNull { d ->
                        val l = d.boundingBox.left * size.width
                        val t = d.boundingBox.top * size.height
                        val r = d.boundingBox.right * size.width
                        val b = d.boundingBox.bottom * size.height
                        offset.x in l..r && offset.y in t..b
                    }
                    if (clicked != null) {
                        onSelectDetection(clicked.id)
                    }
                }
            }
    ) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        val textPaint = Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 34f
            isAntiAlias = true
            isFakeBoldText = true
        }

        val bgPaint = Paint().apply {
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        detections.forEach { detection ->
            val isSelected = (detection.id == selectedId)
            val strokeWidth = if (isSelected) 8f else 5f

            val boxColor = when {
                detection.isLowConfidence -> WarningAmber
                detection.category == "Hazardous" -> HazardousRed
                detection.category == "Recyclable" -> RecyclableBlue
                detection.category == "Organic" -> OrganicGreen
                detection.category == "Non-Recyclable" -> NonRecyclableBrown
                else -> ConfidentGreen
            }

            val left = detection.boundingBox.left * canvasWidth
            val top = detection.boundingBox.top * canvasHeight
            val width = detection.boundingBox.width * canvasWidth
            val height = detection.boundingBox.height * canvasHeight

            // Draw bounding box rectangle
            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = Size(width, height),
                style = Stroke(width = strokeWidth)
            )

            // Draw header label tag
            val labelText = "#${detection.id} ${detection.category} (${detection.confidence.toInt()}%)"
            val textWidth = textPaint.measureText(labelText)
            val tagHeight = 44f
            val tagWidth = textWidth + 24f

            bgPaint.color = boxColor.toArgb()
            drawContext.canvas.nativeCanvas.drawRect(
                left,
                (top - tagHeight).coerceAtLeast(0f),
                left + tagWidth,
                (top).coerceAtLeast(tagHeight),
                bgPaint
            )

            drawContext.canvas.nativeCanvas.drawText(
                labelText,
                left + 12f,
                (top - 12f).coerceAtLeast(32f),
                textPaint
            )
        }
    }
}
