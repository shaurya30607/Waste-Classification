package com.shaurya.wasteclassfication.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.shaurya.wasteclassfication.data.model.WasteDetection
import com.shaurya.wasteclassfication.ui.components.BoundingBoxOverlay
import com.shaurya.wasteclassfication.ui.components.WasteResultCard
import com.shaurya.wasteclassfication.ui.theme.ForestPrimary
import com.shaurya.wasteclassfication.ui.theme.HazardousRed
import com.shaurya.wasteclassfication.ui.theme.NonRecyclableBrown
import com.shaurya.wasteclassfication.ui.theme.OrganicGreen
import com.shaurya.wasteclassfication.ui.theme.RecyclableBlue

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel,
    onLaunchCamera: () -> Unit,
    onLaunchGallery: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val isHighSensitivity by viewModel.isHighSensitivity.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "♻️ Waste Classifier",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                actions = {
                    if (uiState is MainUiState.Success) {
                        IconButton(onClick = { viewModel.clear() }) {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = "New Scan"
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            when (val state = uiState) {
                is MainUiState.Idle -> IdleView(
                    isHighSensitivity = isHighSensitivity,
                    onToggleSensitivity = { viewModel.setHighSensitivity(it) },
                    onLaunchCamera = onLaunchCamera,
                    onLaunchGallery = onLaunchGallery
                )
                is MainUiState.Analyzing -> AnalyzingView()
                is MainUiState.Success -> SuccessView(
                    bitmap = state.bitmap,
                    detections = state.detections,
                    selectedId = state.selectedId,
                    isHighSensitivity = state.isHighSensitivity,
                    onToggleSensitivity = { viewModel.setHighSensitivity(it) },
                    onAnalyzeFullFrame = { viewModel.analyzeFullFrame() },
                    onSelectDetection = { viewModel.selectDetection(it) },
                    onNewScan = { viewModel.clear() }
                )
                is MainUiState.Error -> ErrorView(
                    message = state.message,
                    onRetry = { viewModel.clear() }
                )
            }
        }
    }
}

@Composable
private fun IdleView(
    isHighSensitivity: Boolean,
    onToggleSensitivity: (Boolean) -> Unit,
    onLaunchCamera: () -> Unit,
    onLaunchGallery: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(96.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)),
            contentAlignment = Alignment.Center
        ) {
            Text(text = "📸", fontSize = 48.sp)
        }

        Spacer(modifier = Modifier.height(20.dp))

        Text(
            text = "On-Device Waste Detection",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurface
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Take a photo or choose from gallery to detect objects and receive disposal recommendations completely offline.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(20.dp))

        FilterChip(
            selected = isHighSensitivity,
            onClick = { onToggleSensitivity(!isHighSensitivity) },
            label = {
                Text(if (isHighSensitivity) "⚡ High Sensitivity (0.12)" else "Normal Mode (0.25)")
            },
            leadingIcon = {
                Icon(
                    imageVector = Icons.Default.Tune,
                    contentDescription = "Sensitivity",
                    modifier = Modifier.size(16.dp)
                )
            }
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = onLaunchCamera,
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = ForestPrimary)
        ) {
            Icon(Icons.Default.CameraAlt, contentDescription = null)
            Spacer(modifier = Modifier.width(10.dp))
            Text("Take Photo", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }

        Spacer(modifier = Modifier.height(14.dp))

        OutlinedButton(
            onClick = onLaunchGallery,
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp),
            shape = RoundedCornerShape(14.dp)
        ) {
            Icon(Icons.Default.PhotoLibrary, contentDescription = null)
            Spacer(modifier = Modifier.width(10.dp))
            Text("Select from Gallery", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }

        Spacer(modifier = Modifier.height(36.dp))

        // Categories indicator row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            CategoryDot("Organic", OrganicGreen)
            CategoryDot("Recyclable", RecyclableBlue)
            CategoryDot("Hazardous", HazardousRed)
            CategoryDot("Non-Recyclable", NonRecyclableBrown)
        }
    }
}

@Composable
private fun CategoryDot(label: String, color: androidx.compose.ui.graphics.Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier
                .size(14.dp)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = label, style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun AnalyzingView() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(60.dp),
            color = ForestPrimary,
            strokeWidth = 5.dp
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "Analyzing on-device...",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Running YOLOv8 detector & ResNet-18 classifier offline",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun SuccessView(
    bitmap: Bitmap,
    detections: List<WasteDetection>,
    selectedId: Int?,
    isHighSensitivity: Boolean,
    onToggleSensitivity: (Boolean) -> Unit,
    onAnalyzeFullFrame: () -> Unit,
    onSelectDetection: (Int) -> Unit,
    onNewScan: () -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Image preview with overlay
        item {
            Card(
                shape = RoundedCornerShape(16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(bitmap.width.toFloat() / bitmap.height.toFloat())
                ) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Analyzed photo",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.FillBounds
                    )

                    BoundingBoxOverlay(
                        detections = detections,
                        selectedId = selectedId,
                        onSelectDetection = onSelectDetection
                    )
                }
            }
        }

        // Header summary & sensitivity toggle
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Detected Items (${detections.size})",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = if (detections.isNotEmpty()) "Tap box or card to inspect" else "No distinct items isolated",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                FilterChip(
                    selected = isHighSensitivity,
                    onClick = { onToggleSensitivity(!isHighSensitivity) },
                    label = {
                        Text(if (isHighSensitivity) "⚡ High (0.12)" else "Normal (0.25)")
                    },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.Tune,
                            contentDescription = "Sensitivity",
                            modifier = Modifier.size(16.dp)
                        )
                    }
                )
            }
        }

        // Empty state card with actionable alternatives
        if (detections.isEmpty()) {
            item {
                Card(
                    shape = RoundedCornerShape(14.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "ℹ️ No items isolated individually",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "In complex scenes with chargers or set-top boxes, switch to High Sensitivity or scan the whole photo.",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.height(14.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            if (!isHighSensitivity) {
                                Button(
                                    onClick = { onToggleSensitivity(true) },
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Text("⚡ High Sensitivity", fontSize = 13.sp)
                                }
                            }
                            OutlinedButton(
                                onClick = onAnalyzeFullFrame,
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text("🖼️ Scan Whole Scene", fontSize = 13.sp)
                            }
                        }
                    }
                }
            }
        }

        // List of detected waste items
        items(detections, key = { it.id }) { detection ->
            WasteResultCard(
                detection = detection,
                isSelected = (detection.id == selectedId),
                onClick = { onSelectDetection(detection.id) }
            )
        }

        // Action button for new scan
        item {
            Spacer(modifier = Modifier.height(8.dp))
            Button(
                onClick = onNewScan,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.CameraAlt, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Scan Another Image")
            }
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ErrorView(
    message: String,
    onRetry: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "⚠️", fontSize = 48.sp)
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Analysis Error",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onRetry) {
            Text("Try Again")
        }
    }
}
