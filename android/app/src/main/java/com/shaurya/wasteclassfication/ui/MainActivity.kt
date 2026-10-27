package com.shaurya.wasteclassfication.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.shaurya.wasteclassfication.data.ml.BitmapUtils
import com.shaurya.wasteclassfication.ui.theme.WasteClassifierTheme
import java.io.File

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()
    private var tempCameraUri: Uri? = null

    // Camera launcher
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            tempCameraUri?.let { uri ->
                val bitmap = BitmapUtils.decodeUriWithExif(this, uri)
                if (bitmap != null) {
                    viewModel.analyzeBitmap(bitmap)
                } else {
                    Toast.makeText(this, "Failed to load camera photo", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // Gallery launcher
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            val bitmap = BitmapUtils.decodeUriWithExif(this, uri)
            if (bitmap != null) {
                viewModel.analyzeBitmap(bitmap)
            } else {
                Toast.makeText(this, "Failed to load selected image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Permission launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            launchCameraInternal()
        } else {
            Toast.makeText(this, "Camera permission is required to take photos", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            WasteClassifierTheme {
                MainScreen(
                    viewModel = viewModel,
                    onLaunchCamera = { checkCameraPermissionAndLaunch() },
                    onLaunchGallery = { pickImageLauncher.launch("image/*") }
                )
            }
        }
    }

    private fun checkCameraPermissionAndLaunch() {
        val permission = Manifest.permission.CAMERA
        if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            launchCameraInternal()
        } else {
            requestPermissionLauncher.launch(permission)
        }
    }

    private fun launchCameraInternal() {
        val photoFile = File(cacheDir, "photos").apply { mkdirs() }.let {
            File(it, "waste_scan_${System.currentTimeMillis()}.jpg")
        }
        val uri = FileProvider.getUriForFile(
            this,
            "${applicationContext.packageName}.fileprovider",
            photoFile
        )
        tempCameraUri = uri
        takePictureLauncher.launch(uri)
    }
}
