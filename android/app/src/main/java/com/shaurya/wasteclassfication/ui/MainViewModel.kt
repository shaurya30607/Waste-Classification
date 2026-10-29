package com.shaurya.wasteclassfication.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.shaurya.wasteclassfication.data.model.WasteDetection
import com.shaurya.wasteclassfication.domain.WasteRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface MainUiState {
    object Idle : MainUiState
    object Analyzing : MainUiState
    data class Success(
        val bitmap: Bitmap,
        val detections: List<WasteDetection>,
        val selectedId: Int? = null,
        val isHighSensitivity: Boolean = false
    ) : MainUiState
    data class Error(val message: String) : MainUiState
}

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = WasteRepository(application.applicationContext)

    private val _uiState = MutableStateFlow<MainUiState>(MainUiState.Idle)
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private val _isHighSensitivity = MutableStateFlow(false)
    val isHighSensitivity: StateFlow<Boolean> = _isHighSensitivity.asStateFlow()

    fun analyzeBitmap(
        bitmap: Bitmap,
        highSensitivity: Boolean = _isHighSensitivity.value,
        allowFullFrame: Boolean = false
    ) {
        viewModelScope.launch {
            _uiState.value = MainUiState.Analyzing
            try {
                val results = repository.analyzeImage(
                    bitmap = bitmap,
                    highSensitivity = highSensitivity,
                    allowFullFrameFallback = allowFullFrame
                )
                _uiState.value = MainUiState.Success(
                    bitmap = bitmap,
                    detections = results,
                    selectedId = results.firstOrNull()?.id,
                    isHighSensitivity = highSensitivity
                )
            } catch (e: Exception) {
                e.printStackTrace()
                _uiState.value = MainUiState.Error(e.localizedMessage ?: "Failed to analyze image.")
            }
        }
    }

    fun setHighSensitivity(enabled: Boolean) {
        _isHighSensitivity.value = enabled
        val current = _uiState.value
        if (current is MainUiState.Success) {
            analyzeBitmap(current.bitmap, highSensitivity = enabled, allowFullFrame = false)
        }
    }

    fun analyzeFullFrame() {
        val current = _uiState.value
        if (current is MainUiState.Success) {
            analyzeBitmap(current.bitmap, highSensitivity = _isHighSensitivity.value, allowFullFrame = true)
        }
    }

    fun selectDetection(id: Int) {
        val current = _uiState.value
        if (current is MainUiState.Success) {
            _uiState.value = current.copy(selectedId = id)
        }
    }

    fun clear() {
        _uiState.value = MainUiState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        repository.release()
    }
}
