package com.snowleopard.docscanner

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class ScanViewModel : ViewModel() {
    private val _livePreviewPolygon = MutableStateFlow<List<Pair<Float, Float>>>(emptyList())
    val livePreviewPolygon = _livePreviewPolygon.asStateFlow()

    private val _liveThumbnail = MutableStateFlow<Bitmap?>(null)
    val liveThumbnail = _liveThumbnail.asStateFlow()

    private val _finalDoc = MutableStateFlow<Bitmap?>(null)
    val finalDoc = _finalDoc.asStateFlow()

    private val _frameSize = MutableStateFlow<Pair<Int, Int>?>(null)
    val frameSize = _frameSize.asStateFlow()

    // --- DEBUG ---
    private val _debugFrame = MutableStateFlow<Bitmap?>(null)
    val debugFrame = _debugFrame.asStateFlow()

    private val _status = MutableStateFlow("")
    val status = _status.asStateFlow()

    fun setStatus(s: String) { _status.value = s }
    fun setDebugFrame(bmp: Bitmap?) { _debugFrame.value = bmp }

    // -------------

    fun updateOverlay(polygon: List<Pair<Float, Float>>, thumb: Bitmap?) {
        _livePreviewPolygon.value = polygon
        if (thumb != null) _liveThumbnail.value = thumb
    }

    fun setFinalDoc(bitmap: Bitmap) {
        _finalDoc.value = bitmap
    }

    fun setFrameSize(w: Int, h: Int) {
        _frameSize.value = w to h
    }

    fun clear() {
        _livePreviewPolygon.value = emptyList()
        _liveThumbnail.value = null
        _finalDoc.value = null
        _frameSize.value = null
        _debugFrame.value = null
        _status.value = ""
    }
}