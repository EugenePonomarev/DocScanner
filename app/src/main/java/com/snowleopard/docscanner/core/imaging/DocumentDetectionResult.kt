package com.snowleopard.docscanner.core.imaging

import android.graphics.Bitmap

data class DocumentDetectionResult(
    val polygon: List<Pair<Float, Float>> = emptyList(),
    val cropped: Bitmap? = null,
    val thumbnail: Bitmap? = null,
    val qualityAccepted: Boolean = false,
    val confidence: Float = 0f,
)