package com.snowleopard.docscanner.feature.viewmodels

import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.scale
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.snowleopard.docscanner.core.data.model.ScannedPage
import com.snowleopard.docscanner.core.data.repository.PagesRepository
import com.snowleopard.docscanner.core.data.store.PagesStore
import com.snowleopard.docscanner.core.imaging.DocumentDetectionResult
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.sqrt
import kotlin.time.Duration.Companion.milliseconds
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "ScanViewModel"

class ScanViewModel(
    private val repo: PagesRepository,
    private val store: PagesStore,
) : ViewModel() {

    // ----- Live overlay -----

    private val _livePreviewPolygon =
        MutableStateFlow<List<Pair<Float, Float>>>(emptyList())
    val livePreviewPolygon = _livePreviewPolygon.asStateFlow()

    private val _liveThumbnail = MutableStateFlow<Bitmap?>(null)
    val liveThumbnail = _liveThumbnail.asStateFlow()

    private val _frameSize = MutableStateFlow<Pair<Int, Int>?>(null)
    val frameSize = _frameSize.asStateFlow()

    private val _debugFrame = MutableStateFlow<Bitmap?>(null)
    val debugFrame = _debugFrame.asStateFlow()

    private val _status = MutableStateFlow("")
    val status = _status.asStateFlow()

    val pages: StateFlow<List<ScannedPage>> = store.pages

    // ----- Session meta -----

    private val _captureInProgress = MutableStateFlow(false)
    val captureInProgress = _captureInProgress.asStateFlow()

    private val _lastCapturedAt = MutableStateFlow(0L)
    val lastCapturedAt = _lastCapturedAt.asStateFlow()

    private val _wavePolygon =
        MutableStateFlow<List<Pair<Float, Float>>>(emptyList())
    val wavePolygon = _wavePolygon.asStateFlow()

    // ----- Capture/session guards -----

    private val captureLock = AtomicBoolean(false)
    private val sessionResetInProgress = AtomicBoolean(false)

    private var captureJob: Job? = null

    // ----- Freeze/duplicate FSM -----

    private val _captureFreeze = MutableStateFlow(false)
    private var frozenAt: Long = 0L
    private var framesWithoutDoc: Int = 0

    private data class CaptureSig(
        val hash: Long,
    )

    private val capturedSignatures = mutableListOf<CaptureSig>()
    private val maxStoredSignatures = 20

    private var lastPolyAreaNorm: Float = 0f
    private var lastPolyCentroid: Pair<Float, Float>? = null

    private val _lockProgress = MutableStateFlow(0f)
    val lockProgress = _lockProgress.asStateFlow()

    private val _confirmAt = MutableStateFlow(0L)
    val confirmAt = _confirmAt.asStateFlow()

    private var prevPoly: List<Pair<Float, Float>>? = null
    private var stableFrames: Int = 0

    private var latestCropped: Bitmap? = null
    private var latestCroppedAt: Long = 0L
    private var latestCroppedPolygon: List<Pair<Float, Float>>? = null

    private var confirmJob: Job? = null

    // ----- Parameters -----

    private val captureCooldownMs = 900L
    private val minFreezeMs = 1400L
    private val requireDocAbsenceFrames = 5

    private val moveEpsNormalized = 0.02f
    private val angleMin = 0.60
    private val aspectMin = 0.35
    private val areaMinNorm = 0.035f

    private val lockFramesRequired = 6

    private val confirmDurationMs = 420L
    private val acceptStalenessMs = 2200L
    private val dupHashHammingThresh = 5

    fun setStatus(value: String) {
        if (!sessionResetInProgress.get()) {
            _status.value = value
        }
    }

    fun setDebugFrame(bitmap: Bitmap?) {
        if (!sessionResetInProgress.get()) {
            _debugFrame.value = bitmap
        }
    }

    fun setFrameSize(width: Int, height: Int) {
        if (!sessionResetInProgress.get()) {
            _frameSize.value = width to height
        }
    }

    fun updateOverlay(
        polygon: List<Pair<Float, Float>>,
        thumb: Bitmap?,
    ) {
        if (sessionResetInProgress.get()) return

        _livePreviewPolygon.value = polygon

        if (thumb != null) {
            _liveThumbnail.value = thumb
        }
    }

    fun onAnalyzerResult(result: DocumentDetectionResult) {
        if (sessionResetInProgress.get()) return

        val validPolygon = if (result.qualityAccepted) {
            result.polygon
        } else {
            emptyList()
        }

        val captureReady =
            result.qualityAccepted && result.cropped != null

        val lockPolygon = if (captureReady) {
            result.polygon
        } else {
            emptyList()
        }

        updateOverlay(
            polygon = validPolygon,
            thumb = result.thumbnail,
        )

        if (captureReady) {
            latestCropped = result.cropped
            latestCroppedPolygon = result.polygon
            latestCroppedAt = System.currentTimeMillis()
        } else {
            latestCropped = null
            latestCroppedPolygon = null
            latestCroppedAt = 0L
        }

        if (!_captureFreeze.value) {
            updateLockProgress(lockPolygon)
            checkConfirmAndCapture()
        } else {
            _lockProgress.value = 0f
            maybeUnfreeze(validPolygon)
        }
    }

    private fun cancelPendingConfirmation() {
        confirmJob?.cancel()
        confirmJob = null
        _confirmAt.value = 0L
    }

    private fun resetLockProgress(
        clearPrevious: Boolean = true,
    ) {
        stableFrames = 0
        _lockProgress.value = 0f

        if (clearPrevious) {
            prevPoly = null
        }

        cancelPendingConfirmation()
    }

    private fun updateLockProgress(
        poly: List<Pair<Float, Float>>,
    ) {
        val frameSize = _frameSize.value ?: run {
            resetLockProgress()
            return
        }

        if (poly.size < 4) {
            resetLockProgress()
            return
        }

        val (width, height) = frameSize

        val diagonal = hypot(
            width.toDouble(),
            height.toDouble(),
        ).toFloat().coerceAtLeast(1f)

        val areaNorm = (
                polygonArea(poly) / (width.toFloat() * height.toFloat())
                ).coerceIn(0f, 1f)

        val centroid = polygonCentroid(poly)
        val angles = rightAngleScore(poly)
        val aspect = aspectA4Score(poly)

        val previousPoly = prevPoly

        val moved = previousPoly?.let {
            distance(
                centroid,
                polygonCentroid(it),
            ) / diagonal
        } ?: Float.POSITIVE_INFINITY

        val geometryOk =
            angles >= angleMin &&
                    aspect >= aspectMin &&
                    areaNorm >= areaMinNorm

        if (!geometryOk) {
            resetLockProgress()
            return
        }

        prevPoly = poly
        lastPolyAreaNorm = areaNorm
        lastPolyCentroid = centroid

        if (previousPoly == null || moved >= moveEpsNormalized) {
            stableFrames = 0
            _lockProgress.value = 0f
            cancelPendingConfirmation()
            return
        }

        stableFrames++

        val step = 1f / lockFramesRequired

        _lockProgress.value = (
                _lockProgress.value + step
                ).coerceAtMost(1f)
    }

    private fun checkConfirmAndCapture() {
        if (
            _lockProgress.value < 1f ||
            _confirmAt.value != 0L ||
            _captureFreeze.value ||
            _captureInProgress.value ||
            captureLock.get() ||
            latestCropped == null
        ) {
            return
        }

        val startedAt = System.currentTimeMillis()
        val cropAtStart = latestCroppedAt

        _confirmAt.value = startedAt

        confirmJob?.cancel()
        confirmJob = viewModelScope.launch {
            try {
                delay(confirmDurationMs.milliseconds)

                val bitmap = latestCropped

                if (bitmap != null) {
                    val stillValid =
                        _lockProgress.value >= 1f &&
                                !_captureFreeze.value &&
                                !_captureInProgress.value &&
                                latestCroppedAt >= cropAtStart &&
                                System.currentTimeMillis() - latestCroppedAt <= acceptStalenessMs

                    if (stillValid) {
                        performCapture(
                            cropped = bitmap,
                            polygonForWave = latestCroppedPolygon
                                ?: prevPoly
                                ?: emptyList(),
                        )
                    }
                }
            } finally {
                if (_confirmAt.value == startedAt) {
                    _confirmAt.value = 0L
                    _lockProgress.value = 0f
                    stableFrames = 0
                    confirmJob = null
                }
            }
        }
    }

    private fun performCapture(
        cropped: Bitmap,
        polygonForWave: List<Pair<Float, Float>>,
    ) {
        if (sessionResetInProgress.get()) return

        val now = System.currentTimeMillis()

        if (_captureInProgress.value) return

        if (!captureLock.compareAndSet(false, true)) {
            return
        }

        if (now - _lastCapturedAt.value < captureCooldownMs) {
            captureLock.set(false)
            return
        }

        _captureInProgress.value = true

        captureJob = viewModelScope.launch {
            try {
                val newHash = withContext(Dispatchers.Default) {
                    averageHash64(cropped)
                }

                val duplicate = isLikelyDuplicate(newHash)

                if (duplicate) {
                    enableFreeze()
                } else {
                    val page = withContext(Dispatchers.Default) {
                        repo.savePage(
                            cropped,
                            preferPortrait = true,
                        )
                    }

                    _wavePolygon.value = polygonForWave
                    _lastCapturedAt.value = now
                    store.add(page)

                    capturedSignatures += CaptureSig(
                        hash = newHash,
                    )

                    if (capturedSignatures.size > maxStoredSignatures) {
                        capturedSignatures.removeAt(0)
                    }

                    enableFreeze()
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (throwable: Throwable) {
                Log.e(TAG, "Capture failed", throwable)

                _captureFreeze.value = false
                framesWithoutDoc = 0
                stableFrames = 0
                _lockProgress.value = 0f
                _status.value =
                    "capture error: ${throwable.message ?: "unknown error"}"
            } finally {
                captureJob = null
                captureLock.set(false)
                _captureInProgress.value = false
            }
        }
    }

    private fun enableFreeze() {
        _captureFreeze.value = true
        frozenAt = System.currentTimeMillis()
        framesWithoutDoc = 0
    }

    private fun maybeUnfreeze(
        currentPoly: List<Pair<Float, Float>>,
    ) {
        if (!_captureFreeze.value) return

        if (System.currentTimeMillis() - frozenAt < minFreezeMs) {
            return
        }

        val frameSize = _frameSize.value
        val hasDoc = currentPoly.size >= 4

        if (!hasDoc) {
            framesWithoutDoc++

            if (framesWithoutDoc >= requireDocAbsenceFrames) {
                disableFreeze()
            }

            return
        } else {
            framesWithoutDoc = 0
        }

        val previousCentroid = lastPolyCentroid

        if (frameSize != null && previousCentroid != null) {
            val (width, height) = frameSize

            val diagonal = hypot(
                width.toDouble(),
                height.toDouble(),
            ).toFloat().coerceAtLeast(1f)

            val currentAreaNorm = (
                    polygonArea(currentPoly) / (width.toFloat() * height.toFloat())
                    ).coerceIn(0f, 1f)

            val currentCentroid = polygonCentroid(currentPoly)
            val shift = distance(
                currentCentroid,
                previousCentroid,
            ) / diagonal

            val areaRelativeChange =
                if (lastPolyAreaNorm > 1e-6f) {
                    abs(currentAreaNorm - lastPolyAreaNorm) /
                            lastPolyAreaNorm
                } else {
                    1f
                }

            if (shift > 0.10f || areaRelativeChange > 0.25f) {
                disableFreeze()
            }
        }
    }

    private fun disableFreeze() {
        _captureFreeze.value = false
        framesWithoutDoc = 0
    }

    fun removeLastPage() {
        val removed = store.removeLast() ?: return

        viewModelScope.launch(Dispatchers.IO) {
            repo.deletePage(removed)
        }
    }

    fun clearSession() {
        if (!sessionResetInProgress.compareAndSet(false, true)) {
            return
        }

        viewModelScope.launch {
            var errorMessage: String? = null

            try {
                confirmJob?.cancelAndJoin()
                captureJob?.cancelAndJoin()

                withContext(Dispatchers.IO) {
                    repo.clearSession()
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (throwable: Throwable) {
                Log.e(TAG, "Clear session failed", throwable)

                errorMessage =
                    "clear session error: ${throwable.message ?: "unknown error"}"
            } finally {
                store.clear()
                resetSessionState()

                if (errorMessage != null) {
                    _status.value = errorMessage
                }

                sessionResetInProgress.set(false)
            }
        }
    }

    private fun resetSessionState() {
        _livePreviewPolygon.value = emptyList()
        _liveThumbnail.value = null
        _frameSize.value = null
        _debugFrame.value = null
        _status.value = ""

        _wavePolygon.value = emptyList()
        _lastCapturedAt.value = 0L
        _captureFreeze.value = false
        _captureInProgress.value = false

        frozenAt = 0L
        framesWithoutDoc = 0

        prevPoly = null
        stableFrames = 0
        lastPolyAreaNorm = 0f
        lastPolyCentroid = null

        _lockProgress.value = 0f
        _confirmAt.value = 0L

        latestCropped = null
        latestCroppedAt = 0L
        latestCroppedPolygon = null

        capturedSignatures.clear()

        confirmJob = null
        captureJob = null
        captureLock.set(false)
    }

    // ----- Utils for duplicate -----

    private fun isLikelyDuplicate(
        newHash: Long,
    ): Boolean {
        return capturedSignatures.any { signature ->
            hammingDistance(
                newHash,
                signature.hash,
            ) <= dupHashHammingThresh
        }
    }

    private fun polygonArea(
        poly: List<Pair<Float, Float>>,
    ): Float {
        var sum = 0f

        for (index in poly.indices) {
            val (x1, y1) = poly[index]
            val (x2, y2) = poly[(index + 1) % poly.size]

            sum += x1 * y2 - x2 * y1
        }

        return abs(sum) * 0.5f
    }

    private fun polygonCentroid(
        poly: List<Pair<Float, Float>>,
    ): Pair<Float, Float> {
        var centerX = 0f
        var centerY = 0f
        var areaSum = 0f

        for (index in poly.indices) {
            val (x1, y1) = poly[index]
            val (x2, y2) = poly[(index + 1) % poly.size]

            val cross = x1 * y2 - x2 * y1

            centerX += (x1 + x2) * cross
            centerY += (y1 + y2) * cross
            areaSum += cross
        }

        val area = areaSum * 0.5f

        if (abs(area) < 1e-6f) {
            return 0f to 0f
        }

        val factor = 1f / (6f * area)

        return (centerX * factor) to (centerY * factor)
    }

    private fun distance(
        first: Pair<Float, Float>,
        second: Pair<Float, Float>,
    ): Float {
        val dx = first.first - second.first
        val dy = first.second - second.second

        return sqrt(dx * dx + dy * dy)
    }

    private fun rightAngleScore(
        poly: List<Pair<Float, Float>>,
    ): Double {
        fun angle(
            ax: Float,
            ay: Float,
            bx: Float,
            by: Float,
            cx: Float,
            cy: Float,
        ): Double {
            val abx = ax - bx
            val aby = ay - by
            val cbx = cx - bx
            val cby = cy - by

            val dot = abx * cbx + aby * cby

            val norm = sqrt(
                (abx * abx + aby * aby).toDouble() *
                        (cby * cby + cbx * cbx).toDouble()
            ).coerceAtLeast(1e-6)

            val cos = (dot / norm).coerceIn(-1.0, 1.0)

            return Math.toDegrees(acos(cos))
        }

        if (poly.size < 4) {
            return 0.0
        }

        val angles = listOf(
            angle(
                poly[3].first,
                poly[3].second,
                poly[0].first,
                poly[0].second,
                poly[1].first,
                poly[1].second,
            ),
            angle(
                poly[0].first,
                poly[0].second,
                poly[1].first,
                poly[1].second,
                poly[2].first,
                poly[2].second,
            ),
            angle(
                poly[1].first,
                poly[1].second,
                poly[2].first,
                poly[2].second,
                poly[3].first,
                poly[3].second,
            ),
            angle(
                poly[2].first,
                poly[2].second,
                poly[3].first,
                poly[3].second,
                poly[0].first,
                poly[0].second,
            ),
        )

        val scores = angles.map {
            1.0 - (abs(it - 90.0) / 20.0)
                .coerceIn(0.0, 1.0)
        }

        return scores.average()
    }

    private fun aspectA4Score(
        poly: List<Pair<Float, Float>>,
    ): Double {
        if (poly.size < 4) {
            return 0.0
        }

        val xs = poly.map { it.first }
        val ys = poly.map { it.second }

        val maxX = xs.maxOrNull() ?: return 0.0
        val minX = xs.minOrNull() ?: return 0.0
        val maxY = ys.maxOrNull() ?: return 0.0
        val minY = ys.minOrNull() ?: return 0.0

        val width = (maxX - minX).coerceAtLeast(1f)
        val height = (maxY - minY).coerceAtLeast(1f)

        val ratio = if (width > height) {
            width / height
        } else {
            height / width
        }

        val target = 1.4142f
        val diff = abs(ratio - target)

        return (1.0 - (diff / 0.8f))
            .coerceIn(0.0, 1.0)
    }

    private fun averageHash64(
        bitmap: Bitmap,
    ): Long {
        val small = bitmap.scale(8, 8)
        val pixels = IntArray(64)

        small.getPixels(
            pixels,
            0,
            8,
            0,
            0,
            8,
            8,
        )

        var sum = 0
        val grayscale = IntArray(64)

        for (index in 0 until 64) {
            val pixel = pixels[index]

            val red = (pixel shr 16) and 0xFF
            val green = (pixel shr 8) and 0xFF
            val blue = pixel and 0xFF

            val value = (red * 30 + green * 59 + blue * 11) / 100

            grayscale[index] = value
            sum += value
        }

        val average = sum / 64
        var hash = 0L

        for (index in 0 until 64) {
            if (grayscale[index] >= average) {
                hash = hash or (1L shl index)
            }
        }

        small.recycle()

        return hash
    }

    private fun hammingDistance(
        first: Long,
        second: Long,
    ): Int {
        return java.lang.Long.bitCount(first xor second)
    }
}
