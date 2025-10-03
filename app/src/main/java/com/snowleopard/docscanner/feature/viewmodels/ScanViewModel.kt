package com.snowleopard.docscanner.feature.viewmodels

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.snowleopard.docscanner.core.data.repository.PagesRepository
import com.snowleopard.docscanner.core.data.store.PagesStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt
import androidx.core.graphics.scale
import com.snowleopard.docscanner.core.data.model.ScannedPage
import kotlinx.coroutines.flow.StateFlow

class ScanViewModel(
    private val repo: PagesRepository,
    private val store: PagesStore,
) : ViewModel() {

    // ----- Live overlay -----
    private val _livePreviewPolygon = MutableStateFlow<List<Pair<Float, Float>>>(emptyList())
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

    private val _wavePolygon = MutableStateFlow<List<Pair<Float, Float>>>(emptyList())
    val wavePolygon = _wavePolygon.asStateFlow()

    // Freeze/duplicate FSM
    private val _captureFreeze = MutableStateFlow(false)
    val captureFreeze = _captureFreeze.asStateFlow()
    private var frozenAt: Long = 0L
    private var framesWithoutDoc: Int = 0

    private data class CaptureSig(
        val hash: Long,
        val time: Long,
        val centroidN: Pair<Float, Float>,
        val areaN: Float,
    )
    private var lastSig: CaptureSig? = null

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

    // Params
    private val captureCooldownMs = 900L
    private val minFreezeMs = 1400L
    private val requireDocAbsenceFrames = 5

    private val moveEpsNormalized = 0.02f
    private val angleMin = 0.60
    private val aspectMin = 0.35
    private val areaMinNorm = 0.08f

    private val lockFramesRequired = 8
    private val lockDecay = 0.18f

    private val confirmDurationMs = 380L
    private val acceptStalenessMs = 2200L

    private val dupTimeWindowMs = 2000L
    private val dupHashHammingThresh = 5
    private val dupCentroidShiftThresh = 0.03f
    private val dupAreaRelDiffThresh = 0.12f

    fun setStatus(s: String) { _status.value = s }
    fun setDebugFrame(bmp: Bitmap?) { _debugFrame.value = bmp }
    fun setFrameSize(w: Int, h: Int) { _frameSize.value = w to h }
    fun updateOverlay(polygon: List<Pair<Float, Float>>, thumb: Bitmap?) {
        _livePreviewPolygon.value = polygon
        if (thumb != null) _liveThumbnail.value = thumb
    }

    fun onAnalyzerResult(polygon: List<Pair<Float, Float>>, cropped: Bitmap?, thumbnail: Bitmap?) {
        updateOverlay(polygon, thumbnail)
        if (cropped != null) {
            latestCropped = cropped
            latestCroppedAt = System.currentTimeMillis()
        }
        if (!_captureFreeze.value) {
            updateLockProgress(polygon)
            checkConfirmAndCapture()
        } else {
            _lockProgress.value = 0f
            maybeUnfreeze(polygon)
        }
    }

    private fun updateLockProgress(poly: List<Pair<Float, Float>>) {
        val fs = _frameSize.value ?: return
        if (poly.size < 4) {
            stableFrames = 0
            prevPoly = null
            _lockProgress.value = max(0f, _lockProgress.value - lockDecay)
            return
        }
        val (w, h) = fs
        val diag = hypot(w.toDouble(), h.toDouble()).toFloat().coerceAtLeast(1f)
        val areaNorm = (polygonArea(poly) / (w.toFloat() * h.toFloat())).coerceIn(0f, 1f)
        val centroid = polygonCentroid(poly)
        val angles = rightAngleScore(poly)
        val aspect = aspectA4Score(poly)
        val moved = prevPoly?.let { distance(centroid, polygonCentroid(it)) / diag } ?: 1f

        val geometryOk = angles >= angleMin && aspect >= aspectMin && areaNorm >= areaMinNorm
        val stable = moved < moveEpsNormalized

        prevPoly = poly
        lastPolyAreaNorm = areaNorm
        lastPolyCentroid = centroid

        if (geometryOk && stable) {
            stableFrames++
            val step = 1f / lockFramesRequired
            _lockProgress.value = (_lockProgress.value + step).coerceAtMost(1f)
        } else {
            stableFrames = 0
            _lockProgress.value = max(0f, _lockProgress.value - lockDecay)
        }
    }

    private fun checkConfirmAndCapture() {
        if (_lockProgress.value >= 1f && _confirmAt.value == 0L) {
            _confirmAt.value = System.currentTimeMillis()
            viewModelScope.launch {
                delay(confirmDurationMs)
                val stillValid = _lockProgress.value >= 0.8f && !_captureFreeze.value
                val recentCropped = (System.currentTimeMillis() - latestCroppedAt) <= acceptStalenessMs
                val bmp = latestCropped
                if (stillValid && recentCropped && bmp != null) {
                    performCapture(bmp, prevPoly ?: emptyList())
                }
                _confirmAt.value = 0L
                _lockProgress.value = 0f
                stableFrames = 0
            }
        }
    }

    private fun performCapture(cropped: Bitmap, polygonForWave: List<Pair<Float, Float>>) {
        val now = System.currentTimeMillis()
        if (_captureInProgress.value) return
        if (now - _lastCapturedAt.value < captureCooldownMs) return

        _captureInProgress.value = true
        viewModelScope.launch {
            val newHash = withContext(Dispatchers.Default) { averageHash64(cropped) }
            val metrics = currentPolyMetrics(polygonForWave)
            val duplicate = isLikelyDuplicate(newHash, metrics, now)
            if (!duplicate) {
                val page = withContext(Dispatchers.Default) {
                    repo.savePage(cropped, preferPortrait = true)
                }
                _wavePolygon.value = polygonForWave
                _lastCapturedAt.value = now
                store.add(page) // <— добавляем в общее хранилище
                enableFreeze(CaptureSig(newHash, now, metrics.centroidN, metrics.areaN))
            }
            _captureInProgress.value = false
        }
    }

    private fun enableFreeze(sig: CaptureSig) {
        _captureFreeze.value = true
        frozenAt = System.currentTimeMillis()
        framesWithoutDoc = 0
        lastSig = sig
    }

    private fun maybeUnfreeze(currentPoly: List<Pair<Float, Float>>) {
        if (!_captureFreeze.value) return
        if (System.currentTimeMillis() - frozenAt < minFreezeMs) return

        val fs = _frameSize.value
        val hasDoc = currentPoly.size >= 4
        if (!hasDoc) {
            framesWithoutDoc++
            if (framesWithoutDoc >= requireDocAbsenceFrames) disableFreeze(true)
            return
        } else framesWithoutDoc = 0

        if (fs != null && lastPolyCentroid != null) {
            val (w, h) = fs
            val diag = hypot(w.toDouble(), h.toDouble()).toFloat().coerceAtLeast(1f)
            val curAreaNorm = (polygonArea(currentPoly) / (w.toFloat() * h.toFloat())).coerceIn(0f, 1f)
            val curCentroid = polygonCentroid(currentPoly)
            val shift = distance(curCentroid, lastPolyCentroid!!) / diag
            val areaRelChange = if (lastPolyAreaNorm > 1e-6f)
                abs(curAreaNorm - lastPolyAreaNorm) / lastPolyAreaNorm else 1f
            if (shift > 0.10f || areaRelChange > 0.25f) disableFreeze(true)
        }
    }

    private fun disableFreeze(resetSignature: Boolean) {
        _captureFreeze.value = false
        framesWithoutDoc = 0
        if (resetSignature) lastSig = null
    }

    fun removeLastPage() {
        val removed = store.removeLast() ?: return
        viewModelScope.launch(Dispatchers.IO) { repo.deletePage(removed) }
    }

    fun clearSession() {
        viewModelScope.launch(Dispatchers.IO) {
            repo.clearSession()
            withContext(Dispatchers.Main) {
                store.clear()
                _liveThumbnail.value = null
                _wavePolygon.value = emptyList()
                _lastCapturedAt.value = 0L
                _captureFreeze.value = false
                framesWithoutDoc = 0
                prevPoly = null
                stableFrames = 0
                _lockProgress.value = 0f
                _confirmAt.value = 0L
                latestCropped = null
                lastSig = null
            }
        }
    }

    // ----- Utils for duplicate -----
    private data class PolyMetrics(val centroidN: Pair<Float, Float>, val areaN: Float)
    private fun currentPolyMetrics(poly: List<Pair<Float, Float>>): PolyMetrics {
        val fs = _frameSize.value
        val (w, h) = fs ?: (1 to 1)
        val areaN = (polygonArea(poly) / (w.toFloat() * h.toFloat())).coerceIn(0f, 1f)
        val centroid = polygonCentroid(poly)
        val diag = hypot(w.toDouble(), h.toDouble()).toFloat().coerceAtLeast(1f)
        val cxN = centroid.first / diag
        val cyN = centroid.second / diag
        return PolyMetrics(cxN to cyN, areaN)
    }

    private fun isLikelyDuplicate(newHash: Long, metrics: PolyMetrics, now: Long): Boolean {
        val sig = lastSig ?: return false
        if (now - sig.time > dupTimeWindowMs) return false
        val hamming = hammingDistance(newHash, sig.hash)
        val dc = sqrt(
            (metrics.centroidN.first - sig.centroidN.first) * (metrics.centroidN.first - sig.centroidN.first) +
                    (metrics.centroidN.second - sig.centroidN.second) * (metrics.centroidN.second - sig.centroidN.second)
        )
        val da = if (sig.areaN > 1e-6f) abs(metrics.areaN - sig.areaN) / sig.areaN else 1f
        return (hamming <= dupHashHammingThresh) && (dc < dupCentroidShiftThresh) && (da < dupAreaRelDiffThresh)
    }

    private fun polygonArea(poly: List<Pair<Float, Float>>): Float {
        var s = 0f
        for (i in poly.indices) {
            val (x1, y1) = poly[i]
            val (x2, y2) = poly[(i + 1) % poly.size]
            s += (x1 * y2 - x2 * y1)
        }
        return abs(s) * 0.5f
    }

    private fun polygonCentroid(poly: List<Pair<Float, Float>>): Pair<Float, Float> {
        var cx = 0f; var cy = 0f; var a = 0f
        for (i in poly.indices) {
            val (x1, y1) = poly[i]
            val (x2, y2) = poly[(i + 1) % poly.size]
            val cross = x1 * y2 - x2 * y1
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
            a += cross
        }
        val area = a * 0.5f
        if (abs(area) < 1e-6f) return 0f to 0f
        val k = 1f / (6f * area)
        return (cx * k) to (cy * k)
    }

    private fun distance(a: Pair<Float, Float>, b: Pair<Float, Float>): Float {
        val dx = a.first - b.first
        val dy = a.second - b.second
        return sqrt(dx * dx + dy * dy)
    }

    private fun rightAngleScore(poly: List<Pair<Float, Float>>): Double {
        fun angle(ax: Float, ay: Float, bx: Float, by: Float, cx: Float, cy: Float): Double {
            val abx = ax - bx
            val aby = ay - by
            val cbx = cx - bx
            val cby = cy - by
            val dot = abx * cbx + aby * cby
            val norm = sqrt((abx * abx + aby * aby).toDouble() * (cbx * cbx + cby * cby).toDouble()).coerceAtLeast(1e-6)
            val cos = (dot / norm).coerceIn(-1.0, 1.0)
            return Math.toDegrees(acos(cos))
        }
        if (poly.size < 4) return 0.0
        val p = poly
        val angs = listOf(
            angle(p[3].first, p[3].second, p[0].first, p[0].second, p[1].first, p[1].second),
            angle(p[0].first, p[0].second, p[1].first, p[1].second, p[2].first, p[2].second),
            angle(p[1].first, p[1].second, p[2].first, p[2].second, p[3].first, p[3].second),
            angle(p[2].first, p[2].second, p[3].first, p[3].second, p[0].first, p[0].second)
        )
        val scores = angs.map { 1.0 - (abs(it - 90.0) / 20.0).coerceIn(0.0, 1.0) }
        return scores.average()
    }

    private fun aspectA4Score(poly: List<Pair<Float, Float>>): Double {
        if (poly.size < 4) return 0.0
        val xs = poly.map { it.first }
        val ys = poly.map { it.second }
        val maxX = xs.maxOrNull() ?: return 0.0
        val minX = xs.minOrNull() ?: return 0.0
        val maxY = ys.maxOrNull() ?: return 0.0
        val minY = ys.minOrNull() ?: return 0.0
        val w = (maxX - minX).coerceAtLeast(1f)
        val h = (maxY - minY).coerceAtLeast(1f)
        val r = if (w > h) w / h else h / w
        val target = 1.4142f
        val diff = abs(r - target)
        return (1.0 - (diff / 0.8f)).coerceIn(0.0, 1.0)
    }

    private fun averageHash64(bmp: Bitmap): Long {
        val small = bmp.scale(8, 8)
        val pixels = IntArray(64)
        small.getPixels(pixels, 0, 8, 0, 0, 8, 8)
        var sum = 0
        val gray = IntArray(64)
        for (i in 0 until 64) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = (p) and 0xFF
            val v = ((r * 30 + g * 59 + b * 11) / 100)
            gray[i] = v
            sum += v
        }
        val avg = sum / 64
        var hash = 0L
        for (i in 0 until 64) if (gray[i] >= avg) hash = hash or (1L shl i)
        small.recycle()
        return hash
    }

    private fun hammingDistance(a: Long, b: Long): Int = java.lang.Long.bitCount(a xor b)
}
