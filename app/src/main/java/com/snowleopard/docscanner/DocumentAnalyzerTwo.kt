package com.snowleopard.docscanner

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

private const val TAG = "DocAnalyzer"

class DocumentAnalyzerTwo(
    private val onFrameSize: (w: Int, h: Int) -> Unit,
    private val onDebug: (bmp: Bitmap?, status: String) -> Unit,
    private val onResult: (polygon: List<Pair<Float, Float>>, cropped: Bitmap?, thumbnail: Bitmap?) -> Unit
) : ImageAnalysis.Analyzer {

    // --- Temporal smoothing (state) ---
    private var prevQuad: Array<Point>? = null
    private var prevScore: Double = 0.0

    // Additional state for "stability"
    private var stableFrames: Int = 0

    // ---------------- Parameters ----------------
    private val alpha = 0.25
    private val minScoreToAccept = 0.40
    private val keepPrevIfBetterDelta = 0.08
    private val minStableFramesToCrop = 2
    private val movementEpsNormalized = 0.004

    override fun analyze(image: ImageProxy) {
        var status = "analyze..."
        try {
            var src = imageProxyToBitmap(image) ?: run {
                status = "no bitmap"
                onDebug(null, status)
                image.close(); return
            }

            // We take into account the rotation of the sensor
            val rotation = image.imageInfo.rotationDegrees
            if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                src = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
            }

            onFrameSize(src.width, src.height)

            val out = detectAndCropDocument(src)
            status = out.status
            onDebug(out.debugBmp, status)

            val thumb = out.cropped?.let { makeThumbnail(it, 200) }
            onResult(out.polygon.map { it.x.toFloat() to it.y.toFloat() }, out.cropped, thumb)
        } catch (t: Throwable) {
            status = "error: ${t.message}"
            Log.e(TAG, "analyze error", t)
            onDebug(null, status)
            onResult(emptyList(), null, null)
        } finally {
            image.close()
        }
    }

    // -------------------- Detection + stabilization --------------------

    private data class DetectOut(
        val polygon: List<Point>,    // coordinates in src
        val cropped: Bitmap?,        // aligned document (or null)
        val debugBmp: Bitmap?,       // debug visualization
        val status: String
    )

    private fun detectAndCropDocument(src: Bitmap): DetectOut {
        // 1) Downscale
        val maxW = 960
        val scale = if (src.width > maxW) maxW.toDouble() / src.width else 1.0
        val down = if (scale < 1.0) src.scale(
            (src.width * scale).roundToInt(),
            (src.height * scale).roundToInt()
        ) else src

        val diagSrc = hypot(src.width.toDouble(), src.height.toDouble())

        // 2) Preprocessing
        val rgba = Mat().also { Utils.bitmapToMat(down, it) }
        val bgr = Mat(); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val gray = Mat(); Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)

        // Contrast
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        // Top-Hat for combating uneven lighting
        runCatching {
            val se = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 15.0))
            val tophat = Mat()
            Imgproc.morphologyEx(gray, tophat, Imgproc.MORPH_TOPHAT, se)
            Core.addWeighted(gray, 0.85, tophat, 0.15, 0.0, gray)
            tophat.release()
            se.release()
        }

        // ❗ IMPORTANT: bilateralFilter is NOT in-place → write it in a separate Mat
        val grayFiltered = Mat()
        Imgproc.bilateralFilter(gray, grayFiltered, /*diameter*/7, 60.0, 60.0)

        // Binarization + Canny on FILTERED
        val bin = Mat()
        Imgproc.adaptiveThreshold(
            grayFiltered, bin, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, /*blockSize*/21, /*C*/5.0
        )
        Core.bitwise_not(bin, bin)

        val edges = Mat()
        val med = median(grayFiltered)
        val lower = max(0.0, 0.66 * med)
        val upper = min(255.0, 1.33 * med)
        Imgproc.Canny(grayFiltered, edges, lower, upper)

        // Let's unite
        val comb = Mat()
        Core.bitwise_or(bin, edges, comb)

        // Morphology
        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val k5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_CLOSE, k3)
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_OPEN, k3)
        Imgproc.dilate(comb, comb, k5)

        // 3) Contours
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            comb,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val frameArea = (down.width * down.height).toDouble()
        val areaMin = frameArea * 0.05
        val perimMin = (down.width + down.height) * 0.5

        val sorted = contours.sortedByDescending { Imgproc.contourArea(it) }.take(12)

        var bestQuadScaled: Array<Point>? = null
        var bestScore = -1.0
        var bestScoreBreakdown = ScoreBreakdown()

        for (c in sorted) {
            val area = Imgproc.contourArea(c)
            if (area < areaMin) continue

            val peri = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            if (peri < perimMin) continue

            // Approximation
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*c.toArray()), approx, 0.02 * peri, true)
            val quad = when (approx.total().toInt()) {
                4 -> approx.toArray()
                else -> minAreaRectToQuad(c)
            }

            val isConvex = Imgproc.isContourConvex(MatOfPoint(*quad))
            if (!isConvex) continue

            val ordered = orderQuad(quad).toTypedArray()

            // Quick screening by aspect
            val rect = Imgproc.boundingRect(MatOfPoint(*ordered))
            val aspect = rect.width.toDouble() / max(1.0, rect.height.toDouble())
            if (aspect < 0.4 || aspect > 2.7) continue

            // frequent counting
            val rectangularity = (area / max(1.0, rect.width.toDouble() * rect.height)).coerceIn(0.0, 1.0)
            val angScore = rightAngleScore(ordered)
            val aspectNorm = normalizeAspect(aspect)
            val sizeNorm = (area / frameArea).coerceIn(0.0, 1.0)
            val borderScore = borderDistanceScore(ordered, down.width, down.height)

            // Brightness/contrast within a polygon — calculated using grayFiltered
            val (meanInside, stdInside) = meanStdInsidePolygon(grayFiltered, ordered)
            val brightScore = ((meanInside - 60.0) / 140.0).coerceIn(0.0, 1.0)
            val contrastScore = (stdInside / 64.0).coerceIn(0.0, 1.0)

            val breakdown = ScoreBreakdown(
                rectangularity = rectangularity,
                rightAngles = angScore,
                aspect = aspectNorm,
                size = sizeNorm,
                border = borderScore,
                bright = brightScore,
                contrast = contrastScore
            )
            val score = combinedScore(breakdown)

            if (score > bestScore) {
                bestScore = score
                bestQuadScaled = ordered
                bestScoreBreakdown = breakdown
            }
        }

        if (bestQuadScaled == null) {
            val dbgBmp = matToBitmap(comb)
            releaseMats(rgba, bgr, gray, grayFiltered, bin, edges, comb, k3, k5)
            return DetectOut(emptyList(), null, dbgBmp, "no quad (contours=${contours.size})")
        }

        // Return to src coordinates
        val currentQuad = bestQuadScaled.map { p ->
            if (scale < 1.0) Point(p.x / scale, p.y / scale) else Point(p.x, p.y)
        }.toTypedArray()

        // --- Temporal stabilization ---
        val usePrev = prevQuad != null && (bestScore + keepPrevIfBetterDelta) < prevScore

        val smoothed = if (prevQuad != null && !usePrev) {
            Array(4) { i -> lerp(prevQuad!![i], currentQuad[i], alpha) }
        } else {
            prevQuad ?: currentQuad
        }

        val moveNorm = if (prevQuad != null) {
            smoothed.indices.map { i -> dist(smoothed[i], prevQuad!![i]) }.average() / diagSrc
        } else 1.0

        prevQuad = smoothed
        prevScore = if (usePrev) prevScore else max(prevScore * (1 - alpha), bestScore)

        stableFrames = if (bestScore >= minScoreToAccept && moveNorm < movementEpsNormalized) {
            (stableFrames + 1).coerceAtMost(10)
        } else 0

        val shouldCrop = bestScore >= minScoreToAccept && stableFrames >= minStableFramesToCrop
        val cropped = if (shouldCrop) warpByQuad(src, smoothed) else null

        val dbgBmp = matToBitmap(comb)
        val status = "score=%.2f  rect=%.2f ang=%.2f asp=%.2f size=%.2f border=%.2f bright=%.2f ctr=%.2f  stable=%d  move=%.3f".format(
            bestScore,
            bestScoreBreakdown.rectangularity,
            bestScoreBreakdown.rightAngles,
            bestScoreBreakdown.aspect,
            bestScoreBreakdown.size,
            bestScoreBreakdown.border,
            bestScoreBreakdown.bright,
            bestScoreBreakdown.contrast,
            stableFrames,
            moveNorm
        )

        releaseMats(rgba, bgr, gray, grayFiltered, bin, edges, comb, k3, k5)
        return DetectOut(smoothed.toList(), cropped, dbgBmp, status)
    }

    // -------------------- Helpers --------------------

    private data class ScoreBreakdown(
        val rectangularity: Double = 0.0,
        val rightAngles: Double = 0.0,
        val aspect: Double = 0.0,
        val size: Double = 0.0,
        val border: Double = 0.0,
        val bright: Double = 0.0,
        val contrast: Double = 0.0
    )

    private fun combinedScore(b: ScoreBreakdown): Double {
        return 0.32 * b.rectangularity +
                0.22 * b.rightAngles +
                0.16 * b.size +
                0.12 * b.aspect +
                0.10 * b.border +
                0.06 * b.bright +
                0.02 * b.contrast
    }

    private fun borderDistanceScore(quad: Array<Point>, w: Int, h: Int): Double {
        val minWH = min(w, h).toDouble()
        val safe = 0.08 * minWH
        val minDist = quad.minOf { p -> min(min(p.x, (w - 1) - p.x), min(p.y, (h - 1) - p.y)) }
        return (minDist / safe).coerceIn(0.0, 1.0)
    }

    private fun minAreaRectToQuad(c: MatOfPoint): Array<Point> {
        val mr = Imgproc.minAreaRect(MatOfPoint2f(*c.toArray()))
        val pts = Array(4) { Point() }
        mr.points(pts)
        return orderQuad(pts).toTypedArray()
    }

    private fun orderQuad(pts: Array<Point>): List<Point> {
        val tl = pts.minBy { it.x + it.y }
        val br = pts.maxBy { it.x + it.y }
        val sortedDiff = pts.sortedBy { it.x - it.y }
        val bl = sortedDiff.first()
        val tr = sortedDiff.last()
        return listOf(tl, tr, br, bl)
    }

    private fun rightAngleScore(quad: Array<Point>): Double {
        fun angle(a: Point, b: Point, c: Point): Double {
            val abx = a.x - b.x
            val aby = a.y - b.y
            val cbx = c.x - b.x
            val cby = c.y - b.y
            val dot = abx * cbx + aby * cby
            val norm = sqrt((abx * abx + aby * aby) * (cbx * cbx + cby * cby))
            val cos = (dot / max(1e-6, norm)).coerceIn(-1.0, 1.0)
            return Math.toDegrees(acos(cos))
        }

        val pts = orderQuad(quad)
        val angs = listOf(
            angle(pts[3], pts[0], pts[1]),
            angle(pts[0], pts[1], pts[2]),
            angle(pts[1], pts[2], pts[3]),
            angle(pts[2], pts[3], pts[0])
        )
        val scores = angs.map { 1.0 - (abs(it - 90.0) / 20.0).coerceIn(0.0, 1.0) }
        return scores.average()
    }

    private fun normalizeAspect(ratio: Double): Double {
        val r = if (ratio < 1.0) 1.0 / ratio else ratio
        val target = 1.4142 // A4 (~√2)
        val diff = abs(r - target)
        return (1.0 - (diff / 0.8)).coerceIn(0.0, 1.0)
    }

    private fun median(gray: Mat): Double {
        val hist = Mat()
        Imgproc.calcHist(
            listOf(gray),
            MatOfInt(0),
            Mat(),
            hist,
            MatOfInt(256),
            MatOfFloat(0f, 256f)
        )
        var acc = 0.0
        val total = gray.rows() * gray.cols().toDouble()
        for (i in 0 until 256) {
            acc += hist.get(i, 0)[0]
            if (acc >= total / 2) {
                hist.release()
                return i.toDouble()
            }
        }
        hist.release()
        return 127.0
    }

    private fun meanStdInsidePolygon(gray: Mat, quadDown: Array<Point>): Pair<Double, Double> {
        val mask = Mat.zeros(gray.size(), CvType.CV_8UC1)
        Imgproc.fillConvexPoly(mask, MatOfPoint(*quadDown), Scalar(255.0))
        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(gray, mean, std, mask)
        val m = mean.get(0, 0)[0]
        val s = std.get(0, 0)[0]
        mean.release(); std.release(); mask.release()
        return m to s
    }

    private fun warpByQuad(src: Bitmap, quad: Array<Point>): Bitmap? {
        val wA = dist(quad[1], quad[2])
        val wB = dist(quad[0], quad[3])
        val hA = dist(quad[0], quad[1])
        val hB = dist(quad[3], quad[2])
        val width = max(wA, wB).roundToInt().coerceAtLeast(200)
        val height = max(hA, hB).roundToInt().coerceAtLeast(200)

        val srcMat = Mat(); Utils.bitmapToMat(src, srcMat)
        val dst = Mat()

        val srcPts = MatOfPoint2f(quad[0], quad[1], quad[2], quad[3])
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(width - 1.0, 0.0),
            Point(width - 1.0, height - 1.0),
            Point(0.0, height - 1.0)
        )
        val mat = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        Imgproc.warpPerspective(srcMat, dst, mat, Size(width.toDouble(), height.toDouble()))

        val out = createBitmap(width, height)
        Utils.matToBitmap(dst, out)

        srcMat.release(); dst.release(); srcPts.release(); dstPts.release(); mat.release()
        return out
    }

    private fun matToBitmap(mat: Mat): Bitmap {
        val vis = Mat()
        if (mat.type() == CvType.CV_8UC1) {
            Imgproc.cvtColor(mat, vis, Imgproc.COLOR_GRAY2RGBA)
        } else {
            Imgproc.cvtColor(mat, vis, Imgproc.COLOR_BGR2RGBA)
        }
        val bmp = createBitmap(vis.cols(), vis.rows())
        Utils.matToBitmap(vis, bmp)
        vis.release()
        return bmp
    }

    private fun makeThumbnail(b: Bitmap, width: Int): Bitmap {
        val r = width.toFloat() / b.width
        val h = (b.height * r).roundToInt()
        return b.scale(width, h)
    }

    private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)

    private fun releaseMats(vararg mats: Mat) {
        mats.forEach { runCatching { it.release() } }
    }

    // -------------------- YUV_420_888 → NV21 → Bitmap --------------------

    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val yuv = image.image ?: return null
        if (image.format != ImageFormat.YUV_420_888) return null

        val nv21 = yuv420888ToNv21(yuv, image.width, image.height)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 95, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun yuv420888ToNv21(image: Image, width: Int, height: Int): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        val out = ByteArray(width * height + 2 * (width / 2) * (height / 2))
        var pos = 0

        // Y
        for (row in 0 until height) {
            for (col in 0 until width) {
                val yIndex = row * yRowStride + col * yPixelStride
                out[pos++] = yBuffer.get(yIndex)
            }
        }

        // VU (NV21)
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                out[pos++] = vBuffer.get(vIndex) // V
                out[pos++] = uBuffer.get(uIndex) // U
            }
        }
        return out
    }
}

private fun lerp(a: Point, b: Point, t: Double): Point {
    return Point(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t
    )
}