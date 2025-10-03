package com.snowleopard.docscanner.core.imaging

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
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import androidx.core.graphics.scale
import androidx.core.graphics.createBitmap
import java.io.ByteArrayOutputStream

private const val TAG = "DocAnalyzer"

class DocumentAnalyzerOne(
    private val onFrameSize: (w: Int, h: Int) -> Unit,
    private val onDebug: (bmp: Bitmap?, status: String) -> Unit,
    private val onResult: (polygon: List<Pair<Float, Float>>, cropped: Bitmap?, thumbnail: Bitmap?) -> Unit
) : ImageAnalysis.Analyzer {

    // --- Temporal smoothing (state) ---
    private var prevQuad: Array<Point>? = null
    private var prevScore: Double = 0.0

    // Stabilization parameters
    private val alpha = 0.25            // share of the “new” frame
    private val minScoreToAccept = 0.35 // below - we consider the candidate weak
    private val keepPrevIfBetterDelta = 0.08 // score hysteresis

    override fun analyze(image: ImageProxy) {
        var status = "analyze..."
        try {
            var src = imageProxyToBitmap(image) ?: run {
                status = "no bitmap"
                onDebug(null, status)
                image.close(); return
            }

            // We take the turn into account
            val rotation = image.imageInfo.rotationDegrees
            if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                src = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
            }

            onFrameSize(src.width, src.height)

            val out = detectAndCropDocument(src)
            status = out.status
            onDebug(out.debugBmp, status)

            // We provide a polygon (for overlay) and a crop (for preview)
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
        val polygon: List<Point>,    // in the coordinates of the original bitmap
        val cropped: Bitmap?,        // aligned document
        val debugBmp: Bitmap?,       // mask/ribs for debugging
        val status: String
    )

    private fun detectAndCropDocument(src: Bitmap): DetectOut {
        // 1) Downscale for stability/speed (to width ≈ 960)
        val maxW = 960
        val scale = if (src.width > maxW) maxW.toDouble() / src.width else 1.0
        val down = if (scale < 1.0) src.scale(
            (src.width * scale).roundToInt(),
            (src.height * scale).roundToInt()
        ) else src

        // 2) Preprocessing
        val rgba = Mat().also { Utils.bitmapToMat(down, it) }
        val bgr = Mat(); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val gray = Mat(); Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)

        // Contrast + Anti-aliasing (preserve edges)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        // Convert to 8-bit format and filter
        val tmp = Mat()
        gray.convertTo(tmp, CvType.CV_8UC1)
        Imgproc.bilateralFilter(tmp, gray, 7, 50.0, 50.0)
        tmp.release()

        // Adaptive threshold gives the “mass”, Canny gives the “edges” – let’s combine
        val bin = Mat()
        Imgproc.adaptiveThreshold(
            gray, bin, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, /*blockSize*/21, /*C*/5.0
        )
        Core.bitwise_not(bin, bin)

        val edges = Mat()
        // Canny Median Thresholds - Auto-tuning to the Scene
        val med = median(gray)
        val lower = max(0.0, 0.66 * med)
        val upper = min(255.0, 1.33 * med)
        Imgproc.Canny(gray, edges, lower, upper)

        // Combining information
        val comb = Mat()
        Core.bitwise_or(bin, edges, comb)

        // Morphology: close gaps, remove “holes”
        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val k5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_CLOSE, k3)
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_OPEN, k3)
        Imgproc.dilate(comb, comb, k5)

        // Filtering small debris by the area of ​​connected contours
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            comb,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val areaMin = (down.width * down.height) * 0.06 // cut off the small change/rugs
        val frameArea = (down.width * down.height).toDouble()

        var bestQuadScaled: Array<Point>? = null
        var bestScore = -1.0

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < areaMin) continue

            // Approximation to a polygon
            val peri = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*c.toArray()), approx, 0.02 * peri, true)

            // We get a 4-point candidate
            val quad = when (approx.total().toInt()) {
                4 -> approx.toArray()
                else -> minAreaRectToQuad(c) // fallback
            }

            // We standardize and evaluate
            val quadConvex = Imgproc.isContourConvex(MatOfPoint(*quad))
            if (!quadConvex) continue

            val rect = Imgproc.boundingRect(MatOfPoint(*quad))
            val rectArea = rect.width.toDouble() * rect.height
            val rectangularity = (area / max(1.0, rectArea)).coerceIn(0.0, 1.0)

            val aspect = rect.width.toDouble() / max(1.0, rect.height.toDouble())
            val aspectNorm = normalizeAspect(aspect) // proximity to 1.414 or 1/1.414

            val rightAngles = rightAngleScore(quad)

            val sizeNorm = (area / frameArea).coerceIn(0.0, 1.0)

            // Combined score
            val score =
                0.40 * rectangularity +
                        0.25 * rightAngles +
                        0.20 * aspectNorm +
                        0.15 * sizeNorm

            if (score > bestScore) {
                bestScore = score
                bestQuadScaled = orderQuad(quad).toTypedArray()
            }
        }

        // If there is nothing worthy, we will return empty.
        if (bestQuadScaled == null) {
            val dbgBmp = matToBitmap(comb)
            return DetectOut(emptyList(), null, dbgBmp, "no quad (contours=${contours.size})")
        }

        // Let's scale it back to the coordinates of the original bitmap.
        val toSrc: (Point) -> Point = { p ->
            if (scale < 1.0) Point(p.x / scale, p.y / scale) else Point(p.x, p.y)
        }
        val currentQuad = bestQuadScaled.map(toSrc).toTypedArray()

        // --- Temporal stabilization ---
        val usePrev = prevQuad != null &&
                (bestScore + keepPrevIfBetterDelta) < prevScore // the old one was clearly more stable

        val smoothed = if (prevQuad != null && !usePrev) {
            Array(4) { i -> lerp(prevQuad!![i], currentQuad[i], alpha) }
        } else {
            prevQuad ?: currentQuad
        }

        prevQuad = smoothed
        prevScore = if (usePrev) prevScore else max(prevScore * (1 - alpha), bestScore)

        // If the candidate is weak, we show only the smoothed past (without updating the crop)
        val quadForWarp =
            if (bestScore < minScoreToAccept && prevQuad != null) prevQuad!! else currentQuad

        val warped = warpByQuad(src, quadForWarp)

        val dbgBmp = matToBitmap(comb)
        val status = "score=%.2f rect=%.2f ang=%.2f asp=%.2f size=%.2f".format(
            bestScore,
            // layouts for clarity
            /*rectangularity*/ 0.0, /*rightAngles*/ 0.0, /*aspect*/ 0.0, /*size*/ 0.0
        )
        return DetectOut(smoothed.toList(), warped, dbgBmp, status)
    }

    // -------------------- Evaluation/Transformation Assistants --------------------

    private fun minAreaRectToQuad(c: MatOfPoint): Array<Point> {
        val mr = Imgproc.minAreaRect(MatOfPoint2f(*c.toArray()))
        val pts = Array(4) { Point() }
        mr.points(pts)
        return orderQuad(pts).toTypedArray()
    }

    private fun orderQuad(pts: Array<Point>): List<Point> {
        // tl = min(x+y), br = max(x+y), tr = max(x-y), bl = min(x-y)
        val tl = pts.minBy { it.x + it.y }
        val br = pts.maxBy { it.x + it.y }
        val sortedDiff = pts.sortedBy { it.x - it.y }
        val bl = sortedDiff.first()
        val tr = sortedDiff.last()
        return listOf(tl, tr, br, bl)
    }

    private fun rightAngleScore(quad: Array<Point>): Double {
        // We evaluate 4 angles for proximity to 90°
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
        // 1.0 = perfect 90°, tolerance ±20°
        val scores = angs.map { 1.0 - (abs(it - 90.0) / 20.0).coerceIn(0.0, 1.0) }
        return scores.average()
    }

    private fun normalizeAspect(ratio: Double): Double {
        val r = if (ratio < 1.0) 1.0 / ratio else ratio
        val target = 1.4142 // A4
        val diff = abs(r - target)
        // 1.0 when A4 is exact, 0.0 when r is far (> 1.414 + 0.8)
        return (1.0 - (diff / 0.8)).coerceIn(0.0, 1.0)
    }

    private fun median(gray: Mat): Double {
        // fast histogram median
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
            if (acc >= total / 2) return i.toDouble()
        }
        return 127.0
    }

    private fun warpByQuad(src: Bitmap, quad: Array<Point>): Bitmap? {
        val widthTop = dist(quad[0], quad[1])      // TL-TR
        val widthBottom = dist(quad[3], quad[2])   // BL-BR
        val heightLeft = dist(quad[0], quad[3])    // TL-BL
        val heightRight = dist(quad[1], quad[2])   // TR-BR

        val width = max(widthTop, widthBottom).roundToInt().coerceAtLeast(200)
        val height = max(heightLeft, heightRight).roundToInt().coerceAtLeast(200)

        val srcMat = Mat().also { Utils.bitmapToMat(src, it) }
        val dst = Mat()

        val srcPts = MatOfPoint2f(quad[0], quad[1], quad[2], quad[3]) // TL,TR,BR,BL
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(width - 1.0, 0.0),
            Point(width - 1.0, height - 1.0),
            Point(0.0, height - 1.0)
        )
        val m = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        Imgproc.warpPerspective(srcMat, dst, m, Size(width.toDouble(), height.toDouble()))

        val out = createBitmap(width, height)
        Utils.matToBitmap(dst, out)

        srcMat.release(); dst.release(); srcPts.release(); dstPts.release(); m.release()
        return out
    }

    private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)

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

    // -------------------- Safe YUV_420_888 → NV21 → Bitmap --------------------

    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val yuv = image.image ?: return null
        if (image.format != ImageFormat.YUV_420_888) return null

        val nv21 = yuv420888ToNv21(yuv, image.width, image.height)

        val yuvImage = YuvImage(
            nv21, ImageFormat.NV21, image.width, image.height, null
        )
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

private fun lerp(
    a: Point,
    b: Point,
    t: Double
): Point {
    return Point(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t
    )
}