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
import kotlin.math.*

private const val TAG = "DocAnalyzerTwo"

class DocumentAnalyzerTwo(
    private val onFrameSize: (w: Int, h: Int) -> Unit,
    private val onDebug: (bmp: Bitmap?, status: String) -> Unit,
    private val onResult: (polygon: List<Pair<Float, Float>>, cropped: Bitmap?, thumbnail: Bitmap?) -> Unit
) : ImageAnalysis.Analyzer {

    // --- Throttle / FPS ---
    private val minAnalyzeIntervalMs = 70L // ~14 FPS
    private var lastAnalyzeAt = 0L

    // --- Temporal smoothing ---
    private var prevQuad: Array<Point>? = null
    private var prevScore: Double = 0.0
    private var stableFrames: Int = 0

    // --- Params (чуть мягче) ---
    private val alpha = 0.25
    private val minScoreToAccept = 0.41            // было 0.43
    private val keepPrevIfBetterDelta = 0.08
    private val minStableFramesToCrop = 4
    private val movementEpsNormalized = 0.004

    // Hough
    private val houghAngleTol = 15.0
    private val houghMinLineLenRatio = 0.40

    override fun analyze(image: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastAnalyzeAt < minAnalyzeIntervalMs) { image.close(); return }
        lastAnalyzeAt = now

        var status = "analyze..."
        try {
            var src = imageProxyToBitmap(image) ?: run {
                status = "no bitmap"
                onDebug(null, status)
                image.close(); return
            }

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

    private data class DetectOut(
        val polygon: List<Point>,
        val cropped: Bitmap?,
        val debugBmp: Bitmap?,
        val status: String
    )

    private fun detectAndCropDocument(src: Bitmap): DetectOut {
        // Даунскейл
        val maxW = 800
        val scale = if (src.width > maxW) maxW.toDouble() / src.width else 1.0
        val down = if (scale < 1.0) src.scale(
            (src.width * scale).roundToInt(),
            (src.height * scale).roundToInt()
        ) else src

        val diagSrc = hypot(src.width.toDouble(), src.height.toDouble())

        // ---- Предобработка ----
        val rgba = Mat().also { Utils.bitmapToMat(down, it) }
        val bgr = Mat(); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val gray = Mat(); Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)

        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        runCatching {
            val se = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 15.0))
            val tophat = Mat()
            Imgproc.morphologyEx(gray, tophat, Imgproc.MORPH_TOPHAT, se)
            Core.addWeighted(gray, 0.85, tophat, 0.15, 0.0, gray)
            tophat.release(); se.release()
        }

        val grayFiltered = Mat()
        Imgproc.bilateralFilter(gray, grayFiltered, 7, 60.0, 60.0)

        // Бинар + Canny
        val bin = Mat()
        Imgproc.adaptiveThreshold(
            grayFiltered, bin, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, 21, 5.0
        )
        Core.bitwise_not(bin, bin)

        val edgesThin = Mat()
        val med = median(grayFiltered)
        val lower = max(0.0, 0.66 * med)
        val upper = min(255.0, 1.33 * med)
        Imgproc.Canny(grayFiltered, edgesThin, lower, upper)

        // «Толстые» края для edgeSupport
        val edgesFat = Mat()
        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.dilate(edgesThin, edgesFat, k3)

        val comb = Mat()
        Core.bitwise_or(bin, edgesThin, comb)

        val k5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        val k7 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_CLOSE, k3)
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_OPEN, k3)
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_CLOSE, k7)
        Imgproc.dilate(comb, comb, k5)

        // ---- Кандидаты по контурам ----
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(comb, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val frameArea = (down.width * down.height).toDouble()
        val areaMin = frameArea * 0.06
        val perimMin = (down.width + down.height) * 0.55

        val sorted = contours.sortedByDescending { Imgproc.contourArea(it) }.take(12)

        var bestQuadScaled: Array<Point>? = null
        var bestScore = -1.0
        var bestBrk = ScoreBreakdown()
        var bestSrcTag = "contour"

        for (c in sorted) {
            val area = Imgproc.contourArea(c)
            if (area < areaMin) continue

            val peri = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            if (peri < perimMin) continue

            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*c.toArray()), approx, 0.02 * peri, true)
            val quad = when (approx.total().toInt()) {
                4 -> approx.toArray()
                3 -> reconstruct4thFromTriangle(approx.toArray())
                else -> minAreaRectToQuad(c)
            }

            if (!Imgproc.isContourConvex(MatOfPoint(*quad))) continue
            val ordered = orderQuad(quad).toTypedArray()

            val breakdown = scoreQuad(ordered, area, frameArea, down.width, down.height, grayFiltered, edgesFat)
            val score = combinedScore(breakdown)
            if (score > bestScore) {
                bestScore = score
                bestQuadScaled = ordered
                bestBrk = breakdown
                bestSrcTag = "contour"
            }
        }

        // ---- Кандидат по Хаффу (fallback/ко-кандидат) ----
        val houghQuadScaled = findRectByHough(edgesThin, down.width, down.height)
        if (houghQuadScaled != null) {
            val rectArea = polygonArea(orderedList(houghQuadScaled))
            val breakdown = scoreQuad(houghQuadScaled, rectArea, frameArea, down.width, down.height, grayFiltered, edgesFat)
            val score = combinedScore(breakdown)
            val houghValid = breakdown.border >= 0.10 && breakdown.edge >= 0.12
            if ((bestQuadScaled == null && houghValid) || (houghValid && score > bestScore + 0.02)) {
                bestScore = score
                bestQuadScaled = houghQuadScaled
                bestBrk = breakdown
                bestSrcTag = "hough"
            }
        }

        // Нет кандидатов вовсе
        if (bestQuadScaled == null) {
            val dbgBmp = matToBitmap(comb)
            releaseMats(rgba, bgr, gray, grayFiltered, bin, edgesThin, edgesFat, comb, k3, k5, k7)
            return DetectOut(emptyList(), null, dbgBmp, "no quad (contours=${contours.size})")
        }

        // ---- Доп. валидация: теперь МЯГЧЕ и адаптивно ----
        val geomStrong = (bestBrk.rectangularity > 0.70 && bestBrk.rightAngles > 0.75 && bestBrk.size > 0.14)
        val borderOk = bestBrk.border >= 0.08
        val edgeOk = bestBrk.edge >= 0.10 || (geomStrong && bestBrk.border >= 0.06)
        val acceptForCrop = (bestScore >= minScoreToAccept) && borderOk && edgeOk

        // ---- Снятие масштаба к исходному bitmap ----
        val currentQuad = bestQuadScaled.map { p ->
            if (scale < 1.0) Point(p.x / scale, p.y / scale) else Point(p.x, p.y)
        }.toTypedArray()

        // ---- Сглаживание и устойчивость ----
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

        val shouldCrop = acceptForCrop && stableFrames >= minStableFramesToCrop
        val cropped = if (shouldCrop) warpByQuad(src, smoothed) else null

        val dbgBmp = matToBitmap(comb)
        val status = "src=$bestSrcTag " +
                "score=%.2f rect=%.2f ang=%.2f asp=%.2f size=%.2f border=%.2f edge=%.2f bright=%.2f ctr=%.2f ".format(
                    bestScore, bestBrk.rectangularity, bestBrk.rightAngles, bestBrk.aspect,
                    bestBrk.size, bestBrk.border, bestBrk.edge, bestBrk.bright, bestBrk.contrast
                ) +
                "move=%.3f stable=%d accept=%s".format(moveNorm, stableFrames, acceptForCrop)

        releaseMats(rgba, bgr, gray, grayFiltered, bin, edgesThin, edgesFat, comb, k3, k5, k7)
        // ВАЖНО: даже если reject — возвращаем полигон (cropped=null), чтобы рамка рисовалась
        return DetectOut(smoothed.toList(), cropped, dbgBmp, status)
    }

    // ---------- СКОРИНГ ----------
    private data class ScoreBreakdown(
        val rectangularity: Double = 0.0,
        val rightAngles: Double = 0.0,
        val aspect: Double = 0.0,
        val size: Double = 0.0,
        val border: Double = 0.0,
        val edge: Double = 0.0,     // поддержка граней edge-картой
        val bright: Double = 0.0,
        val contrast: Double = 0.0
    )

    private fun scoreQuad(
        quadDown: Array<Point>,
        contourArea: Double,
        frameArea: Double,
        w: Int,
        h: Int,
        grayFiltered: Mat,
        edgesForSupport: Mat
    ): ScoreBreakdown {
        val rect = Imgproc.boundingRect(MatOfPoint(*quadDown))
        val aspect = rect.width.toDouble() / max(1.0, rect.height.toDouble())
        val rectangularity = (contourArea / max(1.0, rect.width.toDouble() * rect.height)).coerceIn(0.0, 1.0)
        val angScore = rightAngleScore(quadDown)
        val aspectNorm = normalizeAspect(aspect)
        val sizeNorm = (contourArea / frameArea).coerceIn(0.0, 1.0)
        val borderScore = borderDistanceScore(quadDown, w, h)
        val edgeScore = edgeSupportScore(quadDown, edgesForSupport)

        val (meanInside, stdInside) = meanStdInsidePolygon(grayFiltered, quadDown)
        val brightScore = ((meanInside - 60.0) / 140.0).coerceIn(0.0, 1.0)
        val contrastScore = (stdInside / 64.0).coerceIn(0.0, 1.0)

        return ScoreBreakdown(rectangularity, angScore, aspectNorm, sizeNorm, borderScore, edgeScore, brightScore, contrastScore)
    }

    private fun combinedScore(b: ScoreBreakdown): Double {
        return 0.30 * b.rectangularity +
                0.20 * b.rightAngles +
                0.16 * b.size +
                0.10 * b.aspect +
                0.10 * b.border +
                0.08 * b.edge +
                0.04 * b.bright +
                0.02 * b.contrast
    }

    // поддержка граней: пересечение линий квада с «утолщённой» картой edges
    private fun edgeSupportScore(quad: Array<Point>, edgesFat: Mat): Double {
        val mask = Mat.zeros(edgesFat.size(), CvType.CV_8UC1)
        val pts = arrayOf(quad[0], quad[1], quad[2], quad[3])
        for (i in 0..3) {
            Imgproc.line(mask, pts[i], pts[(i + 1) % 4], Scalar(255.0), 3)
        }
        val inter = Mat()
        Core.bitwise_and(mask, edgesFat, inter)
        val maskCount = Core.countNonZero(mask)
        val interCount = Core.countNonZero(inter)
        mask.release(); inter.release()
        if (maskCount <= 0) return 0.0
        return (interCount.toDouble() / maskCount.toDouble()).coerceIn(0.0, 1.0)
    }

    private fun borderDistanceScore(quad: Array<Point>, w: Int, h: Int): Double {
        val minWH = min(w, h).toDouble()
        val safe = 0.08 * minWH
        val minDist = quad.minOf { p ->
            val dx = min(p.x, (w - 1).toDouble() - p.x)
            val dy = min(p.y, (h - 1).toDouble() - p.y)
            min(dx, dy)
        }
        return (minDist / max(1e-6, safe)).coerceIn(0.0, 1.0)
    }

    // ---------- Hough прямоугольник (без кастов) ----------
    private fun findRectByHough(edges: Mat, w: Int, h: Int): Array<Point>? {
        val lines = Mat()
        val minLen = min(w, h) * houghMinLineLenRatio
        Imgproc.HoughLinesP(
            edges, lines,
            1.0, Math.PI / 180.0, 120,
            minLen, 10.0
        )
        if (lines.empty()) { lines.release(); return null }

        val segs = ArrayList<Seg>(lines.rows())
        for (i in 0 until lines.rows()) {
            val v = lines.get(i, 0)
            val x1 = v[0]; val y1 = v[1]; val x2 = v[2]; val y2 = v[3]
            val angleDeg = Math.toDegrees(atan2(y2 - y1, x2 - x1))
            segs.add(Seg(x1, y1, x2, y2, angleDeg))
        }
        lines.release()

        val horiz = segs.filter { a ->
            val ang = abs(normalizeAngle(a.angleDeg))
            ang < houghAngleTol || ang > (180.0 - houghAngleTol)
        }.sortedBy { it.cy }

        val vert = segs.filter { a ->
            val ang = abs(normalizeAngle(a.angleDeg))
            ang in (90.0 - houghAngleTol)..(90.0 + houghAngleTol)
        }.sortedBy { it.cx }

        if (horiz.size < 2 || vert.size < 2) return null

        val top = horiz.first()
        val bottom = horiz.last()
        val left = vert.first()
        val right = vert.last()

        val lTop = lineFromSegment(top)
        val lBottom = lineFromSegment(bottom)
        val lLeft = lineFromSegment(left)
        val lRight = lineFromSegment(right)

        val tl = intersect(lTop, lLeft) ?: return null
        val tr = intersect(lTop, lRight) ?: return null
        val br = intersect(lBottom, lRight) ?: return null
        val bl = intersect(lBottom, lLeft) ?: return null

        val inBounds = fun(p: Point): Boolean =
            p.x >= -w * 0.2 && p.x <= w * 1.2 && p.y >= -h * 0.2 && p.y <= h * 1.2
        if (!inBounds(tl) || !inBounds(tr) || !inBounds(br) || !inBounds(bl)) return null

        val quad = orderQuad(arrayOf(tl, tr, br, bl)).toTypedArray()
        val area = polygonArea(orderedList(quad))
        val frameArea = (w * h).toDouble()
        if (area < frameArea * 0.06) return null

        return quad
    }

    private fun normalizeAngle(a: Double): Double {
        var x = a
        while (x <= -180) x += 360.0
        while (x > 180) x -= 360.0
        return x
    }

    private data class Line(val a: Double, val b: Double, val c: Double)

    private data class Seg(val x1: Double, val y1: Double, val x2: Double, val y2: Double, val angleDeg: Double) {
        val cx = (x1 + x2) * 0.5
        val cy = (y1 + y2) * 0.5
    }

    private fun lineFromSegment(s: Seg): Line {
        val x1 = s.x1; val y1 = s.y1; val x2 = s.x2; val y2 = s.y2
        val a = y1 - y2
        val b = x2 - x1
        val c = x1 * y2 - x2 * y1
        val norm = sqrt(a * a + b * b).coerceAtLeast(1e-6)
        return Line(a / norm, b / norm, c / norm)
    }

    private fun intersect(l1: Line, l2: Line): Point? {
        val d = l1.a * l2.b - l2.a * l1.b
        if (abs(d) < 1e-6) return null
        val x = (l2.b * (-l1.c) - l1.b * (-l2.c)) / d
        val y = (l1.a * (-l2.c) - l2.a * (-l1.c)) / d
        return Point(x, y)
    }

    // ---------- Геометрия/оценки ----------
    private fun minAreaRectToQuad(c: MatOfPoint): Array<Point> {
        val mr = Imgproc.minAreaRect(MatOfPoint2f(*c.toArray()))
        val pts = Array(4) { Point() }
        mr.points(pts)
        return orderQuad(pts).toTypedArray()
    }

    private fun reconstruct4thFromTriangle(tri: Array<Point>): Array<Point> {
        val p0 = tri[0]; val p1 = tri[1]; val p2 = tri[2]
        val l01 = lineAB(p0, p1)
        val l12 = lineAB(p1, p2)
        val l20 = lineAB(p2, p0)
        val cands = listOfNotNull(
            intersect(l01, l12),
            intersect(l12, l20),
            intersect(l20, l01)
        )
        var bestQuad: Array<Point>? = null
        var bestArea = -1.0
        for (p in cands) {
            val arr = arrayOf(p0, p1, p2, p)
            val ordered = orderQuad(arr).toTypedArray()
            val area = polygonArea(orderedList(ordered))
            if (area > bestArea) { bestArea = area; bestQuad = ordered }
        }
        return bestQuad ?: minAreaRectToQuad(MatOfPoint(*tri))
    }

    private fun lineAB(a: Point, b: Point): Line {
        val A = a.y - b.y
        val B = b.x - a.x
        val C = a.x * b.y - b.x * a.y
        val norm = sqrt(A * A + B * B).coerceAtLeast(1e-6)
        return Line(A / norm, B / norm, C / norm)
    }

    private fun orderQuad(pts: Array<Point>): List<Point> {
        val tl = pts.minBy { it.x + it.y }
        val br = pts.maxBy { it.x + it.y }
        val sortedDiff = pts.sortedBy { it.x - it.y }
        val bl = sortedDiff.first()
        val tr = sortedDiff.last()
        return listOf(tl, tr, br, bl)
    }

    private fun orderedList(arr: Array<Point>) = listOf(arr[0], arr[1], arr[2], arr[3])

    private fun rightAngleScore(quad: Array<Point>): Double {
        fun angle(a: Point, b: Point, c: Point): Double {
            val abx = a.x - b.x; val aby = a.y - b.y
            val cbx = c.x - b.x; val cby = c.y - b.y
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
        val target = 1.4142
        val diff = abs(r - target)
        return (1.0 - (diff / 0.8)).coerceIn(0.0, 1.0)
    }

    private fun median(gray: Mat): Double {
        val hist = Mat()
        Imgproc.calcHist(listOf(gray), MatOfInt(0), Mat(), hist, MatOfInt(256), MatOfFloat(0f, 256f))
        var acc = 0.0
        val total = gray.rows() * gray.cols().toDouble()
        for (i in 0 until 256) {
            acc += hist.get(i, 0)[0]
            if (acc >= total / 2) { hist.release(); return i.toDouble() }
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

    private fun polygonArea(quad: List<Point>): Double {
        var s = 0.0
        for (i in quad.indices) {
            val a = quad[i]; val b = quad[(i + 1) % quad.size]
            s += a.x * b.y - b.x * a.y
        }
        return abs(s) * 0.5
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

    // -------------------- YUV → Bitmap --------------------
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

        for (row in 0 until height) {
            for (col in 0 until width) {
                val yIndex = row * yRowStride + col * yPixelStride
                out[pos++] = yBuffer.get(yIndex)
            }
        }

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                out[pos++] = vBuffer.get(vIndex)
                out[pos++] = uBuffer.get(uIndex)
            }
        }
        return out
    }

    private fun releaseMats(vararg mats: Mat) {
        mats.forEach { runCatching { it.release() } }
    }

    private fun lerp(a: Point, b: Point, t: Double): Point {
        return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
    }
}