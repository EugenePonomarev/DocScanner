package com.snowleopard.docscanner

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
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
import kotlin.math.*

private const val TAG = "DocAnalyzerTwo"

class DocumentAnalyzerTwo(
    private val onFrameSize: (w: Int, h: Int) -> Unit,
    private val onDebug: (bmp: Bitmap?, status: String) -> Unit,
    private val onResult: (polygon: List<Pair<Float, Float>>, cropped: Bitmap?, thumbnail: Bitmap?) -> Unit,
) : ImageAnalysis.Analyzer {

    // --- Тайминг анализа ---
    private val minAnalyzeIntervalMs = 60L
    private var lastAnalyzeAt = 0L
    private var frameIndex = 0

    // --- Стабилизация/порог ---
    private var prevQuad: Array<Point>? = null
    private var prevScore: Double = 0.0
    private var stableFrames: Int = 0

    private val alpha = 0.22
    private val minScoreToAccept = 0.34
    private val keepPrevIfBetterDelta = 0.08
    private val minStableFramesToCrop = 2
    private val fastLockScore = 0.65
    private val movementEpsNormalized = 0.010

    // --- ROI-ускорение ---
    private val maxWDetectFull = 520                    // глобальный поиск
    private val roiPadFrac = 0.18                       // запас вокруг предыдущего квада
    private val maxWDetectRoiMin = 320                  // минимальная ширина детекции для ROI

    // --- Pass-план ---
    private val heavyEvery = 5                          // heavy-pass реже
    private var missCounter = 0                         // если подряд нет кандидата — форсим heavy-pass

    // --- Hough ---
    private val houghAngleTol = 15.0
    private val houghMinLineLenRatio = 0.40

    // Постпроцесс (только при фиксации)
    private enum class EnhanceProfile { NONE, SOFT, BW }
    private val enhanceProfile = EnhanceProfile.SOFT

    // --- Профайлинг ---
    private data class T(var ms: Long = 0)
    private data class Times(
        val tResize: T = T(),
        val tPrep: T = T(),
        val tEdges: T = T(),
        val tCnt: T = T(),
        val tHough: T = T(),
        val tTotal: T = T(),
    )
    private val times = Times()

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastAnalyzeAt < minAnalyzeIntervalMs) { image.close(); return }
        lastAnalyzeAt = now
        frameIndex++

        var status = "analyze..."
        try {
            val t0 = System.nanoTime()

            // 1) Серый канал (Y)
            val graySrc = yPlaneToGrayMat(image) ?: run {
                onDebug(null, "no gray")
                image.close(); return
            }

            // 2) Поворот под дисплей
            val rotation = image.imageInfo.rotationDegrees
            val gray = rotateMat(graySrc, rotation)
            graySrc.release()
            onFrameSize(gray.cols(), gray.rows())

            // 3) Детекция/кроп (цвет тянем ТОЛЬКО при фиксации)
            val out = detectAndCropFromGray(
                gray = gray,
                colorSupplier = {
                    val color = imageProxyToColorBitmap(image) ?: return@detectAndCropFromGray null
                    if (rotation != 0) {
                        val m = Matrix().apply { postRotate(rotation.toFloat()) }
                        Bitmap.createBitmap(color, 0, 0, color.width, color.height, m, true)
                    } else color
                }
            )
            gray.release()

            times.tTotal.ms = ((System.nanoTime() - t0) / 1_000_000).coerceAtLeast(0)

            status = out.status +
                    " | t(ms):R=${times.tResize.ms} P=${times.tPrep.ms} E=${times.tEdges.ms} C=${times.tCnt.ms} H=${times.tHough.ms} T=${times.tTotal.ms}"
            onDebug(out.debugBmp, status)

            val thumb = out.cropped?.let { makeThumbnail(it, 240) }
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

    // ------------------------------------------------------------------------------------

    private data class DetectOut(
        val polygon: List<Point>,
        val cropped: Bitmap?,
        val debugBmp: Bitmap?,
        val status: String,
    )

    private fun detectAndCropFromGray(
        gray: Mat,
        colorSupplier: () -> Bitmap?,
    ): DetectOut {
        val srcW = gray.cols()
        val srcH = gray.rows()
        val diagSrc = hypot(srcW.toDouble(), srcH.toDouble())

        val tR0 = System.nanoTime()
        // --- 1) downscale (глобальный) ---
        val scaleFull = if (srcW > maxWDetectFull) maxWDetectFull.toDouble() / srcW else 1.0
        val dwFull = (srcW * scaleFull).roundToInt()
        val dhFull = (srcH * scaleFull).roundToInt()
        val grayDownFull = Mat()
        Imgproc.resize(gray, grayDownFull, Size(dwFull.toDouble(), dhFull.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        times.tResize.ms = ((System.nanoTime() - tR0) / 1_000_000).coerceAtLeast(0)

        // --- 2) ROI вокруг prevQuad ---
        val useRoi = prevQuad != null
        val roiRectDown: org.opencv.core.Rect? = if (useRoi) {
            val q = prevQuad!!.map { p -> Point(p.x * scaleFull, p.y * scaleFull) }
            val minX = q.minOf { it.x }; val maxX = q.maxOf { it.x }
            val minY = q.minOf { it.y }; val maxY = q.maxOf { it.y }
            val pad = (max(maxX - minX, maxY - minY) * roiPadFrac).coerceAtLeast(12.0)
            val x0 = floor((minX - pad).coerceAtLeast(0.0)).toInt()
            val y0 = floor((minY - pad).coerceAtLeast(0.0)).toInt()
            val x1 = ceil((maxX + pad).coerceAtMost(dwFull - 1.0)).toInt()
            val y1 = ceil((maxY + pad).coerceAtMost(dhFull - 1.0)).toInt()
            val w = (x1 - x0).coerceAtLeast(32)
            val h = (y1 - y0).coerceAtLeast(32)
            org.opencv.core.Rect(x0, y0, w, h)
        } else null

        val grayWork: Mat
        val dwWork: Int
        val dhWork: Int
        val roiOffsetX: Int
        val roiOffsetY: Int
        val scaleRoi: Double

        if (roiRectDown != null) {
            val grayRoi = grayDownFull.submat(roiRectDown)
            val needScale = roiRectDown.width > maxWDetectRoiMin
            scaleRoi = if (needScale) maxWDetectRoiMin.toDouble() / roiRectDown.width else 1.0
            if (scaleRoi < 1.0) {
                val tmp = Mat()
                Imgproc.resize(grayRoi, tmp, Size((roiRectDown.width * scaleRoi), (roiRectDown.height * scaleRoi)))
                grayRoi.release()
                grayWork = tmp
            } else {
                grayWork = grayRoi
            }
            dwWork = grayWork.cols(); dhWork = grayWork.rows()
            roiOffsetX = roiRectDown.x
            roiOffsetY = roiRectDown.y
        } else {
            grayWork = grayDownFull
            dwWork = dwFull; dhWork = dhFull
            roiOffsetX = 0; roiOffsetY = 0
            scaleRoi = 1.0
        }

        fun workToDownFull(p: Point): Point =
            Point(p.x / scaleRoi + roiOffsetX, p.y / scaleRoi + roiOffsetY)

        // --- 3) Cheap/Heavy препроцесс ---
        val doHeavy = (frameIndex % heavyEvery == 0) || missCounter >= 3 || !useRoi
        val tP0 = System.nanoTime()
        if (doHeavy) {
            // Heavy: немного CLAHE для стабильности
            val clahe = Imgproc.createCLAHE(1.2, Size(8.0, 8.0))
            clahe.apply(grayWork, grayWork)
        }
        // Общий blur
        Imgproc.GaussianBlur(grayWork, grayWork, Size(5.0, 5.0), 0.0)
        times.tPrep.ms = ((System.nanoTime() - tP0) / 1_000_000).coerceAtLeast(0)

        val tE0 = System.nanoTime()
        // Edges
        val edgesThin = Mat()
        if (doHeavy) {
            // Автопорог по медиане дороже — используем только в heavy
            val med = median(grayWork)
            val lower = max(0.0, 0.66 * med)
            val upper = min(255.0, 1.33 * med)
            Imgproc.Canny(grayWork, edgesThin, lower, upper)
        } else {
            // Дёшево: пороги от среднего
            val meanVal = Core.mean(grayWork).`val`[0].coerceIn(1.0, 254.0)
            val lower = (meanVal * 0.66).coerceIn(0.0, 255.0)
            val upper = (meanVal * 1.33).coerceIn(0.0, 255.0)
            Imgproc.Canny(grayWork, edgesThin, lower, upper)
        }

        // Дополнительно бинар в heavy (даёт массу)
        val comb = if (doHeavy) {
            val bin = Mat()
            Imgproc.adaptiveThreshold(
                grayWork, bin, 255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY, 19, 4.0
            )
            Core.bitwise_not(bin, bin)
            val tmp = Mat()
            Core.bitwise_or(bin, edgesThin, tmp)
            bin.release()
            tmp
        } else {
            edgesThin.clone()
        }

        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_CLOSE, k3, Point(-1.0, -1.0), 1)
        times.tEdges.ms = ((System.nanoTime() - tE0) / 1_000_000).coerceAtLeast(0)

        // --- 4) Контуры (в WORK-координатах) ---
        val tC0 = System.nanoTime()
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(comb, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val frameAreaWork = (dwWork * dhWork).toDouble()
        val areaMin = frameAreaWork * 0.08                      // чуть строже — меньше соринок
        val perimMin = (dwWork + dhWork) * 0.55

        val sorted = contours.sortedByDescending { Imgproc.contourArea(it) }.take(8)

        var bestQuadWork: Array<Point>? = null
        var bestScore = -1.0
        var bestBrk = ScoreBreakdown()
        var srcTag = if (useRoi) if (doHeavy) "contour-roi-H" else "contour-roi" else if (doHeavy) "contour-full-H" else "contour-full"

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

            val orderedWork = orderQuad(quad).toTypedArray()

            // ⚠️ ВАЖНО: borderDistance считаем в КООРДИНАТАХ downFull!
            val orderedFull = Array(4) { i -> workToDownFull(orderedWork[i]) }

            // Быстрый «гео»-скользящий скор (без edgeSupport, чтобы экономить)
            val breakdownFast = scoreQuadFast(orderedWork, orderedFull, area, frameAreaWork, dwFull, dhFull)
            var score = combinedScoreFast(breakdownFast)

            // Только если быстрый скор перспективный — считаем edgeSupport (дороже)
            val breakdown = if (score >= 0.25) {
                val edgeScore = edgeSupportScore(orderedWork, comb) // comb в WORK-координатах
                breakdownFast.copy(edge = edgeScore)
            } else breakdownFast

            // Итоговый комбинированный скор
            score = combinedScore(breakdown)

            if (score > bestScore) {
                bestScore = score
                bestQuadWork = orderedWork
                bestBrk = breakdown
            }
        }
        times.tCnt.ms = ((System.nanoTime() - tC0) / 1_000_000).coerceAtLeast(0)

        // --- 5) Hough — только при необходимости и редко ---
        val tH0 = System.nanoTime()
        if (bestQuadWork == null && doHeavy) {
            val houghQuad = findRectByHough(edgesThin, dwWork, dhWork)
            if (houghQuad != null) {
                val rectArea = polygonArea(orderedList(houghQuad))
                val orderedFull = Array(4) { i -> workToDownFull(houghQuad[i]) }
                val breakdownFast = scoreQuadFast(houghQuad, orderedFull, rectArea, frameAreaWork, dwFull, dhFull)
                val edgeScore = edgeSupportScore(houghQuad, comb)
                val breakdown = breakdownFast.copy(edge = edgeScore)
                val score = combinedScore(breakdown)
                val hValid = breakdown.border >= 0.08 && (breakdown.edge >= 0.10 || breakdown.rectangularity >= 0.80)
                if (hValid) {
                    bestQuadWork = houghQuad
                    bestScore = score
                    bestBrk = breakdown
                    srcTag = if (useRoi) "hough-roi" else "hough-full"
                }
            }
        }
        times.tHough.ms = ((System.nanoTime() - tH0) / 1_000_000).coerceAtLeast(0)

        // --- 6) Координаты → downFull → src ---
        if (bestQuadWork == null) {
            missCounter++
            val dbg = matToBitmap(comb)
            releaseMats(grayDownFull, grayWork, edgesThin, comb, k3)
            return DetectOut(emptyList(), null, dbg, "no quad (${srcTag}) miss=$missCounter")
        } else {
            missCounter = 0
        }

        val bestQuadDownFull = bestQuadWork.map(::workToDownFull).toTypedArray()

        val toSrc: (Point) -> Point = { p ->
            if (scaleFull < 1.0) Point(p.x / scaleFull, p.y / scaleFull) else Point(p.x, p.y)
        }
        val currentQuad = bestQuadDownFull.map(toSrc).toTypedArray()

        // --- 7) стабилизация/решение ---
        val usePrevKeep = prevQuad != null && (bestScore + keepPrevIfBetterDelta) < prevScore
        val smoothed = if (prevQuad != null && !usePrevKeep) {
            Array(4) { i -> lerp(prevQuad!![i], currentQuad[i], alpha) }
        } else prevQuad ?: currentQuad

        val moveNorm = if (prevQuad != null) {
            smoothed.indices.map { i -> dist(smoothed[i], prevQuad!![i]) }.average() / hypot(srcW.toDouble(), srcH.toDouble())
        } else 1.0

        prevQuad = smoothed
        prevScore = if (usePrevKeep) prevScore else max(prevScore * (1 - alpha), bestScore)

        // Fast-lock: если очень уверены и почти нет движения — не ждём много кадров
        val needStable = if (bestScore >= fastLockScore && moveNorm < movementEpsNormalized * 0.7) 1 else minStableFramesToCrop

        stableFrames = if (bestScore >= minScoreToAccept && moveNorm < movementEpsNormalized) {
            (stableFrames + 1).coerceAtMost(8)
        } else 0

        val geomStrong = (bestBrk.rectangularity > 0.68 && bestBrk.rightAngles > 0.72 && bestBrk.size > 0.12)
        val borderOk = bestBrk.border >= 0.06
        val edgeOk = bestBrk.edge >= 0.09 || (geomStrong && bestBrk.border >= 0.05)
        val acceptForCrop = (bestScore >= minScoreToAccept) && borderOk && edgeOk
        val shouldCrop = acceptForCrop && stableFrames >= needStable

        var cropped: Bitmap? = null
        if (shouldCrop) {
            val color = colorSupplier()
            if (color != null) {
                cropped = warpByQuad(color, smoothed)
                if (cropped != null) cropped = applyPostproc(cropped, enhanceProfile)
            }
        }

        val dbgBmp = matToBitmap(comb)
        val status = "src=$srcTag score=%.2f rect=%.2f ang=%.2f asp=%.2f size=%.2f border=%.2f edge=%.2f move=%.3f stable=%d/%d accept=%s".format(
            bestScore, bestBrk.rectangularity, bestBrk.rightAngles, bestBrk.aspect,
            bestBrk.size, bestBrk.border, bestBrk.edge, moveNorm, stableFrames, needStable, acceptForCrop
        )

        releaseMats(grayDownFull, grayWork, edgesThin, comb, k3)
        return DetectOut(smoothed.toList(), cropped, dbgBmp, status)
    }

    // ---------- СКОРИНГ / УТИЛИТЫ ----------

    private data class ScoreBreakdown(
        val rectangularity: Double = 0.0,
        val rightAngles: Double = 0.0,
        val aspect: Double = 0.0,
        val size: Double = 0.0,
        val border: Double = 0.0,
        val edge: Double = 0.0,
    )

    // Быстрый скор без edge/bright/contrast
    private fun scoreQuadFast(
        quadWork: Array<Point>,               // в WORK-координатах
        quadDownFull: Array<Point>,           // тот же контур в downFull-координатах
        contourAreaWork: Double,
        frameAreaWork: Double,
        dwFull: Int,
        dhFull: Int,
    ): ScoreBreakdown {
        val rect = Imgproc.boundingRect(MatOfPoint(*quadWork))
        val aspect = rect.width.toDouble() / max(1.0, rect.height.toDouble())
        val rectangularity = (contourAreaWork / max(1.0, rect.width.toDouble() * rect.height)).coerceIn(0.0, 1.0)
        val angScore = rightAngleScore(quadWork)
        val aspectNorm = normalizeAspect(aspect)
        val sizeNorm = (contourAreaWork / frameAreaWork).coerceIn(0.0, 1.0) // размер относительно ROI/WORK
        val borderScore = borderDistanceScore(quadDownFull, dwFull, dhFull) // ⚠️ глобальные границы
        return ScoreBreakdown(rectangularity, angScore, aspectNorm, sizeNorm, borderScore, 0.0)
    }

    private fun combinedScoreFast(b: ScoreBreakdown): Double {
        return 0.36 * b.rectangularity +
                0.24 * b.rightAngles +
                0.18 * b.size +
                0.12 * b.aspect +
                0.10 * b.border
    }

    private fun combinedScore(b: ScoreBreakdown): Double {
        return 0.30 * b.rectangularity +
                0.22 * b.rightAngles +
                0.18 * b.size +
                0.12 * b.aspect +
                0.12 * b.border +
                0.06 * b.edge
    }

    private fun edgeSupportScore(quadWork: Array<Point>, edgesMaskWork: Mat): Double {
        val mask = Mat.zeros(edgesMaskWork.size(), CvType.CV_8UC1)
        val pts = arrayOf(quadWork[0], quadWork[1], quadWork[2], quadWork[3])
        for (i in 0..3) Imgproc.line(mask, pts[i], pts[(i + 1) % 4], Scalar(255.0), 3)
        val inter = Mat()
        Core.bitwise_and(mask, edgesMaskWork, inter)
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

    private data class Line(val a: Double, val b: Double, val c: Double)
    private data class Seg(val x1: Double, val y1: Double, val x2: Double, val y2: Double, val angleDeg: Double) {
        val cx = (x1 + x2) * 0.5
        val cy = (y1 + y2) * 0.5
    }

    private fun findRectByHough(edges: Mat, w: Int, h: Int): Array<Point>? {
        val lines = Mat()
        val minLen = min(w, h) * houghMinLineLenRatio
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 120, minLen, 10.0)
        if (lines.empty()) { lines.release(); return null }

        val segs = ArrayList<Seg>(lines.rows())
        for (i in 0 until lines.rows()) {
            val v = lines.get(i, 0)
            val x1 = v[0]; val y1 = v[1]; val x2 = v[2]; val y2 = v[3]
            val angleDeg = Math.toDegrees(atan2(y2 - y1, x2 - x1))
            segs.add(Seg(x1, y1, x2, y2, angleDeg))
        }
        lines.release()

        val horiz = segs.filter {
            val ang = abs(normalizeAngle(it.angleDeg))
            ang < houghAngleTol || ang > (180.0 - houghAngleTol)
        }.sortedBy { it.cy }

        val vert = segs.filter {
            val ang = abs(normalizeAngle(it.angleDeg))
            ang in (90.0 - houghAngleTol)..(90.0 + houghAngleTol)
        }.sortedBy { it.cx }

        if (horiz.size < 2 || vert.size < 2) return null

        val top = horiz.first(); val bottom = horiz.last()
        val left = vert.first(); val right = vert.last()

        val lTop = lineFromSegment(top)
        val lBottom = lineFromSegment(bottom)
        val lLeft = lineFromSegment(left)
        val lRight = lineFromSegment(right)

        val tl = intersect(lTop, lLeft) ?: return null
        val tr = intersect(lTop, lRight) ?: return null
        val br = intersect(lBottom, lRight) ?: return null
        val bl = intersect(lBottom, lLeft) ?: return null

        val inBounds = fun(p: Point) = p.x >= -w * 0.2 && p.x <= w * 1.2 && p.y >= -h * 0.2 && p.y <= h * 1.2
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

    private fun lineFromSegment(s: Seg): Line {
        val a = s.y1 - s.y2
        val b = s.x2 - s.x1
        val c = s.x1 * s.y2 - s.x2 * s.y1
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

    // ---------- Геометрия / Утилиты ----------

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
        val cands = listOfNotNull(intersect(l01, l12), intersect(l12, l20), intersect(l20, l01))
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
        val targets = doubleArrayOf(1.414213562, 1.294117647) // A4, Letter
        val diff = targets.minOf { kotlin.math.abs(r - it) }
        return (1.0 - (diff / 0.6)).coerceIn(0.0, 1.0)
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

    private fun polygonArea(quad: List<Point>): Double {
        var s = 0.0
        for (i in quad.indices) {
            val a = quad[i]; val b = quad[(i + 1) % quad.size]
            s += a.x * b.y - b.x * a.y
        }
        return abs(s) * 0.5
    }

    private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)
    private fun lerp(a: Point, b: Point, t: Double): Point = Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)

    // ---------- Warp ----------
    private fun warpByQuad(src: Bitmap, quad: Array<Point>): Bitmap? {
        val widthTop = dist(quad[0], quad[1])
        val widthBottom = dist(quad[3], quad[2])
        val heightLeft = dist(quad[0], quad[3])
        val heightRight = dist(quad[1], quad[2])

        val width = max(widthTop, widthBottom).roundToInt().coerceAtLeast(300)
        val height = max(heightLeft, heightRight).roundToInt().coerceAtLeast(300)

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

    // ---------- Постпроцесс ----------
    private fun applyPostproc(bmp: Bitmap, profile: EnhanceProfile): Bitmap = when (profile) {
        EnhanceProfile.NONE -> bmp
        EnhanceProfile.SOFT -> enhanceSoft(bmp)
        EnhanceProfile.BW   -> enhanceBW(bmp)
    }

    private fun enhanceSoft(bmp: Bitmap): Bitmap {
        val rgba = Mat().also { Utils.bitmapToMat(bmp, it) }
        val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        val ycrcb = Mat(); Imgproc.cvtColor(rgb, ycrcb, Imgproc.COLOR_RGB2YCrCb)
        val channels = ArrayList<Mat>(3)
        Core.split(ycrcb, channels)
        val clahe = Imgproc.createCLAHE(1.0, Size(16.0, 16.0))
        clahe.apply(channels[0], channels[0])
        Core.merge(channels, ycrcb)
        Imgproc.cvtColor(ycrcb, rgb, Imgproc.COLOR_YCrCb2RGB)
        Imgproc.cvtColor(rgb, rgba, Imgproc.COLOR_RGB2RGBA)
        val blur = Mat()
        Imgproc.GaussianBlur(rgba, blur, Size(0.0, 0.0), 0.9)
        Core.addWeighted(rgba, 1.03, blur, -0.03, 0.0, rgba)
        blur.release()
        rgba.convertTo(rgba, -1, 1.02, -2.0)
        val out = createBitmap(bmp.width, bmp.height)
        Utils.matToBitmap(rgba, out)
        channels.forEach { it.release() }
        ycrcb.release(); rgb.release(); rgba.release()
        return out
    }

    private fun enhanceBW(bmp: Bitmap): Bitmap {
        val rgba = Mat().also { Utils.bitmapToMat(bmp, it) }
        val gray = Mat(); Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        val bw = Mat()
        Imgproc.adaptiveThreshold(
            gray, bw, 255.0,
            Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY, 25, 10.0
        )
        Imgproc.medianBlur(bw, bw, 3)
        val outRgba = Mat(); Imgproc.cvtColor(bw, outRgba, Imgproc.COLOR_GRAY2RGBA)
        val out = createBitmap(bmp.width, bmp.height)
        Utils.matToBitmap(outRgba, out)
        rgba.release(); gray.release(); bw.release(); outRgba.release()
        return out
    }

    // ---------- Работа с ImageProxy ----------
    @OptIn(ExperimentalGetImage::class)
    private fun yPlaneToGrayMat(image: ImageProxy): Mat? {
        val img = image.image ?: return null
        if (image.format != ImageFormat.YUV_420_888) return null

        val w = image.width
        val h = image.height
        val yPlane = img.planes[0]
        val yBuffer = yPlane.buffer
        val rowStride = yPlane.rowStride
        val pixelStride = yPlane.pixelStride

        val out = Mat(h, w, CvType.CV_8UC1)
        val row = ByteArray(w)
        for (r in 0 until h) {
            var c = 0
            var srcIndex = r * rowStride
            while (c < w) {
                row[c] = yBuffer.get(srcIndex)
                c++; srcIndex += pixelStride
            }
            out.put(r, 0, row)
        }
        return out
    }

    private fun rotateMat(src: Mat, rotationDeg: Int): Mat {
        if (rotationDeg == 0) return src
        val dst = Mat()
        when (rotationDeg) {
            90 -> Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(src, dst, Core.ROTATE_180)
            270 -> Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> return src
        }
        src.release()
        return dst
    }

    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToColorBitmap(image: ImageProxy): Bitmap? {
        val img = image.image ?: return null
        if (image.format != ImageFormat.YUV_420_888) return null

        val nv21 = yuv420888ToNv21(img, image.width, image.height)
        val yuvMat = Mat(image.height + image.height / 2, image.width, CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)
        val rgba = Mat()
        Imgproc.cvtColor(yuvMat, rgba, Imgproc.COLOR_YUV2RGBA_NV21)
        val bmp = createBitmap(image.width, image.height)
        Utils.matToBitmap(rgba, bmp)
        yuvMat.release(); rgba.release()
        return bmp
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
            var col = 0
            while (col < width) {
                val yIndex = row * yRowStride + col * yPixelStride
                out[pos++] = yBuffer.get(yIndex)
                col++
            }
        }

        // VU (NV21)
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            var col = 0
            while (col < chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                out[pos++] = vBuffer.get(vIndex)
                out[pos++] = uBuffer.get(uIndex)
                col++
            }
        }
        return out
    }

    private fun releaseMats(vararg mats: Mat) { mats.forEach { runCatching { it.release() } } }
}
