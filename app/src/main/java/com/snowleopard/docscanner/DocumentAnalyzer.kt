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

class DocumentAnalyzer(
    private val onFrameSize: (w: Int, h: Int) -> Unit,
    private val onDebug: (bmp: Bitmap?, status: String) -> Unit,
    private val onResult: (polygon: List<Pair<Float, Float>>, cropped: Bitmap?, thumbnail: Bitmap?) -> Unit
) : ImageAnalysis.Analyzer {

    // --- Темпоральное сглаживание (state) ---
    private var prevQuad: Array<Point>? = null
    private var prevScore: Double = 0.0

    // Параметры стабилизации
    private val alpha = 0.25            // доля “нового” кадра
    private val minScoreToAccept = 0.35 // ниже — считаем кандидат слабым
    private val keepPrevIfBetterDelta = 0.08 // гистерезис по score

    override fun analyze(image: ImageProxy) {
        var status = "analyze..."
        try {
            var src = imageProxyToBitmap(image) ?: run {
                status = "no bitmap"
                onDebug(null, status)
                image.close(); return
            }

            // Учитываем поворот
            val rotation = image.imageInfo.rotationDegrees
            if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                src = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
            }

            onFrameSize(src.width, src.height)

            val out = detectAndCropDocument(src)
            status = out.status
            onDebug(out.debugBmp, status)

            // Отдаём полигон (для оверлея) и кроп (для предпросмотра)
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

    // -------------------- Детекция + стабилизация --------------------

    private data class DetectOut(
        val polygon: List<Point>,    // в координатах исходного bitmap
        val cropped: Bitmap?,        // выровненный документ
        val debugBmp: Bitmap?,       // маска/рёбра для отладки
        val status: String
    )

    private fun detectAndCropDocument(src: Bitmap): DetectOut {
        // 1) Даунскейлим для устойчивости/скорости (до ширины ≈ 960)
        val maxW = 960
        val scale = if (src.width > maxW) maxW.toDouble() / src.width else 1.0
        val down = if (scale < 1.0) src.scale(
            (src.width * scale).roundToInt(),
            (src.height * scale).roundToInt()
        ) else src

        // 2) Предобработка
        val rgba = Mat().also { Utils.bitmapToMat(down, it) }
        val bgr = Mat(); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val gray = Mat(); Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)

        // Контраст + сглаживание (сохраняем края)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        // Приводим к 8-битному формату и фильтруем
        val tmp = Mat()
        gray.convertTo(tmp, CvType.CV_8UC1)
        Imgproc.bilateralFilter(tmp, gray, 7, 50.0, 50.0)
        tmp.release()

        // Адаптивный порог даёт “массу”, Canny даёт “края” — объединим
        val bin = Mat()
        Imgproc.adaptiveThreshold(
            gray, bin, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, /*blockSize*/21, /*C*/5.0
        )
        Core.bitwise_not(bin, bin)

        val edges = Mat()
        // Пороги Canny по медиане — автонастройка под сцену
        val med = median(gray)
        val lower = max(0.0, 0.66 * med)
        val upper = min(255.0, 1.33 * med)
        Imgproc.Canny(gray, edges, lower, upper)

        // Объединяем информацию
        val comb = Mat()
        Core.bitwise_or(bin, edges, comb)

        // Морфология: закрыть разрывы, убрать “дырочки”
        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val k5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_CLOSE, k3)
        Imgproc.morphologyEx(comb, comb, Imgproc.MORPH_OPEN, k3)
        Imgproc.dilate(comb, comb, k5)

        // Фильтруем мелкий мусор по площади связных контуров
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            comb,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val areaMin = (down.width * down.height) * 0.06 // отсечь мелочь/коврики
        val frameArea = (down.width * down.height).toDouble()

        var bestQuadScaled: Array<Point>? = null
        var bestScore = -1.0

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < areaMin) continue

            // Аппроксим до многоугольника
            val peri = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*c.toArray()), approx, 0.02 * peri, true)

            // Получаем 4-точечный кандидат
            val quad = when (approx.total().toInt()) {
                4 -> approx.toArray()
                else -> minAreaRectToQuad(c) // фолбэк
            }

            // Нормируем и оцениваем
            val quadConvex = Imgproc.isContourConvex(MatOfPoint(*quad))
            if (!quadConvex) continue

            val rect = Imgproc.boundingRect(MatOfPoint(*quad))
            val rectArea = rect.width.toDouble() * rect.height
            val rectangularity = (area / max(1.0, rectArea)).coerceIn(0.0, 1.0)

            val aspect = rect.width.toDouble() / max(1.0, rect.height.toDouble())
            val aspectNorm = normalizeAspect(aspect) // близость к 1.414 или 1/1.414

            val rightAngles = rightAngleScore(quad)

            val sizeNorm = (area / frameArea).coerceIn(0.0, 1.0)

            // Комбинированный скор
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

        // Если ничего достойного — вернём пусто
        if (bestQuadScaled == null) {
            val dbgBmp = matToBitmap(comb)
            return DetectOut(emptyList(), null, dbgBmp, "no quad (contours=${contours.size})")
        }

        // Снимем масштаб обратно в координаты исходного bitmap
        val toSrc: (Point) -> Point = { p ->
            if (scale < 1.0) Point(p.x / scale, p.y / scale) else Point(p.x, p.y)
        }
        val currentQuad = bestQuadScaled.map(toSrc).toTypedArray()

        // --- Темпоральная стабилизация ---
        val usePrev = prevQuad != null &&
                (bestScore + keepPrevIfBetterDelta) < prevScore // старый был явно устойчивее

        val smoothed = if (prevQuad != null && !usePrev) {
            Array(4) { i -> lerp(prevQuad!![i], currentQuad[i], alpha) }
        } else {
            prevQuad ?: currentQuad
        }

        prevQuad = smoothed
        prevScore = if (usePrev) prevScore else max(prevScore * (1 - alpha), bestScore)

        // Если кандидат слабый — показываем только сглаженный прошлый (не обновляем crop)
        val quadForWarp =
            if (bestScore < minScoreToAccept && prevQuad != null) prevQuad!! else currentQuad

        val warped = warpByQuad(src, quadForWarp)

        val dbgBmp = matToBitmap(comb)
        val status = "score=%.2f rect=%.2f ang=%.2f asp=%.2f size=%.2f".format(
            bestScore,
            // расклады для наглядности
            /*rectangularity*/ 0.0, /*rightAngles*/ 0.0, /*aspect*/ 0.0, /*size*/ 0.0
        )
        return DetectOut(smoothed.toList(), warped, dbgBmp, status)
    }

    // -------------------- Помощники оценки/преобразований --------------------

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
        // Оцениваем 4 угла на близость к 90°
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
        // 1.0 = идеально 90°, допускаем ±20°
        val scores = angs.map { 1.0 - (abs(it - 90.0) / 20.0).coerceIn(0.0, 1.0) }
        return scores.average()
    }

    private fun normalizeAspect(ratio: Double): Double {
        val r = if (ratio < 1.0) 1.0 / ratio else ratio
        val target = 1.4142 // A4
        val diff = abs(r - target)
        // 1.0 при точном A4, 0.0 когда r далеко (> 1.414 + 0.8)
        return (1.0 - (diff / 0.8)).coerceIn(0.0, 1.0)
    }

    private fun median(gray: Mat): Double {
        // быстрая медиана по гистограмме
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
        // целевой прямоугольник — по ориентации квада
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

    // -------------------- Безопасный YUV_420_888 → NV21 → Bitmap --------------------

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