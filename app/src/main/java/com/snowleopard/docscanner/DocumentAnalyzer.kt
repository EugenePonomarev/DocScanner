package com.snowleopard.docscanner

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.media.Image
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

private const val TAG = "DocAnalyzer"

class DocumentAnalyzer(
    private val onFrameSize: (w: Int, h: Int) -> Unit,
    private val onDebug: (bmp: Bitmap?, status: String) -> Unit,
    private val onResult: (polygon: List<Pair<Float, Float>>, cropped: Bitmap?, thumbnail: Bitmap?) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        var status = "analyze..."
        try {
            var src = imageProxyToBitmap(image) ?: run {
                status = "no bitmap"
                onDebug(null, status)
                image.close(); return
            }

            // Учитываем поворот кадра
            val rotation = image.imageInfo.rotationDegrees
            if (rotation != 0) {
                val m = android.graphics.Matrix().apply { postRotate(rotation.toFloat()) }
                src = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
            }

            // Сообщаем размеры для корректного оверлея
            onFrameSize(src.width, src.height)

            val (poly, cropped, dbg, dbgStatus) = detectAndCropDocument(src)
            status = dbgStatus
            onDebug(dbg, status)

            val thumb = cropped?.let { makeThumbnail(it, 200) }
            onResult(poly, cropped, thumb)
        } catch (t: Throwable) {
            status = "error: ${t.message}"
            Log.e(TAG, "analyze error", t)
            onDebug(null, status)
            onResult(emptyList(), null, null)
        } finally {
            image.close()
        }
    }

    // -------- Детекция документа --------

    private data class DetectOut(
        val polygon: List<Pair<Float, Float>>,
        val cropped: Bitmap?,
        val debugBmp: Bitmap?,
        val status: String
    )

    private fun detectAndCropDocument(src: Bitmap): DetectOut {
        val rgba = Mat()
        Utils.bitmapToMat(src, rgba)

        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)

        val gray = Mat()
        Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)

        // Контраст + сглаживание
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        // Адаптивный порог (белый лист на сером фоне)
        val bin = Mat()
        Imgproc.adaptiveThreshold(
            gray, bin, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, 21, 5.0
        )
        Core.bitwise_not(bin, bin)

        // Закрываем разрывы на краях
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        repeat(2) { Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, kernel) }

        // Отладочная картинка того, что «видит» детектор
        val dbgBmp = Bitmap.createBitmap(bin.cols(), bin.rows(), Bitmap.Config.ARGB_8888).also {
            val vis = Mat()
            Imgproc.cvtColor(bin, vis, Imgproc.COLOR_GRAY2RGBA)
            Utils.matToBitmap(vis, it)
            vis.release()
        }

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(bin, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var bestQuad: MatOfPoint2f? = null
        var bestArea = 0.0
        val areaMin = (src.width * src.height) * 0.08 // отсечь мелочь

        for (c in contours) {
            val peri = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*c.toArray()), approx, 0.02 * peri, true)
            if (approx.total() == 4L && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                val area = Imgproc.contourArea(MatOfPoint(*approx.toArray()))
                if (area > bestArea && area > areaMin) {
                    bestArea = area
                    bestQuad = approx
                }
            }
        }

        if (bestQuad == null) {
            return DetectOut(emptyList(), null, dbgBmp, "no quad (contours=${contours.size})")
        }

        val sorted = orderQuad(bestQuad!!.toArray().toMutableList())

        val widthA = dist(sorted[2], sorted[3])
        val widthB = dist(sorted[1], sorted[0])
        val maxW = max(widthA, widthB).roundToInt().coerceAtLeast(200)

        val heightA = dist(sorted[1], sorted[2])
        val heightB = dist(sorted[0], sorted[3])
        val maxH = max(heightA, heightB).roundToInt().coerceAtLeast(200)

        val srcPts = MatOfPoint2f(sorted[0], sorted[1], sorted[2], sorted[3])
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(maxW - 1.0, 0.0),
            Point(maxW - 1.0, maxH - 1.0),
            Point(0.0, maxH - 1.0)
        )

        val M = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val warped = Mat()
        Imgproc.warpPerspective(rgb, warped, M, Size(maxW.toDouble(), maxH.toDouble()))

        val out = Bitmap.createBitmap(maxW, maxH, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(warped, out)

        val polygon = sorted.map { it.x.toFloat() to it.y.toFloat() }
        return DetectOut(polygon, out, dbgBmp, "quad ok ${maxW}x${maxH}")
    }

    private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)

    private fun orderQuad(pts: MutableList<Point>): List<Point> {
        val tl = pts.minBy { it.x + it.y }
        val br = pts.maxBy { it.x + it.y }
        val sortedDiff = pts.sortedBy { it.x - it.y }
        val bl = sortedDiff.first()
        val tr = sortedDiff.last()
        return listOf(tl, tr, br, bl)
    }

    private fun makeThumbnail(b: Bitmap, w: Int): Bitmap {
        val r = w.toFloat() / b.width
        val h = (b.height * r).roundToInt()
        return Bitmap.createScaledBitmap(b, w, h, true)
    }

    // -------- Надёжная конвертация YUV_420_888 → NV21 → Bitmap --------

    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val yuv = image.image ?: return null
        if (image.format != ImageFormat.YUV_420_888) return null

        val nv21 = yuv420888ToNv21(yuv, image.width, image.height)

        val yuvImage = android.graphics.YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val bytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun yuv420888ToNv21(image: Image, width: Int, height: Int): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1] // Cb
        val vPlane = image.planes[2] // Cr

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride

        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride

        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        // NV21 = Y плотно, затем interleaved VU
        val out = ByteArray(width * height + 2 * (width / 2) * (height / 2))
        var pos = 0

        // --- Y: по абсолютным индексам, без bulk get() ---
        var row = 0
        while (row < height) {
            var col = 0
            while (col < width) {
                val yIndex = row * yRowStride + col * yPixelStride
                out[pos++] = yBuffer.get(yIndex)
                col++
            }
            row++
        }

        // --- VU: по абсолютным индексам (NV21 = V, затем U) ---
        val chromaHeight = height / 2
        val chromaWidth = width / 2

        row = 0
        while (row < chromaHeight) {
            var col = 0
            while (col < chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                out[pos++] = vBuffer.get(vIndex) // V
                out[pos++] = uBuffer.get(uIndex) // U
                col++
            }
            row++
        }

        return out
    }
}