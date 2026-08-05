package com.snowleopard.docscanner.core.data.repository

import android.content.Context
import android.graphics.*
import android.graphics.pdf.PdfDocument
import android.os.Environment
import androidx.core.graphics.scale
import com.snowleopard.docscanner.core.data.model.PdfOptions
import com.snowleopard.docscanner.core.data.model.ScannedPage
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import kotlin.math.max
import kotlin.math.roundToInt

private const val A4_WIDTH_POINTS = 595
private const val A4_HEIGHT_POINTS = 842
private const val POINTS_PER_INCH = 72f
private const val MILLIMETERS_PER_INCH = 25.4f

class PagesRepository(private val context: Context) {

    private var sessionDir: File = newSessionDir()
    private var counter: Int = 0

    fun newSession() {
        sessionDir = newSessionDir()
        counter = 0
    }

    fun getSessionDir(): File = sessionDir

    fun savePage(original: Bitmap, preferPortrait: Boolean = false): ScannedPage {
        val maxSide = 3000
        var bmp = original
        if (max(bmp.width, bmp.height) > maxSide) {
            val scale = maxSide.toFloat() / max(bmp.width, bmp.height)
            bmp = bmp.scale((bmp.width * scale).toInt(), (bmp.height * scale).toInt())
        }

        val id = UUID.randomUUID().toString()
        val index = counter++
        val pageFile = File(sessionDir, "page_${index}_$id.jpg")
        val thumbFile = File(sessionDir, "thumb_${index}_$id.jpg")

        FileOutputStream(pageFile).use { fos ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 92, fos)
        }

        val thumbW = 240
        val r = thumbW.toFloat() / bmp.width
        val th = (bmp.height * r).toInt().coerceAtLeast(1)
        val thumb = bmp.scale(thumbW, th)
        FileOutputStream(thumbFile).use { fos ->
            thumb.compress(Bitmap.CompressFormat.JPEG, 85, fos)
        }

        return ScannedPage(
            id = id,
            file = pageFile,
            thumbFile = thumbFile,
            width = bmp.width,
            height = bmp.height,
            createdAt = System.currentTimeMillis()
        )
    }

    fun deletePage(page: ScannedPage) {
        runCatching { page.file.delete() }
        runCatching { page.thumbFile.delete() }
    }

    fun clearSession() {
        sessionDir.listFiles()?.forEach { runCatching { it.delete() } }
        runCatching { sessionDir.delete() }
        newSession()
    }

    fun decodeThumb(page: ScannedPage): Bitmap? =
        BitmapFactory.decodeFile(page.thumbFile.absolutePath)

    fun rotatePage(page: ScannedPage, degrees: Int): ScannedPage {
        val src = BitmapFactory.decodeFile(page.file.absolutePath) ?: return page

        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)

        // overwrite page
        FileOutputStream(page.file).use { rotated.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        page.file.setLastModified(System.currentTimeMillis())

        // overwrite thumb
        val thumbW = 240
        val r = thumbW.toFloat() / rotated.width.toFloat()
        val th = (rotated.height * r).roundToInt().coerceAtLeast(1)
        val thumb = rotated.scale(thumbW, th)
        FileOutputStream(page.thumbFile).use { thumb.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        page.thumbFile.setLastModified(System.currentTimeMillis())

        val rotatedWidth = rotated.width
        val rotatedHeight = rotated.height

        // free bitmaps
        if (!src.isRecycled) src.recycle()
        if (!rotated.isRecycled) rotated.recycle()
        if (!thumb.isRecycled) thumb.recycle()

        // return updated dims so StateFlow emits new object
        return page.copy(
            width = rotatedWidth,
            height = rotatedHeight,
        )
    }

    fun buildPdf(pages: List<ScannedPage>, options: PdfOptions): File {
        require(pages.isNotEmpty()) { "pages is empty" }
        require(options.marginMm >= 0) { "marginMm must be non-negative" }

        val outDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: context.getExternalFilesDir(null)
            ?: error("Documents directory is unavailable")

        if (!outDir.exists() && !outDir.mkdirs()) {
            error("Unable to create output directory")
        }

        val margin = mmToPoints(options.marginMm)

        require(margin * 2 < A4_WIDTH_POINTS) {
            "Margin is too large for A4 page"
        }

        val outFile = File(outDir, "Scan_${System.currentTimeMillis()}.pdf")
        val pdf = PdfDocument()

        try {
            pages.forEachIndexed { index, page ->
                val raw = BitmapFactory.decodeFile(page.file.absolutePath)
                    ?: error("Unable to decode page: ${page.file.absolutePath}")

                var content = raw

                try {
                    if (options.autoRotateToPortrait && raw.width > raw.height) {
                        content = rotate(raw, 90f)
                    }

                    val isPortrait = content.height >= content.width
                    val pageWidth = if (isPortrait) {
                        A4_WIDTH_POINTS
                    } else {
                        A4_HEIGHT_POINTS
                    }
                    val pageHeight = if (isPortrait) {
                        A4_HEIGHT_POINTS
                    } else {
                        A4_WIDTH_POINTS
                    }

                    val contentWidth = pageWidth - margin * 2
                    val contentHeight = pageHeight - margin * 2

                    val bitmapScale = minOf(
                        contentWidth.toFloat() / content.width,
                        contentHeight.toFloat() / content.height,
                    )

                    val drawWidth = (content.width * bitmapScale).roundToInt()
                    val drawHeight = (content.height * bitmapScale).roundToInt()

                    val drawLeft = margin + (contentWidth - drawWidth) / 2f
                    val drawTop = margin + (contentHeight - drawHeight) / 2f

                    val pageInfo = PdfDocument.PageInfo.Builder(
                        pageWidth,
                        pageHeight,
                        index + 1,
                    ).create()

                    val pdfPage = pdf.startPage(pageInfo)

                    pdfPage.canvas.drawColor(Color.WHITE)
                    pdfPage.canvas.drawBitmap(
                        content,
                        null,
                        RectF(
                            drawLeft,
                            drawTop,
                            drawLeft + drawWidth,
                            drawTop + drawHeight,
                        ),
                        Paint(Paint.FILTER_BITMAP_FLAG),
                    )

                    pdf.finishPage(pdfPage)
                } finally {
                    if (content !== raw && !content.isRecycled) {
                        content.recycle()
                    }

                    if (!raw.isRecycled) {
                        raw.recycle()
                    }
                }
            }

            FileOutputStream(outFile).use { output ->
                pdf.writeTo(output)
            }

            return outFile
        } finally {
            pdf.close()
        }
    }

    // --- helpers ---
    private fun newSessionDir(): File {
        val dir = File(context.cacheDir, "scans/session_${System.currentTimeMillis()}")
        dir.mkdirs()
        return dir
    }

    private fun rotate(src: Bitmap, deg: Float): Bitmap {
        val m = Matrix().apply { postRotate(deg) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun mmToPx(mm: Int, dpi: Int): Int = ((dpi.toFloat() * mm) / 25.4f).roundToInt()

    private fun mmToPoints(mm: Int): Int = (mm * POINTS_PER_INCH / MILLIMETERS_PER_INCH).roundToInt()
}
