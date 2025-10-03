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

        // free bitmaps
        if (!src.isRecycled) src.recycle()
        if (!rotated.isRecycled) rotated.recycle()
        if (!thumb.isRecycled) thumb.recycle()

        // return updated dims so StateFlow emits new object
        return page.copy(width = rotated.width, height = rotated.height)
    }

    fun buildPdf(pages: List<ScannedPage>, options: PdfOptions): File {
        require(pages.isNotEmpty()) { "pages is empty" }

        val outDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: context.getExternalFilesDir(null)!!
        if (!outDir.exists()) outDir.mkdirs()

        val outFile = File(outDir, "Scan_${System.currentTimeMillis()}.pdf")
        val pdf = PdfDocument()

        val a4W = (8.27f * options.dpi).roundToInt()
        val a4H = (11.69f * options.dpi).roundToInt()
        val margin = mmToPx(options.marginMm, options.dpi)

        pages.forEachIndexed { idx, p ->
            val raw = BitmapFactory.decodeFile(p.file.absolutePath) ?: return@forEachIndexed

            val bmp = if (options.autoRotateToPortrait && raw.width > raw.height) {
                rotate(raw, 90f)
            } else raw

            val isPortrait = bmp.height >= bmp.width
            val pageW = if (isPortrait) a4W else a4H
            val pageH = if (isPortrait) a4H else a4W

            val pageInfo = PdfDocument.PageInfo.Builder(pageW, pageH, idx + 1).create()
            val page = pdf.startPage(pageInfo)
            val canvas = page.canvas

            canvas.drawColor(Color.WHITE)

            val cx = margin
            val cy = margin
            val cw = pageW - margin * 2
            val ch = pageH - margin * 2

            // Fit center
            val br = minOf(cw.toFloat() / bmp.width, ch.toFloat() / bmp.height)
            val dw = (bmp.width * br).roundToInt()
            val dh = (bmp.height * br).roundToInt()
            val dx = cx + (cw - dw) / 2
            val dy = cy + (ch - dh) / 2

            val scaled = if (dw != bmp.width || dh != bmp.height)
                bmp.scale(dw, dh) else bmp

            val paint = Paint(Paint.FILTER_BITMAP_FLAG)
            canvas.drawBitmap(scaled, dx.toFloat(), dy.toFloat(), paint)

            pdf.finishPage(page)

            if (scaled !== bmp) scaled.recycle()
            if (bmp !== raw) bmp.recycle()
        }

        FileOutputStream(outFile).use { pdf.writeTo(it) }
        pdf.close()
        return outFile
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
}
