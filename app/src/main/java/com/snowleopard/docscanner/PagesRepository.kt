package com.snowleopard.docscanner

import android.content.Context
import android.graphics.*
import android.os.Environment
import androidx.core.graphics.scale
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import kotlin.math.roundToInt

data class PdfOptions(
    val dpi: Int = 200,                 // качество/вес
    val marginMm: Int = 6,              // поля
    val jpegQuality: Int = 90,          // качество рендера
    val autoRotateToPortrait: Boolean = false, // ❌ по умолчанию НЕ крутим контент
)

class PagesRepository(private val context: Context) {

    private var sessionDir: File = newSessionDir()
    private var counter: Int = 0

    fun newSession() {
        sessionDir = newSessionDir()
        counter = 0
    }

    fun getSessionDir(): File = sessionDir

    /**
     * Сохраняем страницу как есть. НИКАКОЙ авто-ориентации.
     */
    fun savePage(original: Bitmap, preferPortrait: Boolean = false): ScannedPage {
        val bmp = original

        val id = UUID.randomUUID().toString()
        val index = counter++
        val pageFile = File(sessionDir, "page_${index}_$id.jpg")
        val thumbFile = File(sessionDir, "thumb_${index}_$id.jpg")

        FileOutputStream(pageFile).use { fos ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 92, fos)
        }

        // Превью 240px по ширине
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

    // Поворот файла страницы (перезаписываем и thumb)
    fun rotatePage(page: ScannedPage, degrees: Int): ScannedPage {
        val src = BitmapFactory.decodeFile(page.file.absolutePath) ?: return page
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        val rot = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)

        FileOutputStream(page.file).use { rot.compress(Bitmap.CompressFormat.JPEG, 92, it) }

        val thumbW = 240
        val r = thumbW.toFloat() / rot.width
        val th = (rot.height * r).roundToInt().coerceAtLeast(1)
        val thumb = rot.scale(thumbW, th)
        FileOutputStream(page.thumbFile).use { thumb.compress(Bitmap.CompressFormat.JPEG, 85, it) }

        return page.copy(width = rot.width, height = rot.height)
    }

    /**
     * PDF: сохраняем ориентацию каждой страницы КАК ЕСТЬ.
     * Если картинка альбомная — создаём альбомную PDF-страницу.
     * Никаких автоповоротов (кроме явного включения флагом).
     */
    fun buildPdf(pages: List<ScannedPage>, options: PdfOptions): File {
        require(pages.isNotEmpty()) { "pages is empty" }

        val outDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: context.getExternalFilesDir(null)!!
        if (!outDir.exists()) outDir.mkdirs()

        val outFile = File(outDir, "Scan_${System.currentTimeMillis()}.pdf")
        val pdf = android.graphics.pdf.PdfDocument()

        // A4 базовый размер (портрет)
        val a4W = (8.27f * options.dpi).roundToInt()
        val a4H = (11.69f * options.dpi).roundToInt()
        val margin = mmToPx(options.marginMm, options.dpi)

        pages.forEachIndexed { idx, p ->
            val raw = BitmapFactory.decodeFile(p.file.absolutePath) ?: return@forEachIndexed

            // autoRotateToPortrait=false -> оставляем как есть,
            // true -> поворачиваем в портрет при необходимости
            val bmp = if (options.autoRotateToPortrait && raw.width > raw.height) {
                rotate(raw, 90f)
            } else raw

            val isPortrait = bmp.height >= bmp.width
            val pageW = if (isPortrait) a4W else a4H
            val pageH = if (isPortrait) a4H else a4W

            val pageInfo = android.graphics.pdf.PdfDocument.PageInfo.Builder(pageW, pageH, idx + 1).create()
            val page = pdf.startPage(pageInfo)
            val canvas = page.canvas

            // Белый фон
            canvas.drawColor(Color.WHITE)

            // Контентная область
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
