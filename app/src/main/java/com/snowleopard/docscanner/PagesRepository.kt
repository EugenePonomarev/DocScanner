package com.snowleopard.docscanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import androidx.core.graphics.scale

class PagesRepository(private val context: Context) {

    private var sessionDir: File = newSessionDir()
    private var counter: Int = 0

    fun newSession() {
        sessionDir = newSessionDir()
        counter = 0
    }

    fun getSessionDir(): File = sessionDir

    fun savePage(original: Bitmap, preferPortraitA4: Boolean = true): ScannedPage {
        // Авто-ориентация A4: если аспект близок к A4 (~√2), предпочитаем портрет
        val bmp = if (preferPortraitA4) autoOrientA4(original) else original

        val id = UUID.randomUUID().toString()
        val index = counter++
        val pageFile = File(sessionDir, "page_${index}_$id.jpg")
        val thumbFile = File(sessionDir, "thumb_${index}_$id.jpg")

        // Сохраняем основную страницу (качество 92)
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

    private fun newSessionDir(): File {
        val dir = File(context.cacheDir, "scans/session_${System.currentTimeMillis()}")
        dir.mkdirs()
        return dir
    }

    private fun autoOrientA4(src: Bitmap): Bitmap {
        val w = src.width.toFloat()
        val h = src.height.toFloat()
        if (w <= 0 || h <= 0) return src

        // Нормализуем аспект (>=1.0)
        val r = if (w >= h) w / h else h / w
        val target = 1.4142f // √2
        val close = kotlin.math.abs(r - target) < 0.35f // «похоже на A4/Letter»

        // Если похоже на лист и он лежит горизонтально — переворачиваем в портрет
        return if (close && w > h) rotate(src, 90f) else src
    }

    private fun rotate(src: Bitmap, deg: Float): Bitmap {
        val m = Matrix().apply { postRotate(deg) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }
}
