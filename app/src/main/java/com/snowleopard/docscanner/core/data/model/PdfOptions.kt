package com.snowleopard.docscanner.core.data.model

/**
 * PDF rendering options.
 *
 * @param dpi                     Target page DPI (controls output resolution & file size).
 * @param marginMm                Page margins in millimeters (on all sides).
 * @param jpegQuality             Not applied directly by [android.graphics.pdf.PdfDocument].
 *                                Kept for future use or alternative PDF engines; file size is
 *                                primarily controlled via [dpi] and scaling.
 * @param autoRotateToPortrait    If true, landscape bitmaps are rotated to portrait before drawing.
 *                                This only affects content orientation, not the final page size.
 */
data class PdfOptions(
    val dpi: Int = 200,
    val marginMm: Int = 6,
    val jpegQuality: Int = 90,
    val autoRotateToPortrait: Boolean = false,
)