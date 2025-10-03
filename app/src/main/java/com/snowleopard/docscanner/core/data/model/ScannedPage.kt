package com.snowleopard.docscanner.core.data.model

import java.io.File

data class ScannedPage(
    val id: String,
    val file: File,
    val thumbFile: File,
    val width: Int,
    val height: Int,
    val createdAt: Long
)