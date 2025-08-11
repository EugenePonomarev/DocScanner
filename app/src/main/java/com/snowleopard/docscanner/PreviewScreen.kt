package com.snowleopard.docscanner

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.pdf.PdfDocument
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.TransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import java.io.File
import java.io.FileOutputStream

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PreviewScreen(
    viewModel: ScanViewModel,
    onBack: () -> Unit
) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val bmp by viewModel.finalDoc.collectAsState()

    if (bmp == null) {
        LaunchedEffect(Unit) { onBack() }
        return
    }

    var scale by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var offsetY by remember { mutableFloatStateOf(0f) }
    val state = remember {
        TransformableState { zoomChange, panChange, _ ->
            scale = (scale * zoomChange).coerceIn(0.5f, 6f)
            offsetX += panChange.x
            offsetY += panChange.y
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Предпросмотр") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = null)
                    }
                }
            )
        },
        bottomBar = {
            Row(
                Modifier.fillMaxWidth().padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                OutlinedButton(onClick = {
                    scope.launch {
                        val text = runOcr(bmp!!)
                        val clip = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        clip.setPrimaryClip(ClipData.newPlainText("doc", text))
                    }
                }) { Text("Копировать текст") }

                Row {
                    Button(onClick = { saveAsPdf(ctx, bmp!!) }) { Text("Сохранить PDF") }
                    Spacer(Modifier.width(8.dp))
                    Button(onClick = { saveAsJpg(ctx, bmp!!) }) { Text("Сохранить JPG") }
                }
            }
        }
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding)) {
            Image(
                bitmap = bmp!!.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier
                    .fillMaxSize()
                    .transformable(state)
                    .graphicsLayer(
                        scaleX = scale, scaleY = scale,
                        translationX = offsetX, translationY = offsetY
                    )
            )
        }
    }
}

private suspend fun runOcr(bitmap: Bitmap): String {
    val image = InputImage.fromBitmap(bitmap, 0)
    val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    val result = recognizer.process(image).await()
    return result.text
}

private fun saveAsPdf(ctx: Context, bitmap: Bitmap): File {
    val pdf = PdfDocument()
    val pageInfo = PdfDocument.PageInfo.Builder(bitmap.width, bitmap.height, 1).create()
    val page = pdf.startPage(pageInfo)
    page.canvas.drawBitmap(bitmap, 0f, 0f, null)
    pdf.finishPage(page)

    val out = File(ctx.getExternalFilesDir(null), "scan_${System.currentTimeMillis()}.pdf")
    FileOutputStream(out).use { pdf.writeTo(it) }
    pdf.close()
    return out
}

private fun saveAsJpg(ctx: Context, bitmap: Bitmap): File {
    val out = File(ctx.getExternalFilesDir(null), "scan_${System.currentTimeMillis()}.jpg")
    FileOutputStream(out).use { fos ->
        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos)
    }
    return out
}

