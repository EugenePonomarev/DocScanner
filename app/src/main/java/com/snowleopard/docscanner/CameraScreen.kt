package com.snowleopard.docscanner

import androidx.compose.ui.unit.dp
import com.google.accompanist.permissions.isGranted
import android.Manifest
import android.view.ViewGroup
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberPermissionState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlin.math.max
import kotlin.math.min

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraScreen(
    viewModel: ScanViewModel,
    onSave: () -> Unit
) {
    val cameraPermission = rememberPermissionState(Manifest.permission.CAMERA)
    val context = LocalContext.current

    LaunchedEffect(Unit) {
        if (!cameraPermission.status.isGranted) cameraPermission.launchPermissionRequest()
    }

    if (!cameraPermission.status.isGranted) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Нужно разрешение камеры")
        }
        return
    }

    val polygonRaw by viewModel.livePreviewPolygon.collectAsState()
    val frameSize by viewModel.frameSize.collectAsState()
    val thumb by viewModel.liveThumbnail.collectAsState()
    val debug by viewModel.debugFrame.collectAsState()
    val status by viewModel.status.collectAsState()

    var viewW by remember { mutableStateOf(0) }
    var viewH by remember { mutableStateOf(0) }

    Box(
        Modifier
            .fillMaxSize()
            .onSizeChanged { viewW = it.width; viewH = it.height }
    ) {
        var previewView by remember { mutableStateOf<PreviewView?>(null) }

        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FIT_CENTER // center-crop
                    previewView = this
                }
            },
            update = { pv ->
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val rotation = pv.display.rotation
                    val resolutionSelector = ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .build()

                    val preview = Preview.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .setTargetRotation(rotation)
                        .build().also {
                            it.surfaceProvider = pv.surfaceProvider
                        }

                    val analyzer = ImageAnalysis.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .setTargetRotation(rotation)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build().also { analysis ->
                            analysis.setAnalyzer(
                                Dispatchers.Default.asExecutor(),
                                DocumentAnalyzer(
                                    onFrameSize = { w, h -> viewModel.setFrameSize(w, h) },
                                    onDebug = { bmp, text ->
                                        viewModel.setDebugFrame(bmp)
                                        viewModel.setStatus(text)
                                    },
                                    onResult = { poly, cropped, thumbnail ->
                                        viewModel.updateOverlay(poly, thumbnail)
                                        if (cropped != null) viewModel.setFinalDoc(cropped)
                                    }
                                )
                            )
                        }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            (context as androidx.activity.ComponentActivity),
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            analyzer
                        )
                    } catch (_: Exception) {}
                }, ContextCompat.getMainExecutor(context))
            }
        )

        // Статус сверху
        if (status.isNotEmpty()) {
            Surface(
                color = Color(0x88000000),
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 8.dp)
            ) {
                Text(
                    text = status,
                    color = Color.White,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                )
            }
        }

        // Оверлей: маппим координаты кадра в координаты PreviewView (center-crop)
        Canvas(Modifier.fillMaxSize()) {
            val src = frameSize
            val poly = polygonRaw
            if (src != null && poly.size >= 4 && viewW > 0 && viewH > 0) {
                val (srcW, srcH) = src
                val scale = min(viewW.toFloat() / srcW, viewH.toFloat() / srcH)
                val dx = (viewW - srcW * scale) / 2f
                val dy = (viewH - srcH * scale) / 2f

                val pts = poly.map { (x, y) -> Offset(x * scale + dx, y * scale + dy) }
                for (i in pts.indices) {
                    val a = pts[i]
                    val b = pts[(i + 1) % pts.size]
                    drawLine(Color(0xFF00E676), a, b, strokeWidth = 6f)
                }
            }
        }

        // Нижняя панель: отладочная миниатюра/превью + Сохранить
        Row(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.35f))
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            val mini = debug ?: thumb
            if (mini != null) {
                Image(
                    bitmap = mini.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.size(88.dp)
                )
            } else {
                Spacer(Modifier.size(88.dp))
            }
            Button(
                onClick = { onSave() },
                enabled = viewModel.finalDoc.collectAsState().value != null
            ) { Text("Сохранить") }
        }
    }
}