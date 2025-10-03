package com.snowleopard.docscanner

import android.Manifest
import android.graphics.BitmapFactory
import android.util.Size
import android.view.Surface
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import kotlinx.coroutines.delay
import org.koin.androidx.compose.koinViewModel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraScreen(
    viewModel: ScanViewModel = koinViewModel(),
    onOpenStack: () -> Unit,
) {
    val cameraPermission = rememberPermissionState(Manifest.permission.CAMERA)
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val density = LocalDensity.current

    LaunchedEffect(Unit) {
        if (!cameraPermission.status.isGranted) cameraPermission.launchPermissionRequest()
    }
    if (!cameraPermission.status.isGranted) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Требуется разрешение камеры")
        }
        return
    }

    // VM state
    val polygonRaw by viewModel.livePreviewPolygon.collectAsState()
    val frameSize by viewModel.frameSize.collectAsState()
    val thumb by viewModel.liveThumbnail.collectAsState()
    val debug by viewModel.debugFrame.collectAsState()
    val status by viewModel.status.collectAsState()
    val pages by viewModel.pages.collectAsState()
    val captureInProgress by viewModel.captureInProgress.collectAsState()
    val lastCapturedAt by viewModel.lastCapturedAt.collectAsState()
    val wavePolygon by viewModel.wavePolygon.collectAsState()
    val lockProgress by viewModel.lockProgress.collectAsState()
    val confirmAt by viewModel.confirmAt.collectAsState()

    var viewW by remember { mutableStateOf(0) }
    var viewH by remember { mutableStateOf(0) }

    // PreviewView + executor
    val previewView = remember {
        PreviewView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
    }
    val cameraExecutor: ExecutorService = remember { Executors.newSingleThreadExecutor() }

    // Bind once
    DisposableEffect(lifecycleOwner) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val cameraProvider = cameraProviderFuture.get()

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        val rotation = previewView.display?.rotation ?: Surface.ROTATION_0

        val previewSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER)
            )
            .build()

        val preview = Preview.Builder()
            .setResolutionSelector(previewSelector)
            .setTargetRotation(rotation)
            .build().also { it.surfaceProvider = previewView.surfaceProvider }

        val analysisSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
            .build()

        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(analysisSelector)
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setImageQueueDepth(2)
            // 🔑 Гарантируем формат YUV 420, чтобы цветовая конвертация не падала:
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()

        val analyzer = DocumentAnalyzerTwo(
            onFrameSize = { w, h -> viewModel.setFrameSize(w, h) },
            onDebug = { bmp, text ->
                viewModel.setDebugFrame(bmp)
                viewModel.setStatus(text)
            },
            onResult = { poly, cropped, thumbnail ->
                viewModel.onAnalyzerResult(poly, cropped, thumbnail)
            }
        )
        analysis.setAnalyzer(cameraExecutor, analyzer)

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner, cameraSelector, preview, analysis
            )
        } catch (_: Exception) {}

        onDispose {
            runCatching { cameraProvider.unbindAll() }
            runCatching { cameraExecutor.shutdown() }
        }
    }

    // ===== UI — твой, без потерь =====

    var showAdded by remember(lastCapturedAt) { mutableStateOf(lastCapturedAt != 0L) }
    LaunchedEffect(lastCapturedAt) {
        if (lastCapturedAt == 0L) return@LaunchedEffect
        showAdded = true
        delay(1200)
        showAdded = false
    }

    val waveProgress = remember { Animatable(1f) }
    LaunchedEffect(lastCapturedAt) {
        if (lastCapturedAt == 0L || wavePolygon.isEmpty()) return@LaunchedEffect
        waveProgress.snapTo(0f)
        waveProgress.animateTo(1f, tween(420, easing = LinearEasing))
    }

    var animBmp by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
    val animProgress = remember { Animatable(1f) }
    LaunchedEffect(lastCapturedAt, pages.size) {
        val page = pages.lastOrNull()
        if (lastCapturedAt == 0L || page == null) return@LaunchedEffect
        animBmp = BitmapFactory.decodeFile(page.thumbFile.absolutePath)
        if (animBmp != null) {
            animProgress.snapTo(0f)
            animProgress.animateTo(1f, tween(600, easing = LinearEasing))
            animBmp = null
        }
    }

    val confirmProgress = remember { Animatable(1f) }
    LaunchedEffect(confirmAt) {
        if (confirmAt == 0L) return@LaunchedEffect
        confirmProgress.snapTo(0f)
        confirmProgress.animateTo(1f, tween(420, easing = LinearEasing))
    }

    Box(
        Modifier
            .fillMaxSize()
            .onSizeChanged { viewW = it.width; viewH = it.height }
    ) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { previewView })

        Column(
            Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(top = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (status.isNotEmpty()) {
                Surface(color = Color(0x88000000), shape = MaterialTheme.shapes.small) {
                    Text(status, color = Color.White, modifier = Modifier.padding(12.dp, 6.dp))
                }
            }
            if (lockProgress > 0f && confirmAt == 0L) {
                Spacer(Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { lockProgress },
                    modifier = Modifier.width(200.dp).height(6.dp),
                    color = Color(0xFF00E676),
                    trackColor = Color.White.copy(alpha = 0.25f)
                )
                Spacer(Modifier.height(2.dp))
                Text("Фокусируемся на документе…", color = Color.White)
            }
            if (pages.isNotEmpty() && lastCapturedAt != 0L) {
                Spacer(Modifier.height(6.dp))
                AssistChip(onClick = {}, label = { Text("Добавлено: ${pages.size}") })
            }
        }

        Canvas(Modifier.fillMaxSize()) {
            val src = frameSize
            val poly = polygonRaw
            if (src != null && poly.size >= 4 && viewW > 0 && viewH > 0) {
                val (srcW, srcH) = src
                val scale = min(viewW.toFloat() / srcW, viewH.toFloat() / srcH)
                val dx = (viewW - srcW * scale) / 2f
                val dy = (viewH - srcH * scale) / 2f

                fun mapPts(p: List<Pair<Float, Float>>) =
                    p.map { (x, y) -> Offset(x * scale + dx, y * scale + dy) }
                val pts = mapPts(poly)

                val base = Color(0xFFFFEB3B)
                val target = Color(0xFF00E676)
                fun lerpColor(a: Color, b: Color, t: Float) =
                    Color(a.red + (b.red - a.red) * t, a.green + (b.green - a.green) * t, a.blue + (b.blue - a.blue) * t)
                val polyColor = if (confirmAt > 0L) target else lerpColor(base, target, lockProgress)

                for (i in pts.indices) {
                    val a = pts[i]; val b = pts[(i + 1) % pts.size]
                    drawLine(polyColor, a, b, strokeWidth = 6f)
                }

                if (wavePolygon.isNotEmpty() && waveProgress.value < 1f) {
                    val wPts = mapPts(wavePolygon)
                    val path = Path().apply {
                        moveTo(wPts[0].x, wPts[0].y)
                        for (i in 1 until wPts.size) lineTo(wPts[i].x, wPts[i].y)
                        close()
                    }
                    val p = waveProgress.value
                    val alpha1 = (1f - p).coerceIn(0f, 1f)
                    val alpha2 = (0.6f * (1f - p)).coerceIn(0f, 1f)
                    drawPath(path, Color(0xFF00E676).copy(alpha = alpha1), style = Stroke(6f + 10f * p))
                    drawPath(path, Color(0xFF00E676).copy(alpha = alpha2), style = Stroke(6f + 22f * p))
                }

                if (confirmAt > 0L && confirmProgress.value < 1f) {
                    val path = Path().apply {
                        moveTo(pts[0].x, pts[0].y)
                        for (i in 1 until pts.size) lineTo(pts[i].x, pts[i].y)
                        close()
                    }
                    val p = confirmProgress.value
                    clipPath(path) {
                        val left = pts.minOf { it.x }
                        val right = pts.maxOf { it.x }
                        val top = pts.minOf { it.y }
                        val bottom = pts.maxOf { it.y }
                        val width = right - left
                        val barW = width / 10f
                        val gap = barW
                        val total = (barW + gap) * 4
                        val startX = left - total
                        val travel = width + total * 2
                        val baseX = startX + travel * p
                        repeat(4) { i ->
                            val x = baseX + i * (barW + gap)
                            drawRect(
                                color = Color.White.copy(alpha = 0.18f),
                                topLeft = Offset(x, top),
                                size = androidx.compose.ui.geometry.Size(barW, bottom - top)
                            )
                        }
                    }
                }
            }
        }

        if (animBmp != null && viewW > 0 && viewH > 0) {
            val p = animProgress.value.coerceIn(0f, 1f)
            val aspect = animBmp!!.width.toFloat() / animBmp!!.height.toFloat()
            val startW = min(viewW.toFloat(), viewH.toFloat()) * 0.6f
            val startH = startW / aspect
            val endW = with(density) { 72.dp.toPx() }
            val endH = endW / aspect

            val startX = (viewW - startW) / 2f
            val startY = (viewH - startH) / 2f
            val endX = viewW - endW - with(density) { 16.dp.toPx() }
            val endY = viewH - endH - with(density) { 16.dp.toPx() }

            fun flerp(a: Float, b: Float, t: Float) = a + (b - a) * t
            val curW = flerp(startW, endW, p)
            val curH = flerp(startH, endH, p)
            val curX = flerp(startX, endX, p)
            val curY = flerp(startY, endY, p)

            val curWdp = with(density) { (curW / density.density).dp }
            val curHdp = with(density) { (curH / density.density).dp }
            val curXdp = with(density) { (curX / density.density).dp }
            val curYdp = with(density) { (curY / density.density).dp }

            Image(
                bitmap = animBmp!!.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier
                    .size(width = curWdp, height = curHdp)
                    .offset(x = curXdp, y = curYdp)
                    .background(Color.Black.copy(alpha = 0.12f), shape = MaterialTheme.shapes.small)
            )
        }

        Row(
            Modifier
                .align(Alignment.BottomStart)
                .padding(12.dp)
                .background(Color.Black.copy(alpha = 0.28f), shape = MaterialTheme.shapes.small)
                .padding(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            val mini = debug ?: thumb
            if (mini != null) {
                Image(bitmap = mini.asImageBitmap(), contentDescription = null, modifier = Modifier.size(88.dp))
            } else {
                Text("Нет предпросмотра", color = Color.White)
            }
        }

        Column(
            Modifier
                .align(Alignment.BottomEnd)
                .padding(16.dp),
            horizontalAlignment = Alignment.End
        ) {
            if (pages.isNotEmpty() && !captureInProgress) {
                FilledTonalButton(
                    onClick = { viewModel.removeLastPage() },
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                ) { Text("↩ Отменить") }
                Spacer(Modifier.height(10.dp))
            }

            if (pages.isNotEmpty()) {
                Box(
                    modifier = Modifier
                        .size(72.dp)
                        .clickable { onOpenStack() }
                ) {
                    val tail = pages.takeLast(3)
                    tail.forEachIndexed { idx, page ->
                        val offset = (tail.size - 1 - idx) * 6
                        val bmp by remember(page.thumbFile) {
                            mutableStateOf(BitmapFactory.decodeFile(page.thumbFile.absolutePath))
                        }
                        if (bmp != null) {
                            Image(
                                bitmap = bmp!!.asImageBitmap(),
                                contentDescription = null,
                                modifier = Modifier
                                    .matchParentSize()
                                    .padding(all = offset.dp)
                                    .background(Color.Black.copy(alpha = 0.12f), shape = MaterialTheme.shapes.small)
                            )
                        }
                    }
                    Box(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .offset(x = 6.dp, y = (-6).dp)
                            .background(Color(0xFF00E676), shape = CircleShape)
                    ) {
                        Text(
                            text = pages.size.toString(),
                            color = Color.Black,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                        )
                    }
                }
            }
        }

        if (pages.isNotEmpty()) {
            TextButton(
                onClick = { viewModel.clearSession() },
                modifier = Modifier.align(Alignment.TopEnd).padding(12.dp)
            ) { Text("Очистить") }
        }
    }
}
