package com.snowleopard.docscanner.feature.screens

import android.content.Intent
import android.net.Uri
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.TransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Rotate90DegreesCw
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.snowleopard.docscanner.feature.viewmodels.StackViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.snowleopard.docscanner.R
import com.snowleopard.docscanner.core.data.model.PdfOptions

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StackScreen(
    viewModel: StackViewModel,
    onBack: () -> Unit,
    onCloseAndClear: () -> Unit
) {
    val pages by viewModel.pages.collectAsState()
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()

    var selected by remember { mutableIntStateOf(0) }
    selected = selected.coerceIn(0, maxOf(0, pages.lastIndex))

    var building by remember { mutableStateOf(false) }
    var buildError by remember { mutableStateOf<String?>(null) }

    var scale by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var offsetY by remember { mutableFloatStateOf(0f) }
    val transformState = remember {
        TransformableState { zoom, pan, _ ->
            scale = (scale * zoom).coerceIn(0.5f, 6f)
            offsetX += pan.x
            offsetY += pan.y
        }
    }

    val rotateAnim = remember { Animatable(0f) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.documents_title, pages.size)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    if (pages.isNotEmpty()) {
                        IconButton(onClick = onCloseAndClear) {
                            Icon(Icons.Filled.Close, contentDescription = stringResource(R.string.close_and_clear))
                        }
                    }
                }
            )
        },
        bottomBar = {
            Surface(tonalElevation = 2.dp) {
                Row(
                    Modifier.fillMaxWidth().padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    OutlinedButton(
                        onClick = {
                            if (pages.isEmpty()) return@OutlinedButton
                            val uris = pages.mapNotNull {
                                runCatching {
                                    FileProvider.getUriForFile(
                                        ctx, "${ctx.packageName}.fileprovider", it.file
                                    )
                                }.getOrNull()
                            }
                            if (uris.isNotEmpty()) {
                                val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                                    type = "image/jpeg"
                                    putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList<Uri>(uris))
                                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                }
                                runCatching { ctx.startActivity(Intent.createChooser(intent, ctx.getString(R.string.share_images_title))) }
                            }
                        },
                        enabled = pages.isNotEmpty()
                    ) { Text(stringResource(R.string.share_jpg)) }

                    Button(
                        onClick = {
                            scope.launch {
                                building = true
                                buildError = null
                                val file = withContext(Dispatchers.Default) {
                                    runCatching {
                                        viewModel.buildPdf(
                                            PdfOptions(
                                                dpi = 300,
                                                marginMm = 6,
                                                jpegQuality = 92,
                                                autoRotateToPortrait = false
                                            )
                                        )
                                    }.getOrNull()
                                }
                                building = false
                                if (file == null) {
                                    buildError = ctx.getString(R.string.build_pdf_failed)
                                } else {
                                    val uri = FileProvider.getUriForFile(
                                        ctx, "${ctx.packageName}.fileprovider", file
                                    )
                                    val intent = Intent(Intent.ACTION_SEND).apply {
                                        type = "application/pdf"
                                        putExtra(Intent.EXTRA_STREAM, uri)
                                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    }
                                    runCatching {
                                        ctx.startActivity(
                                            Intent.createChooser(intent, ctx.getString(R.string.share_pdf_title))
                                        )
                                    }
                                }
                            }
                        },
                        enabled = pages.isNotEmpty() && !building
                    ) {
                        if (building) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(18.dp),
                                strokeWidth = 2.dp
                            )
                            Spacer(Modifier.width(8.dp))
                        }
                        Text(stringResource(R.string.build_pdf))
                    }
                }
            }
        }
    ) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding)
        ) {
            if (buildError != null) {
                Text(
                    text = buildError ?: "",
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 6.dp)
                )
            }

            Box(
                Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                if (pages.isEmpty()) {
                    Text(stringResource(R.string.no_pages), style = MaterialTheme.typography.titleMedium)
                } else {
                    val page = pages[selected]
                    AsyncImage(
                        model = ImageRequest.Builder(ctx)
                            .data(page.file)
                            .memoryCacheKey("${page.file.absolutePath}:${page.width}x${page.height}:${page.file.lastModified()}")
                            .diskCacheKey("${page.file.absolutePath}:${page.width}x${page.height}:${page.file.lastModified()}")
                            .crossfade(true)
                            .build(),
                        contentDescription = null,
                        modifier = Modifier
                            .fillMaxSize()
                            .transformable(transformState)
                            .graphicsLayer(
                                rotationZ = rotateAnim.value,
                                scaleX = scale, scaleY = scale,
                                translationX = offsetX, translationY = offsetY
                            )
                    )
                }
            }

            if (pages.isNotEmpty()) {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(stringResource(R.string.page_counter, selected + 1, pages.size))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        TextButton(
                            onClick = {
                                if (selected > 0) {
                                    viewModel.movePage(selected, selected - 1)
                                    selected -= 1
                                }
                            },
                            enabled = selected > 0
                        ) { Text("←") }

                        TextButton(
                            onClick = {
                                if (selected < pages.lastIndex) {
                                    viewModel.movePage(selected, selected + 1)
                                    selected += 1
                                }
                            },
                            enabled = selected < pages.lastIndex
                        ) { Text("→") }

                        IconButton(onClick = {
                            scope.launch {
                                rotateAnim.snapTo(0f)
                                rotateAnim.animateTo(90f, tween(durationMillis = 220))
                                rotateAnim.snapTo(0f)
                            }
                            viewModel.rotatePage(selected, +90)
                        }) {
                            Icon(Icons.Filled.Rotate90DegreesCw, contentDescription = stringResource(R.string.rotate))
                        }

                        IconButton(onClick = {
                            viewModel.deletePageAt(selected)
                            selected = selected.coerceAtMost(maxOf(0, pages.lastIndex - 1))
                            scale = 1f; offsetX = 0f; offsetY = 0f
                        }) {
                            Icon(Icons.Filled.Delete, contentDescription = stringResource(R.string.delete))
                        }
                    }
                }
            }

            Surface(tonalElevation = 2.dp) {
                LazyRow(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 10.dp, horizontal = 8.dp)
                ) {
                    itemsIndexed(pages, key = { _, it -> it.id }) { index, page ->
                        Column(
                            modifier = Modifier
                                .padding(horizontal = 6.dp)
                                .width(96.dp)
                                .clip(MaterialTheme.shapes.small)
                                .background(
                                    if (index == selected)
                                        MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)
                                    else
                                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                                )
                                .padding(6.dp)
                                .noRippleClickable {
                                    selected = index
                                    scale = 1f; offsetX = 0f; offsetY = 0f
                                },
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            AsyncImage(
                                model = ImageRequest.Builder(ctx)
                                    .data(page.thumbFile)
                                    .memoryCacheKey("${page.thumbFile.absolutePath}:${page.width}x${page.height}:${page.thumbFile.lastModified()}")
                                    .diskCacheKey("${page.thumbFile.absolutePath}:${page.width}x${page.height}:${page.thumbFile.lastModified()}")
                                    .crossfade(true)
                                    .build(),
                                contentDescription = null,
                                modifier = Modifier
                                    .size(width = 84.dp, height = 100.dp)
                                    .clip(MaterialTheme.shapes.small)
                            )
                            Spacer(Modifier.height(6.dp))
                            AssistChip(
                                onClick = {
                                    selected = index
                                    scale = 1f; offsetX = 0f; offsetY = 0f
                                },
                                leadingIcon = {
                                    if (index == selected) {
                                        Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                                    }
                                },
                                label = { Text("${index + 1}") }
                            )
                        }
                    }
                }
            }
        }
    }
}

fun Modifier.noRippleClickable(onClick: () -> Unit): Modifier = composed {
    clickable(
        interactionSource = remember { MutableInteractionSource() },
        indication = null,
        onClick = onClick
    )
}