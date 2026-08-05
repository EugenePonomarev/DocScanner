package com.snowleopard.docscanner.core.imaging

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.media.Image
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.CLAHE
import kotlin.math.*

private const val TAG = "DocAnalyzerTwo"
private const val THUMBNAIL_WIDTH = 240

class DocumentAnalyzerTwo(
    private val onFrameSize: (w: Int, h: Int) -> Unit,
    private val onDebug: (bmp: Bitmap?, status: String) -> Unit,
    private val onResult: (DocumentDetectionResult) -> Unit,
) : ImageAnalysis.Analyzer {

    // --- Timing / throttling ---
    private val minAnalyzeIntervalMs = 40L
    private var lastAnalyzeAt = 0L
    private var frameIndex = 0

    // --- Stabilization / thresholds ---
    private var prevQuad: Array<Point>? = null
    private var prevScore: Double = 0.0
    private var stableFrames: Int = 0

    private val alpha = 0.22
    private val minScoreToAccept = 0.34
    private val keepPrevIfBetterDelta = 0.08
    private val minStableFramesToCrop = 4
    private val movementEpsNormalized = 0.010

    // --- ROI speed-ups ---
    private val maxWDetectFull = 480
    private val roiPadFrac = 0.16
    private val maxWDetectRoiMin = 300

    // --- Pass plan ---
    private val heavyEvery = 8
    private var missCounter = 0

    // --- Hough ---
    private val houghAngleTol = 15.0
    private val houghMinLineLenRatio = 0.25
    private val minCandidateAreaRatio = 0.035
    private val minCandidatePerimeterRatio = 0.45

    private enum class EnhanceProfile { NONE, SOFT, BW }

    private val enhanceProfile = EnhanceProfile.NONE

    // --- Profiling ---
    private data class T(var ms: Long = 0)
    private data class Times(
        val tResize: T = T(),
        val tPrep: T = T(),
        val tEdges: T = T(),
        val tCnt: T = T(),
        val tHough: T = T(),
        val tTotal: T = T(),
    )

    private val times = Times()

    private val maxRoiMisses = 3

    private val maxTrackingJumpNorm = 0.22
    private val minTrackingAreaRatio = 0.35
    private val maxTrackingAreaRatio = 2.8
    private val debugEveryFrames = 5

    private var stableCrop: Bitmap? = null
    private var stableThumbnail: Bitmap? = null

    private fun forgetStableBitmaps() {
        // Только сбрасываем ссылки.
        // Освобождение переданных Bitmap будет выполняться владельцем состояния.
        stableCrop = null
        stableThumbnail = null
    }

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastAnalyzeAt < minAnalyzeIntervalMs) {
            image.close()
            return
        }

        lastAnalyzeAt = now
        frameIndex++

        var graySource: Mat? = null
        var gray: Mat? = null

        try {
            val t0 = System.nanoTime()

            val source = yPlaneToGrayMat(image)
            if (source == null) {
                onDebug(null, "no gray")
                onResult(DocumentDetectionResult())
                return
            }

            graySource = source

            val rotation = image.imageInfo.rotationDegrees
            gray = rotateMat(source, rotation)

            // Ownership source передан rotateMat/gray.
            graySource = null

            val grayFrame = gray
                ?: error("Gray frame was not created")

            onFrameSize(grayFrame.cols(), grayFrame.rows())

            val out = detectAndCropFromGray(
                grayFull = grayFrame,
                colorSupplier = {
                    val color = imageProxyToColorBitmap(image)
                        ?: return@detectAndCropFromGray null

                    if (rotation == 0) {
                        color
                    } else {
                        try {
                            val rotated = Bitmap.createBitmap(
                                color,
                                0,
                                0,
                                color.width,
                                color.height,
                                Matrix().apply {
                                    postRotate(rotation.toFloat())
                                },
                                true,
                            )

                            if (rotated !== color) {
                                recycleOwnedBitmap(color)
                            }

                            rotated
                        } catch (t: Throwable) {
                            recycleOwnedBitmap(color)
                            throw t
                        }
                    }
                },
            )

            val totalMs = (
                    (System.nanoTime() - t0) / 1_000_000
                    ).coerceAtLeast(0)

            val status = out.status +
                    " | t(ms):R=${times.tResize.ms}" +
                    " P=${times.tPrep.ms}" +
                    " E=${times.tEdges.ms}" +
                    " C=${times.tCnt.ms}" +
                    " H=${times.tHough.ms}" +
                    " T=$totalMs"

            if (frameIndex % debugEveryFrames == 0) {
                onDebug(out.debugBmp, status)
            }

            onResult(
                DocumentDetectionResult(
                    polygon = out.polygon.map {
                        it.x.toFloat() to it.y.toFloat()
                    },
                    cropped = out.cropped,
                    thumbnail = out.thumbnail,
                    qualityAccepted = out.qualityAccepted,
                    confidence = out.confidence
                        .toFloat()
                        .coerceIn(0f, 1f),
                ),
            )
        } catch (t: Throwable) {
            val status = "error: ${t.message}"

            Log.e(TAG, "analyze error", t)
            onDebug(null, status)
            onResult(DocumentDetectionResult())
        } finally {
            releaseMat(graySource)
            releaseMat(gray)
            image.close()
        }
    }

    // ------------------------------------------------------------------------------------

    private data class DetectOut(
        val polygon: List<Point>,
        val cropped: Bitmap?,
        val thumbnail: Bitmap?,
        val debugBmp: Bitmap?,
        val status: String,
        val qualityAccepted: Boolean,
        val confidence: Double,
    )

    private fun detectAndCropFromGray(
        grayFull: Mat,
        colorSupplier: () -> Bitmap?,
    ): DetectOut {
        val srcW = grayFull.cols()
        val srcH = grayFull.rows()

        val grayDownFull = Mat()

        var grayRoi: Mat? = null
        var tmpResize: Mat? = null
        var edgesThin: Mat? = null
        var comb: Mat? = null
        var k3: Mat? = null
        var hierarchy: Mat? = null
        var clahe: CLAHE? = null

        val contours = ArrayList<MatOfPoint>()

        try {
            val tR0 = System.nanoTime()

            val scaleFull =
                if (srcW > maxWDetectFull) {
                    maxWDetectFull.toDouble() / srcW
                } else {
                    1.0
                }

            val dwFull = (srcW * scaleFull).roundToInt()
            val dhFull = (srcH * scaleFull).roundToInt()

            Imgproc.resize(
                grayFull,
                grayDownFull,
                Size(dwFull.toDouble(), dhFull.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_AREA,
            )

            times.tResize.ms = (
                    (System.nanoTime() - tR0) / 1_000_000
                    ).coerceAtLeast(0)

            val useRoi = prevQuad != null && missCounter < maxRoiMisses

            val roiRectDown: Rect? = if (useRoi) {
                val q = prevQuad!!.map { point ->
                    Point(
                        point.x * scaleFull,
                        point.y * scaleFull,
                    )
                }

                val minX = q.minOf { it.x }
                val maxX = q.maxOf { it.x }
                val minY = q.minOf { it.y }
                val maxY = q.maxOf { it.y }

                val pad = (
                        max(maxX - minX, maxY - minY) * roiPadFrac
                        ).coerceAtLeast(12.0)

                val x0 = floor(
                    (minX - pad).coerceAtLeast(0.0),
                ).toInt()

                val y0 = floor(
                    (minY - pad).coerceAtLeast(0.0),
                ).toInt()

                val x1 = ceil(
                    (maxX + pad).coerceAtMost(dwFull - 1.0),
                ).toInt()

                val y1 = ceil(
                    (maxY + pad).coerceAtMost(dhFull - 1.0),
                ).toInt()

                val w = (x1 - x0).coerceAtLeast(32)
                val h = (y1 - y0).coerceAtLeast(32)

                Rect(x0, y0, w, h)
            } else {
                null
            }

            val grayWork: Mat
            val dwWork: Int
            val dhWork: Int
            val roiOffsetX: Int
            val roiOffsetY: Int
            val scaleRoi: Double

            if (roiRectDown != null) {
                val roiMat = grayDownFull.submat(roiRectDown)
                grayRoi = roiMat

                val needScale = roiRectDown.width > maxWDetectRoiMin

                scaleRoi =
                    if (needScale) {
                        maxWDetectRoiMin.toDouble() / roiRectDown.width
                    } else {
                        1.0
                    }

                if (scaleRoi < 1.0) {
                    val resized = Mat()
                    tmpResize = resized

                    Imgproc.resize(
                        roiMat,
                        resized,
                        Size(
                            roiRectDown.width * scaleRoi,
                            roiRectDown.height * scaleRoi,
                        ),
                    )

                    grayWork = resized
                } else {
                    grayWork = roiMat
                }

                dwWork = grayWork.cols()
                dhWork = grayWork.rows()
                roiOffsetX = roiRectDown.x
                roiOffsetY = roiRectDown.y
            } else {
                grayWork = grayDownFull
                dwWork = dwFull
                dhWork = dhFull
                roiOffsetX = 0
                roiOffsetY = 0
                scaleRoi = 1.0
            }

            fun workToDownFull(point: Point): Point =
                Point(
                    point.x / scaleRoi + roiOffsetX,
                    point.y / scaleRoi + roiOffsetY,
                )

            val doHeavy =
                (frameIndex % heavyEvery == 0) ||
                        missCounter >= 3 ||
                        !useRoi

            val tP0 = System.nanoTime()

            if (doHeavy) {
                val createdClahe = Imgproc.createCLAHE(
                    1.1,
                    Size(8.0, 8.0),
                )

                clahe = createdClahe
                createdClahe.apply(grayWork, grayWork)
            }

            Imgproc.GaussianBlur(
                grayWork,
                grayWork,
                Size(3.0, 3.0),
                0.0,
            )

            times.tPrep.ms = (
                    (System.nanoTime() - tP0) / 1_000_000
                    ).coerceAtLeast(0)

            val tE0 = System.nanoTime()

            val edgeMat = Mat()
            edgesThin = edgeMat

            if (doHeavy) {
                val med = median(grayWork)
                val lower = max(0.0, 0.66 * med)
                val upper = min(255.0, 1.33 * med)

                Imgproc.Canny(
                    grayWork,
                    edgeMat,
                    lower,
                    upper,
                )
            } else {
                val meanVal = Core.mean(grayWork)
                    .`val`[0]
                    .coerceIn(1.0, 254.0)

                val lower = (meanVal * 0.66)
                    .coerceIn(0.0, 255.0)

                val upper = (meanVal * 1.33)
                    .coerceIn(0.0, 255.0)

                Imgproc.Canny(
                    grayWork,
                    edgeMat,
                    lower,
                    upper,
                )
            }

            val combinedMat = edgeMat.clone()
            comb = combinedMat

            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(3.0, 3.0),
            )
            k3 = kernel

            Imgproc.morphologyEx(
                combinedMat,
                combinedMat,
                Imgproc.MORPH_CLOSE,
                kernel,
                Point(-1.0, -1.0),
                1,
            )

            times.tEdges.ms = (
                    (System.nanoTime() - tE0) / 1_000_000
                    ).coerceAtLeast(0)

            val tC0 = System.nanoTime()

            val hierarchyMat = Mat()
            hierarchy = hierarchyMat

            Imgproc.findContours(
                combinedMat,
                contours,
                hierarchyMat,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE,
            )

            val frameAreaWork = (
                    dwWork * dhWork
                    ).toDouble()

            val areaMin = frameAreaWork * minCandidateAreaRatio
            val perimMin = min(dwWork, dhWork) *
                    minCandidatePerimeterRatio

            val sorted = contours
                .sortedByDescending {
                    Imgproc.contourArea(it)
                }
                .take(5)

            var bestQuadWork: Array<Point>? = null
            var bestScore = -1.0
            var bestBrk = ScoreBreakdown()

            var srcTag =
                if (useRoi) {
                    if (doHeavy) {
                        "contour-roi-H"
                    } else {
                        "contour-roi"
                    }
                } else {
                    if (doHeavy) {
                        "contour-full-H"
                    } else {
                        "contour-full"
                    }
                }

            for (contour in sorted) {
                val area = Imgproc.contourArea(contour)
                if (area < areaMin) continue

                val contourPoints = contour.toArray()

                val contour2f = MatOfPoint2f(*contourPoints)
                val peri = try {
                    Imgproc.arcLength(contour2f, true)
                } finally {
                    contour2f.release()
                }

                if (peri < perimMin) continue

                val approx = MatOfPoint2f()
                val approxInput = MatOfPoint2f(*contourPoints)

                val quad = try {
                    Imgproc.approxPolyDP(
                        approxInput,
                        approx,
                        0.02 * peri,
                        true,
                    )

                    when (approx.total().toInt()) {
                        4 -> approx.toArray()
                        3 -> reconstruct4thFromTriangle(
                            approx.toArray(),
                        )
                        else -> minAreaRectToQuad(contour)
                    }
                } finally {
                    approxInput.release()
                    approx.release()
                }

                val quadContour = MatOfPoint(*quad)
                val convex = try {
                    Imgproc.isContourConvex(quadContour)
                } finally {
                    quadContour.release()
                }

                if (!convex) continue

                val orderedWork = orderQuad(quad).toTypedArray()

                val orderedFull = Array(4) { index ->
                    workToDownFull(orderedWork[index])
                }

                val breakdownFast = scoreQuadFast(
                    quadWork = orderedWork,
                    quadDownFull = orderedFull,
                    contourAreaWork = area,
                    frameAreaWork = frameAreaWork,
                    dwFull = dwFull,
                    dhFull = dhFull,
                )

                var score = combinedScoreFast(breakdownFast)

                val breakdown =
                    if (score >= 0.25) {
                        val edgeScore = edgeSupportScore(
                            orderedWork,
                            combinedMat,
                        )

                        breakdownFast.copy(edge = edgeScore)
                    } else {
                        breakdownFast
                    }

                score = combinedScore(breakdown)

                if (score > bestScore) {
                    bestScore = score
                    bestQuadWork = orderedWork
                    bestBrk = breakdown
                }
            }

            times.tCnt.ms = (
                    (System.nanoTime() - tC0) / 1_000_000
                    ).coerceAtLeast(0)

            val tH0 = System.nanoTime()

            if (bestQuadWork == null && doHeavy) {
                val houghQuad = findRectByHough(
                    combinedMat,
                    dwWork,
                    dhWork,
                )

                if (houghQuad != null) {
                    val rectArea = polygonArea(
                        orderedList(houghQuad),
                    )

                    val orderedFull = Array(4) { index ->
                        workToDownFull(houghQuad[index])
                    }

                    val breakdownFast = scoreQuadFast(
                        quadWork = houghQuad,
                        quadDownFull = orderedFull,
                        contourAreaWork = rectArea,
                        frameAreaWork = frameAreaWork,
                        dwFull = dwFull,
                        dhFull = dhFull,
                    )

                    val edgeScore = edgeSupportScore(
                        houghQuad,
                        combinedMat,
                    )

                    val breakdown = breakdownFast.copy(
                        edge = edgeScore,
                    )

                    val score = combinedScore(breakdown)

                    val hValid =
                        breakdown.border >= 0.08 &&
                                (
                                        breakdown.edge >= 0.10 ||
                                                breakdown.rectangularity >= 0.80
                                        )

                    if (hValid) {
                        bestQuadWork = houghQuad
                        bestScore = score
                        bestBrk = breakdown
                        srcTag =
                            if (useRoi) {
                                "hough-roi"
                            } else {
                                "hough-full"
                            }
                    }
                }
            }

            times.tHough.ms = (
                    (System.nanoTime() - tH0) / 1_000_000
                    ).coerceAtLeast(0)

            if (bestQuadWork == null) {
                missCounter++
                stableFrames = 0
                forgetStableBitmaps()

                if (missCounter >= maxRoiMisses) {
                    prevQuad = null
                    prevScore = 0.0
                }

                val debug = debugBitmapIfNeeded(combinedMat)

                return DetectOut(
                    polygon = emptyList(),
                    cropped = null,
                    thumbnail = null,
                    debugBmp = debug,
                    status = "no quad ($srcTag) miss=$missCounter",
                    qualityAccepted = false,
                    confidence = 0.0,
                )
            }

            val selectedQuadWork = bestQuadWork
                ?: error("Best quad disappeared")

            val bestQuadDownFull = selectedQuadWork
                .map(::workToDownFull)
                .toTypedArray()

            val toSrc: (Point) -> Point = { point ->
                if (scaleFull < 1.0) {
                    Point(
                        point.x / scaleFull,
                        point.y / scaleFull,
                    )
                } else {
                    Point(point.x, point.y)
                }
            }

            val currentQuad = bestQuadDownFull
                .map(toSrc)
                .toTypedArray()

            val geomStrong =
                bestBrk.rectangularity > 0.68 &&
                        bestBrk.rightAngles > 0.72 &&
                        bestBrk.size > 0.12

            val borderOk = bestBrk.border >= 0.06

            val edgeOk =
                bestBrk.edge >= 0.09 ||
                        (
                                geomStrong &&
                                        bestBrk.border >= 0.05
                                )

            val baseQualityAccepted =
                bestScore >= minScoreToAccept &&
                        borderOk &&
                        edgeOk

            val continuityAccepted = isCandidateConsistent(
                currentQuad = currentQuad,
                previousQuad = prevQuad,
                frameWidth = srcW,
                frameHeight = srcH,
            )

            val qualityAccepted =
                baseQualityAccepted &&
                        continuityAccepted

            if (qualityAccepted) {
                missCounter = 0
            } else {
                missCounter++

                if (missCounter >= maxRoiMisses) {
                    prevQuad = null
                    prevScore = 0.0
                    stableFrames = 0
                }
            }

            val previousQuad = prevQuad

            val usePrevKeep =
                qualityAccepted &&
                        previousQuad != null &&
                        bestScore + keepPrevIfBetterDelta < prevScore

            val moveNorm = previousQuad?.let { previous ->
                currentQuad.indices
                    .map { index ->
                        dist(
                            currentQuad[index],
                            previous[index],
                        )
                    }
                    .average() / hypot(
                    srcW.toDouble(),
                    srcH.toDouble(),
                )
            } ?: 1.0

            val smoothed = when {
                !qualityAccepted -> {
                    previousQuad ?: currentQuad
                }

                previousQuad != null && !usePrevKeep -> {
                    Array(4) { index ->
                        lerp(
                            previousQuad[index],
                            currentQuad[index],
                        )
                    }
                }

                else -> {
                    previousQuad ?: currentQuad
                }
            }

            if (qualityAccepted) {
                prevQuad = smoothed
                prevScore = max(
                    prevScore * (1 - alpha),
                    bestScore,
                )
            }

            val needStable = minStableFramesToCrop
            val stableFramesBefore = stableFrames

            stableFrames =
                if (
                    qualityAccepted &&
                    bestScore >= minScoreToAccept &&
                    moveNorm < movementEpsNormalized
                ) {
                    (stableFrames + 1).coerceAtMost(8)
                } else {
                    0
                }

            val cropReady =
                qualityAccepted &&
                        stableFrames >= needStable

            if (!cropReady) {
                forgetStableBitmaps()
            } else if (
                stableCrop == null ||
                stableFramesBefore < needStable
            ) {
                val color = colorSupplier()

                val warpedCrop = try {
                    if (color != null) {
                        warpByQuadBitmap(color, smoothed)
                    } else {
                        warpByQuadGray(grayFull, smoothed)
                    }
                } finally {
                    // color создан только внутри analyzer и больше не нужен.
                    recycleOwnedBitmap(color)
                }

                val processedCrop = try {
                    applyPostproc(
                        warpedCrop,
                        enhanceProfile,
                    )
                } catch (t: Throwable) {
                    recycleOwnedBitmap(warpedCrop)
                    throw t
                }

                if (processedCrop !== warpedCrop) {
                    recycleOwnedBitmap(warpedCrop)
                }

                val thumbnail = try {
                    makeThumbnail(processedCrop)
                } catch (t: Throwable) {
                    recycleOwnedBitmap(processedCrop)
                    throw t
                }

                stableCrop = processedCrop
                stableThumbnail = thumbnail
            }

            val debug = debugBitmapIfNeeded(combinedMat)

            val status = (
                    "src=$srcTag " +
                            "score=%.2f ".format(bestScore) +
                            "rect=%.2f ".format(bestBrk.rectangularity) +
                            "ang=%.2f ".format(bestBrk.rightAngles) +
                            "asp=%.2f ".format(bestBrk.aspect) +
                            "size=%.2f ".format(bestBrk.size) +
                            "border=%.2f ".format(bestBrk.border) +
                            "edge=%.2f ".format(bestBrk.edge) +
                            "move=%.3f ".format(moveNorm) +
                            "stable=$stableFrames/$needStable " +
                            "accept=$qualityAccepted " +
                            "cont=$continuityAccepted"
                    )

            return DetectOut(
                polygon = if (qualityAccepted) {
                    smoothed.toList()
                } else {
                    emptyList()
                },
                cropped = stableCrop,
                thumbnail = stableThumbnail,
                debugBmp = debug,
                status = status,
                qualityAccepted = qualityAccepted,
                confidence = bestScore,
            )
        } finally {
            clahe?.let {
                runCatching {
                    it.collectGarbage()
                }
            }

            releaseMat(tmpResize)
            releaseMat(grayRoi)
            releaseAll(contours)
            releaseMat(hierarchy)
            releaseMat(edgesThin)
            releaseMat(comb)
            releaseMat(k3)
            releaseMat(grayDownFull)
        }
    }

    // ---------- Scoring / utils ----------
    private data class ScoreBreakdown(
        val rectangularity: Double = 0.0,
        val rightAngles: Double = 0.0,
        val aspect: Double = 0.0,
        val size: Double = 0.0,
        val border: Double = 0.0,
        val edge: Double = 0.0,
    )

    private fun scoreQuadFast(
        quadWork: Array<Point>,
        quadDownFull: Array<Point>,
        contourAreaWork: Double,
        frameAreaWork: Double,
        dwFull: Int,
        dhFull: Int,
    ): ScoreBreakdown {
        val quadMat = MatOfPoint(*quadWork)

        val rect = try {
            Imgproc.boundingRect(quadMat)
        } finally {
            quadMat.release()
        }

        val aspect = rect.width.toDouble() /
                max(1.0, rect.height.toDouble())

        val rectangularity = (
                contourAreaWork /
                        max(
                            1.0,
                            rect.width.toDouble() * rect.height,
                        )
                ).coerceIn(0.0, 1.0)

        val angScore = rightAngleScore(quadWork)
        val aspectNorm = normalizeAspect(aspect)
        val sizeNorm = (
                contourAreaWork / frameAreaWork
                ).coerceIn(0.0, 1.0)

        val borderScore = borderDistanceScore(
            quadDownFull,
            dwFull,
            dhFull,
        )

        return ScoreBreakdown(
            rectangularity = rectangularity,
            rightAngles = angScore,
            aspect = aspectNorm,
            size = sizeNorm,
            border = borderScore,
            edge = 0.0,
        )
    }

    private fun combinedScoreFast(b: ScoreBreakdown): Double {
        return 0.36 * b.rectangularity +
                0.24 * b.rightAngles +
                0.18 * b.size +
                0.12 * b.aspect +
                0.10 * b.border
    }

    private fun combinedScore(b: ScoreBreakdown): Double {
        return 0.30 * b.rectangularity +
                0.22 * b.rightAngles +
                0.18 * b.size +
                0.12 * b.aspect +
                0.12 * b.border +
                0.06 * b.edge
    }

    private fun edgeSupportScore(
        quadWork: Array<Point>,
        edgesMaskWork: Mat,
    ): Double {
        val mask = Mat.zeros(
            edgesMaskWork.size(),
            CvType.CV_8UC1,
        )

        val inter = Mat()

        return try {
            val points = arrayOf(
                quadWork[0],
                quadWork[1],
                quadWork[2],
                quadWork[3],
            )

            for (i in 0..3) {
                Imgproc.line(
                    mask,
                    points[i],
                    points[(i + 1) % 4],
                    Scalar(255.0),
                    3,
                )
            }

            Core.bitwise_and(
                mask,
                edgesMaskWork,
                inter,
            )

            val maskCount = Core.countNonZero(mask)
            val interCount = Core.countNonZero(inter)

            if (maskCount <= 0) {
                0.0
            } else {
                (
                        interCount.toDouble() / maskCount.toDouble()
                        ).coerceIn(0.0, 1.0)
            }
        } finally {
            inter.release()
            mask.release()
        }
    }

    private fun borderDistanceScore(quad: Array<Point>, w: Int, h: Int): Double {
        val minWH = min(w, h).toDouble()
        val safe = 0.08 * minWH
        val minDist = quad.minOf { p ->
            val dx = min(p.x, (w - 1).toDouble() - p.x)
            val dy = min(p.y, (h - 1).toDouble() - p.y)
            min(dx, dy)
        }
        return (minDist / max(1e-6, safe)).coerceIn(0.0, 1.0)
    }

    private data class Line(val a: Double, val b: Double, val c: Double)
    private data class Seg(
        val x1: Double,
        val y1: Double,
        val x2: Double,
        val y2: Double,
        val angleDeg: Double
    ) {
        val cx = (x1 + x2) * 0.5
        val cy = (y1 + y2) * 0.5
    }

    private fun findRectByHough(
        edges: Mat,
        w: Int,
        h: Int,
    ): Array<Point>? {
        val lines = Mat()

        return try {
            val minLen = min(w, h) *
                    houghMinLineLenRatio

            Imgproc.HoughLinesP(
                edges,
                lines,
                1.0,
                Math.PI / 180.0,
                120,
                minLen,
                10.0,
            )

            if (lines.empty()) {
                return null
            }

            val segs = ArrayList<Seg>(lines.rows())

            for (i in 0 until lines.rows()) {
                val values = lines.get(i, 0)

                val x1 = values[0]
                val y1 = values[1]
                val x2 = values[2]
                val y2 = values[3]

                val angleDeg = Math.toDegrees(
                    atan2(y2 - y1, x2 - x1),
                )

                segs += Seg(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    angleDeg = angleDeg,
                )
            }

            val horiz = segs
                .filter {
                    val angle = abs(
                        normalizeAngle(it.angleDeg),
                    )

                    angle < houghAngleTol ||
                            angle > 180.0 - houghAngleTol
                }
                .sortedBy { it.cy }

            val vert = segs
                .filter {
                    val angle = abs(
                        normalizeAngle(it.angleDeg),
                    )

                    angle in (
                            90.0 - houghAngleTol
                            )..(
                            90.0 + houghAngleTol
                            )
                }
                .sortedBy { it.cx }

            if (horiz.size < 2 || vert.size < 2) {
                return null
            }

            val top = horiz.first()
            val bottom = horiz.last()
            val left = vert.first()
            val right = vert.last()

            val lTop = lineFromSegment(top)
            val lBottom = lineFromSegment(bottom)
            val lLeft = lineFromSegment(left)
            val lRight = lineFromSegment(right)

            val tl = intersect(lTop, lLeft) ?: return null
            val tr = intersect(lTop, lRight) ?: return null
            val br = intersect(lBottom, lRight) ?: return null
            val bl = intersect(lBottom, lLeft) ?: return null

            fun inBounds(point: Point): Boolean =
                point.x >= -w * 0.2 &&
                        point.x <= w * 1.2 &&
                        point.y >= -h * 0.2 &&
                        point.y <= h * 1.2

            if (
                !inBounds(tl) ||
                !inBounds(tr) ||
                !inBounds(br) ||
                !inBounds(bl)
            ) {
                return null
            }

            val quad = orderQuad(
                arrayOf(tl, tr, br, bl),
            ).toTypedArray()

            val area = polygonArea(
                orderedList(quad),
            )

            val frameArea = (w * h).toDouble()

            if (area < frameArea * 0.06) {
                return null
            }

            quad
        } finally {
            lines.release()
        }
    }

    private fun normalizeAngle(a: Double): Double {
        var x = a
        while (x <= -180) x += 360.0
        while (x > 180) x -= 360.0
        return x
    }

    private fun lineFromSegment(s: Seg): Line {
        val a = s.y1 - s.y2
        val b = s.x2 - s.x1
        val c = s.x1 * s.y2 - s.x2 * s.y1
        val norm = sqrt(a * a + b * b).coerceAtLeast(1e-6)
        return Line(a / norm, b / norm, c / norm)
    }

    private fun intersect(l1: Line, l2: Line): Point? {
        val d = l1.a * l2.b - l2.a * l1.b
        if (abs(d) < 1e-6) return null
        val x = (l2.b * (-l1.c) - l1.b * (-l2.c)) / d
        val y = (l1.a * (-l2.c) - l2.a * (-l1.c)) / d
        return Point(x, y)
    }

    // ---------- Geometry / helpers ----------

    private fun minAreaRectToQuad(
        c: MatOfPoint,
    ): Array<Point> {
        val points = MatOfPoint2f(*c.toArray())

        return try {
            val rotatedRect = Imgproc.minAreaRect(points)
            val result = Array(4) { Point() }

            rotatedRect.points(result)

            orderQuad(result).toTypedArray()
        } finally {
            points.release()
        }
    }

    private fun reconstruct4thFromTriangle(tri: Array<Point>): Array<Point> {
        val p0 = tri[0]
        val p1 = tri[1]
        val p2 = tri[2]

        val candidates = listOf(
            Point(
                p0.x + p1.x - p2.x,
                p0.y + p1.y - p2.y,
            ),
            Point(
                p0.x + p2.x - p1.x,
                p0.y + p2.y - p1.y,
            ),
            Point(
                p1.x + p2.x - p0.x,
                p1.y + p2.y - p0.y,
            ),
        )

        val bestCandidate = candidates
            .mapNotNull { candidate ->
                val ordered = orderQuad(
                    arrayOf(p0, p1, p2, candidate),
                ).toTypedArray()

                if (!isUsableQuad(ordered)) {
                    return@mapNotNull null
                }

                ordered to rightAngleScore(ordered)
            }
            .maxByOrNull { it.second }
            ?.first

        if (bestCandidate != null) {
            return bestCandidate
        }

        val contour = MatOfPoint(*tri)

        return minAreaRectToQuad(contour).also {
            contour.release()
        }
    }

    private fun isUsableQuad(quad: Array<Point>): Boolean {
        if (quad.size != 4) return false

        val minSideLength = (0 until 4)
            .minOf { index ->
                dist(
                    quad[index],
                    quad[(index + 1) % quad.size],
                )
            }

        if (minSideLength < 3.0) return false

        val contour = MatOfPoint(*quad)
        val isConvex = Imgproc.isContourConvex(contour)
        contour.release()

        return isConvex && polygonArea(orderedList(quad)) > 1.0
    }

    private fun orderQuad(pts: Array<Point>): List<Point> {
        val tl = pts.minByOrNull { it.x + it.y } ?: pts.first()
        val br = pts.maxByOrNull { it.x + it.y } ?: pts.last()
        val sortedDiff = pts.sortedBy { it.x - it.y }
        val bl = sortedDiff.firstOrNull() ?: tl
        val tr = sortedDiff.lastOrNull() ?: br
        return listOf(tl, tr, br, bl)
    }

    private fun orderedList(arr: Array<Point>) = listOf(arr[0], arr[1], arr[2], arr[3])

    private fun rightAngleScore(quad: Array<Point>): Double {
        fun angle(a: Point, b: Point, c: Point): Double {
            val abx = a.x - b.x
            val aby = a.y - b.y
            val cbx = c.x - b.x
            val cby = c.y - b.y
            val dot = abx * cbx + aby * cby
            val norm = sqrt((abx * abx + aby * aby) * (cbx * cbx + cby * cby)).coerceAtLeast(1e-6)
            val cos = (dot / norm).coerceIn(-1.0, 1.0)
            return Math.toDegrees(acos(cos))
        }

        val pts = orderQuad(quad)
        val angs = listOf(
            angle(pts[3], pts[0], pts[1]),
            angle(pts[0], pts[1], pts[2]),
            angle(pts[1], pts[2], pts[3]),
            angle(pts[2], pts[3], pts[0])
        )
        val scores = angs.map { 1.0 - (abs(it - 90.0) / 20.0).coerceIn(0.0, 1.0) }
        return scores.average()
    }

    private fun normalizeAspect(ratio: Double): Double {
        val r = if (ratio < 1.0) 1.0 / ratio else ratio
        val targets = doubleArrayOf(1.414213562, 1.294117647) // A4, Letter
        val diff = targets.minOf { abs(r - it) }
        return (1.0 - (diff / 0.6)).coerceIn(0.0, 1.0)
    }

    private fun median(gray: Mat): Double {
        val hist = Mat()
        val channels = MatOfInt(0)
        val mask = Mat()
        val histSize = MatOfInt(256)
        val ranges = MatOfFloat(0f, 256f)

        return try {
            Imgproc.calcHist(
                listOf(gray),
                channels,
                mask,
                hist,
                histSize,
                ranges,
            )

            var accumulated = 0.0
            val total = gray.rows() *
                    gray.cols().toDouble()

            for (i in 0 until 256) {
                accumulated += hist.get(i, 0)[0]

                if (accumulated >= total / 2) {
                    return i.toDouble()
                }
            }

            127.0
        } finally {
            ranges.release()
            histSize.release()
            mask.release()
            channels.release()
            hist.release()
        }
    }

    private fun polygonArea(quad: List<Point>): Double {
        var s = 0.0
        for (i in quad.indices) {
            val a = quad[i]
            val b = quad[(i + 1) % quad.size]
            s += a.x * b.y - b.x * a.y
        }
        return abs(s) * 0.5
    }

    private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)

    private fun lerp(a: Point, b: Point): Point =
        Point(
            a.x + (b.x - a.x) * alpha,
            a.y + (b.y - a.y) * alpha,
        )

    // ---------- Warp ----------
    private fun warpByQuadBitmap(
        src: Bitmap,
        quad: Array<Point>,
    ): Bitmap {
        val widthTop = dist(quad[0], quad[1])
        val widthBottom = dist(quad[3], quad[2])
        val heightLeft = dist(quad[0], quad[3])
        val heightRight = dist(quad[1], quad[2])

        val width = max(
            widthTop,
            widthBottom,
        ).roundToInt().coerceAtLeast(300)

        val height = max(
            heightLeft,
            heightRight,
        ).roundToInt().coerceAtLeast(300)

        var srcMat: Mat? = null
        var dst: Mat? = null
        var srcPts: MatOfPoint2f? = null
        var dstPts: MatOfPoint2f? = null
        var transform: Mat? = null

        try {
            val sourceMat = Mat()
            srcMat = sourceMat

            val destinationMat = Mat()
            dst = destinationMat

            Utils.bitmapToMat(src, sourceMat)

            val sourcePoints = MatOfPoint2f(
                quad[0],
                quad[1],
                quad[2],
                quad[3],
            )
            srcPts = sourcePoints

            val destinationPoints = MatOfPoint2f(
                Point(0.0, 0.0),
                Point(width - 1.0, 0.0),
                Point(width - 1.0, height - 1.0),
                Point(0.0, height - 1.0),
            )
            dstPts = destinationPoints

            val perspective = Imgproc.getPerspectiveTransform(
                sourcePoints,
                destinationPoints,
            )
            transform = perspective

            Imgproc.warpPerspective(
                sourceMat,
                destinationMat,
                perspective,
                Size(
                    width.toDouble(),
                    height.toDouble(),
                ),
            )

            val output = createBitmap(width, height)

            try {
                Utils.matToBitmap(destinationMat, output)
            } catch (t: Throwable) {
                recycleOwnedBitmap(output)
                throw t
            }

            return output
        } finally {
            releaseMat(transform)
            releaseMat(dstPts)
            releaseMat(srcPts)
            releaseMat(dst)
            releaseMat(srcMat)
        }
    }

    private fun warpByQuadGray(
        srcGray: Mat,
        quad: Array<Point>,
    ): Bitmap {
        val widthTop = dist(quad[0], quad[1])
        val widthBottom = dist(quad[3], quad[2])
        val heightLeft = dist(quad[0], quad[3])
        val heightRight = dist(quad[1], quad[2])

        val width = max(
            widthTop,
            widthBottom,
        ).roundToInt().coerceAtLeast(300)

        val height = max(
            heightLeft,
            heightRight,
        ).roundToInt().coerceAtLeast(300)

        var dst: Mat? = null
        var rgba: Mat? = null
        var srcPts: MatOfPoint2f? = null
        var dstPts: MatOfPoint2f? = null
        var transform: Mat? = null

        try {
            val destinationMat = Mat()
            dst = destinationMat

            val rgbaMat = Mat()
            rgba = rgbaMat

            val sourcePoints = MatOfPoint2f(
                quad[0],
                quad[1],
                quad[2],
                quad[3],
            )
            srcPts = sourcePoints

            val destinationPoints = MatOfPoint2f(
                Point(0.0, 0.0),
                Point(width - 1.0, 0.0),
                Point(width - 1.0, height - 1.0),
                Point(0.0, height - 1.0),
            )
            dstPts = destinationPoints

            val perspective = Imgproc.getPerspectiveTransform(
                sourcePoints,
                destinationPoints,
            )
            transform = perspective

            Imgproc.warpPerspective(
                srcGray,
                destinationMat,
                perspective,
                Size(
                    width.toDouble(),
                    height.toDouble(),
                ),
            )

            Imgproc.cvtColor(
                destinationMat,
                rgbaMat,
                Imgproc.COLOR_GRAY2RGBA,
            )

            val output = createBitmap(width, height)

            try {
                Utils.matToBitmap(rgbaMat, output)
            } catch (t: Throwable) {
                recycleOwnedBitmap(output)
                throw t
            }

            return output
        } finally {
            releaseMat(transform)
            releaseMat(dstPts)
            releaseMat(srcPts)
            releaseMat(rgba)
            releaseMat(dst)
        }
    }

    private fun matToBitmap(mat: Mat): Bitmap {
        val visible = Mat()

        return try {
            if (mat.type() == CvType.CV_8UC1) {
                Imgproc.cvtColor(
                    mat,
                    visible,
                    Imgproc.COLOR_GRAY2RGBA,
                )
            } else {
                Imgproc.cvtColor(
                    mat,
                    visible,
                    Imgproc.COLOR_BGR2RGBA,
                )
            }

            val output = createBitmap(
                visible.cols(),
                visible.rows(),
            )

            try {
                Utils.matToBitmap(visible, output)
            } catch (t: Throwable) {
                recycleOwnedBitmap(output)
                throw t
            }

            output
        } finally {
            visible.release()
        }
    }

    private fun makeThumbnail(b: Bitmap): Bitmap {
        val scale = THUMBNAIL_WIDTH.toFloat() / b.width
        val height = (b.height * scale).roundToInt()

        return b.scale(
            THUMBNAIL_WIDTH,
            height,
        )
    }

    // ---------- Post-process ----------
    private fun applyPostproc(bmp: Bitmap, profile: EnhanceProfile): Bitmap = when (profile) {
        EnhanceProfile.NONE -> bmp
        EnhanceProfile.SOFT -> enhanceSoft(bmp)
        EnhanceProfile.BW -> enhanceBW(bmp)
    }

    private fun enhanceSoft(bmp: Bitmap): Bitmap {
        var rgba: Mat? = null
        var rgb: Mat? = null
        var ycrcb: Mat? = null
        var blur: Mat? = null
        var clahe: CLAHE? = null

        val channels = ArrayList<Mat>(3)

        try {
            val rgbaMat = Mat().also {
                rgba = it
            }

            val rgbMat = Mat().also {
                rgb = it
            }

            val ycrcbMat = Mat().also {
                ycrcb = it
            }

            val blurMat = Mat().also {
                blur = it
            }

            Utils.bitmapToMat(bmp, rgbaMat)

            Imgproc.cvtColor(
                rgbaMat,
                rgbMat,
                Imgproc.COLOR_RGBA2RGB,
            )

            Imgproc.cvtColor(
                rgbMat,
                ycrcbMat,
                Imgproc.COLOR_RGB2YCrCb,
            )

            Core.split(ycrcbMat, channels)

            val createdClahe = Imgproc.createCLAHE(
                0.8,
                Size(64.0, 64.0),
            )
            clahe = createdClahe

            createdClahe.apply(
                channels[0],
                channels[0],
            )

            Core.merge(channels, ycrcbMat)

            Imgproc.cvtColor(
                ycrcbMat,
                rgbaMat,
                Imgproc.COLOR_YCrCb2RGB,
            )

            Imgproc.GaussianBlur(
                rgbaMat,
                blurMat,
                Size(0.0, 0.0),
                0.6,
            )

            Core.addWeighted(
                rgbaMat,
                1.01,
                blurMat,
                -0.01,
                0.0,
                rgbaMat,
            )

            rgbaMat.convertTo(
                rgbaMat,
                -1,
                1.005,
                -0.5,
            )

            val output = createBitmap(
                bmp.width,
                bmp.height,
            )

            try {
                Utils.matToBitmap(rgbaMat, output)
            } catch (t: Throwable) {
                recycleOwnedBitmap(output)
                throw t
            }

            return output
        } finally {
            clahe?.let {
                runCatching {
                    it.collectGarbage()
                }
            }

            channels.forEach {
                releaseMat(it)
            }

            releaseMat(blur)
            releaseMat(ycrcb)
            releaseMat(rgb)
            releaseMat(rgba)
        }
    }

    private fun enhanceBW(bmp: Bitmap): Bitmap {
        var rgba: Mat? = null
        var gray: Mat? = null
        var bw: Mat? = null
        var outRgba: Mat? = null

        try {
            val rgbaMat = Mat().also {
                rgba = it
            }

            val grayMat = Mat().also {
                gray = it
            }

            val bwMat = Mat().also {
                bw = it
            }

            val outRgbaMat = Mat().also {
                outRgba = it
            }

            Utils.bitmapToMat(bmp, rgbaMat)

            Imgproc.cvtColor(
                rgbaMat,
                grayMat,
                Imgproc.COLOR_RGBA2GRAY,
            )

            Imgproc.adaptiveThreshold(
                grayMat,
                bwMat,
                255.0,
                Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY,
                25,
                10.0,
            )

            Imgproc.medianBlur(
                bwMat,
                bwMat,
                3,
            )

            Imgproc.cvtColor(
                bwMat,
                outRgbaMat,
                Imgproc.COLOR_GRAY2RGBA,
            )

            val output = createBitmap(
                bmp.width,
                bmp.height,
            )

            try {
                Utils.matToBitmap(outRgbaMat, output)
            } catch (t: Throwable) {
                recycleOwnedBitmap(output)
                throw t
            }

            return output
        } finally {
            releaseMat(outRgba)
            releaseMat(bw)
            releaseMat(gray)
            releaseMat(rgba)
        }
    }

    // ---------- ImageProxy helpers ----------
    @OptIn(ExperimentalGetImage::class)
    private fun yPlaneToGrayMat(
        image: ImageProxy,
    ): Mat? {
        val img = image.image ?: return null

        if (image.format != ImageFormat.YUV_420_888) {
            return null
        }

        val width = image.width
        val height = image.height

        val yPlane = img.planes[0]
        val yBuffer = yPlane.buffer
        val rowStride = yPlane.rowStride
        val pixelStride = yPlane.pixelStride

        val output = Mat(
            height,
            width,
            CvType.CV_8UC1,
        )

        return try {
            val row = ByteArray(width)

            for (r in 0 until height) {
                var c = 0
                var sourceIndex = r * rowStride

                while (c < width) {
                    row[c] = yBuffer.get(sourceIndex)
                    c++
                    sourceIndex += pixelStride
                }

                output.put(r, 0, row)
            }

            output
        } catch (t: Throwable) {
            releaseMat(output)
            throw t
        }
    }

    private fun rotateMat(
        src: Mat,
        rotationDeg: Int,
    ): Mat {
        if (
            rotationDeg != 90 &&
            rotationDeg != 180 &&
            rotationDeg != 270
        ) {
            return src
        }

        val destination = Mat()

        return try {
            when (rotationDeg) {
                90 -> Core.rotate(
                    src,
                    destination,
                    Core.ROTATE_90_CLOCKWISE,
                )

                180 -> Core.rotate(
                    src,
                    destination,
                    Core.ROTATE_180,
                )

                270 -> Core.rotate(
                    src,
                    destination,
                    Core.ROTATE_90_COUNTERCLOCKWISE,
                )
            }

            src.release()
            destination
        } catch (t: Throwable) {
            releaseMat(destination)
            throw t
        }
    }

    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToColorBitmap(
        image: ImageProxy,
    ): Bitmap? {
        val img = image.image ?: return null

        if (image.format != ImageFormat.YUV_420_888) {
            return null
        }

        val nv21 = yuv420888ToNv21(
            img,
            image.width,
            image.height,
        )

        var yuvMat: Mat? = null
        var rgba: Mat? = null

        try {
            val yuv = Mat(
                image.height + image.height / 2,
                image.width,
                CvType.CV_8UC1,
            )
            yuvMat = yuv

            val rgbaMat = Mat()
            rgba = rgbaMat

            yuv.put(0, 0, nv21)

            Imgproc.cvtColor(
                yuv,
                rgbaMat,
                Imgproc.COLOR_YUV2RGBA_NV21,
            )

            val output = createBitmap(
                image.width,
                image.height,
            )

            try {
                Utils.matToBitmap(rgbaMat, output)
            } catch (t: Throwable) {
                recycleOwnedBitmap(output)
                throw t
            }

            return output
        } finally {
            releaseMat(rgba)
            releaseMat(yuvMat)
        }
    }

    private fun yuv420888ToNv21(image: Image, width: Int, height: Int): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        val out = ByteArray(width * height + 2 * (width / 2) * (height / 2))
        var pos = 0

        // Y
        for (row in 0 until height) {
            var col = 0
            while (col < width) {
                val yIndex = row * yRowStride + col * yPixelStride
                out[pos++] = yBuffer.get(yIndex)
                col++
            }
        }

        // VU (NV21)
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            var col = 0
            while (col < chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                out[pos++] = vBuffer.get(vIndex)
                out[pos++] = uBuffer.get(uIndex)
                col++
            }
        }
        return out
    }

    // --- release helpers ---
    private fun releaseAll(
        contours: List<MatOfPoint>,
    ) {
        contours.forEach {
            releaseMat(it)
        }
    }

    private fun isCandidateConsistent(
        currentQuad: Array<Point>,
        previousQuad: Array<Point>?,
        frameWidth: Int,
        frameHeight: Int,
    ): Boolean {
        if (previousQuad == null) return true

        val frameDiagonal = hypot(
            frameWidth.toDouble(),
            frameHeight.toDouble(),
        ).coerceAtLeast(1.0)

        val currentCenter = quadCenter(currentQuad)
        val previousCenter = quadCenter(previousQuad)

        val jump = dist(currentCenter, previousCenter) / frameDiagonal

        val previousArea = polygonArea(
            orderedList(previousQuad),
        ).coerceAtLeast(1.0)

        val currentArea = polygonArea(
            orderedList(currentQuad),
        ).coerceAtLeast(1.0)

        val areaRatio = currentArea / previousArea

        return jump <= maxTrackingJumpNorm &&
                areaRatio in minTrackingAreaRatio..maxTrackingAreaRatio
    }

    private fun quadCenter(quad: Array<Point>): Point {
        return Point(
            quad.map { it.x }.average(),
            quad.map { it.y }.average(),
        )
    }

    private fun debugBitmapIfNeeded(mat: Mat): Bitmap? {
        return if (frameIndex % debugEveryFrames == 0) {
            matToBitmap(mat)
        } else {
            null
        }
    }

    private fun releaseMat(mat: Mat?) {
        if (mat == null) return

        runCatching {
            mat.release()
        }
    }

    private fun recycleOwnedBitmap(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled) {
            return
        }

        runCatching {
            bitmap.recycle()
        }
    }
}
