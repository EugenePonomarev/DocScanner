# Doc Scanner

> 🚧 **Status:** Under active development

A fast, modern Android document scanner built with **Kotlin** and **Jetpack Compose**.  
It detects document edges (OpenCV) in real time, applies perspective warp, auto-captures stable frames, and lets you build/share PDFs or individual JPEGs — all on-device.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Permissions](#permissions)
- [Usage](#usage)
- [Implementation Notes](#implementation-notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Features

- **Live document detection** (OpenCV: edges, contours, Hough fallback)
- **Temporal stabilization** & **ROI acceleration** for smooth, fast locking
- **Perspective warp** to flatten pages
- **Auto-capture** with lock progression & confirmation animation
- **Duplicate protection** (aHash + centroid/area checks)
- **Page stack** management: reorder, rotate, delete
- **PDF builder** (A4 portrait/landscape per page, DPI & margins)
- **Share** as PDF or multiple JPEGs (FileProvider)
- **Offline** processing (no network)
- **Localization**: EN (default), RU

---

## Tech Stack

- **Language:** Kotlin 17 (K2)
- **UI:** Jetpack Compose, Material 3
- **Navigation:** Navigation Compose
- **Images:** Coil
- **Camera:** CameraX (Preview + ImageAnalysis)
- **CV/Geometry:** OpenCV (via Maven)
- **DI:** Koin
- **Async:** Kotlin Coroutines
- **PDF:** `android.graphics.pdf.PdfDocument`
- **Permissions:** `ActivityResultContracts.RequestPermission`
- **SDKs:** minSdk 26 / targetSdk 36
- **Build:** AGP 8.12.3, Gradle 8.13

> Optional (present in dependencies): **ML Kit Text Recognition** (not wired into UI yet).

---

## Architecture

Single Gradle module (`:app`) with **feature + layered** package structure:

