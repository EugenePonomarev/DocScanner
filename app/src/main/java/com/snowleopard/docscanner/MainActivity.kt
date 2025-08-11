package com.snowleopard.docscanner

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.snowleopard.docscanner.ui.theme.DocScannerTheme
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        OpenCVLoader.initLocal()

        setContent {
            DocScannerTheme {
                val nav = rememberNavController()
                val vm: ScanViewModel = viewModel()

                NavHost(navController = nav, startDestination = "camera") {
                    composable("camera") {
                        CameraScreen(
                            viewModel = vm,
                            onSave = { nav.navigate("preview") }
                        )
                    }
                    composable("preview") {
                        PreviewScreen(
                            viewModel = vm,
                            onBack = { nav.popBackStack() }
                        )
                    }
                }
            }
        }
    }
}