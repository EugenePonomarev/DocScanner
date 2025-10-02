package com.snowleopard.docscanner

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.snowleopard.docscanner.ui.theme.DocScannerTheme
import org.koin.androidx.compose.koinViewModel
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        OpenCVLoader.initLocal()

        setContent {
            DocScannerTheme {
                val nav = rememberNavController()
                // создаём/держим один общий VM на корне
                val vm: ScanViewModel = koinViewModel()

                NavHost(navController = nav, startDestination = "camera") {
                    composable("camera") {
                        CameraScreen(
                            viewModel = vm,
                            onOpenStack = { nav.navigate("stack") }
                        )
                    }
                    composable("stack") {
                        StackScreen(
                            viewModel = vm,
                            onBack = { nav.popBackStack() },
                            onCloseAndClear = {
                                vm.clearSession()
                                nav.popBackStack() // назад на камеру
                            }
                        )
                    }
                }
            }
        }
    }
}