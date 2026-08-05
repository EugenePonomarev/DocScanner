package com.snowleopard.docscanner.app

import android.annotation.SuppressLint
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.snowleopard.docscanner.feature.screens.CameraScreen
import com.snowleopard.docscanner.feature.screens.StackScreen
import com.snowleopard.docscanner.feature.viewmodels.ScanViewModel
import com.snowleopard.docscanner.feature.viewmodels.StackViewModel
import com.snowleopard.docscanner.ui.theme.DocScannerTheme
import org.koin.androidx.compose.koinViewModel
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {

    @SuppressLint("UnrememberedGetBackStackEntry")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        OpenCVLoader.initLocal()

        setContent {
            DocScannerTheme {
                val navController = rememberNavController()

                NavHost(
                    navController = navController,
                    startDestination = "camera",
                ) {
                    composable("camera") {
                        val scanViewModel: ScanViewModel =
                            koinViewModel()

                        CameraScreen(
                            viewModel = scanViewModel,
                            onOpenStack = {
                                navController.navigate("stack")
                            },
                        )
                    }

                    composable("stack") {
                        val stackViewModel: StackViewModel =
                            koinViewModel()

                        val cameraBackStackEntry =
                            navController.getBackStackEntry("camera")

                        val scanViewModel: ScanViewModel =
                            koinViewModel(
                                viewModelStoreOwner = cameraBackStackEntry,
                            )

                        StackScreen(
                            viewModel = stackViewModel,
                            onBack = {
                                navController.popBackStack()
                            },
                            onCloseAndClear = {
                                scanViewModel.clearSession()
                                navController.popBackStack()
                            },
                        )
                    }
                }
            }
        }
    }
}
