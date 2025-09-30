package com.snowleopard.docscanner

import android.app.Application
import com.snowleopard.docscanner.di.appModule
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.startKoin

class DocScannerApp : Application() {
    override fun onCreate() {
        super.onCreate()
        startKoin {
            androidContext(this@DocScannerApp)
            modules(appModule)
        }
    }
}