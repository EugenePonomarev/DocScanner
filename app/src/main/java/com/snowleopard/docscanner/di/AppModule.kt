package com.snowleopard.docscanner.di

import com.snowleopard.docscanner.PagesRepository
import com.snowleopard.docscanner.ScanViewModel
import org.koin.dsl.module
import org.koin.android.ext.koin.androidContext
import org.koin.core.module.dsl.viewModel

val appModule = module {
    single { PagesRepository(androidContext()) }
    viewModel { ScanViewModel(get()) }
}