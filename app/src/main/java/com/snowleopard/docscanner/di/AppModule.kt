package com.snowleopard.docscanner.di

import com.snowleopard.docscanner.core.data.repository.PagesRepository
import com.snowleopard.docscanner.feature.viewmodels.ScanViewModel
import com.snowleopard.docscanner.core.data.store.PagesStore
import com.snowleopard.docscanner.feature.viewmodels.StackViewModel
import org.koin.android.ext.koin.androidContext
import org.koin.core.module.dsl.singleOf
import org.koin.core.module.dsl.viewModel
import org.koin.dsl.module

val dataModule = module {
    single { PagesRepository(androidContext()) }
    singleOf(::PagesStore)
}

val viewModelModule = module {
    viewModel { ScanViewModel(repo = get(), store = get()) }
    viewModel { StackViewModel(repo = get(), store = get()) }
}

val appModule = module {
    includes(dataModule, viewModelModule)
}