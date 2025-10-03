package com.snowleopard.docscanner.feature.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.snowleopard.docscanner.core.data.model.PdfOptions
import com.snowleopard.docscanner.core.data.repository.PagesRepository
import com.snowleopard.docscanner.core.data.model.ScannedPage
import com.snowleopard.docscanner.core.data.store.PagesStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class StackViewModel(
    private val repo: PagesRepository,
    private val store: PagesStore
) : ViewModel() {

    val pages: StateFlow<List<ScannedPage>> = store.pages

    fun movePage(from: Int, to: Int) {
        store.move(from, to)
    }

    fun deletePageAt(index: Int) {
        val removed = store.removeAt(index) ?: return
        viewModelScope.launch(Dispatchers.IO) { repo.deletePage(removed) }
    }

    fun rotatePage(index: Int, degrees: Int) {
        val list = pages.value
        if (index !in list.indices) return
        val page = list[index]
        viewModelScope.launch(Dispatchers.IO) {
            val updated = repo.rotatePage(page, degrees)
            withContext(Dispatchers.Main) { store.updateAt(index, updated) }
        }
    }

    fun clearSession() {
        viewModelScope.launch(Dispatchers.IO) {
            repo.clearSession()
            withContext(Dispatchers.Main) { store.clear() }
        }
    }

    suspend fun buildPdf(options: PdfOptions): File {
        val current = pages.value
        return repo.buildPdf(current, options)
    }
}
