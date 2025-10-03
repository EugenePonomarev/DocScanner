package com.snowleopard.docscanner.core.data.store

import com.snowleopard.docscanner.core.data.model.ScannedPage
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * In-memory pages store shared by Scan & Stack view models.
 * Single source of truth to avoid duplicating list state in multiple VMs.
 */
class PagesStore {

    private val _pages = MutableStateFlow<List<ScannedPage>>(emptyList())
    val pages: StateFlow<List<ScannedPage>> = _pages

    fun set(pages: List<ScannedPage>) {
        _pages.value = pages
    }

    fun add(page: ScannedPage) {
        _pages.value = _pages.value + page
    }

    fun removeLast(): ScannedPage? {
        val list = _pages.value
        if (list.isEmpty()) return null
        val last = list.last()
        _pages.value = list.dropLast(1)
        return last
    }

    fun removeAt(index: Int): ScannedPage? {
        val list = _pages.value.toMutableList()
        if (index !in list.indices) return null
        val removed = list.removeAt(index)
        _pages.value = list
        return removed
    }

    fun updateAt(index: Int, page: ScannedPage) {
        val list = _pages.value.toMutableList()
        if (index !in list.indices) return
        list[index] = page
        _pages.value = list
    }

    fun move(from: Int, to: Int) {
        val list = _pages.value.toMutableList()
        if (from !in list.indices || to !in list.indices) return
        val it = list.removeAt(from)
        list.add(to, it)
        _pages.value = list
    }

    fun clear() {
        _pages.value = emptyList()
    }
}