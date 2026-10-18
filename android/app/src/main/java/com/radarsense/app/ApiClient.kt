package com.radarsense.app

import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request

class ApiClient(private val baseUrl: String = "http://10.0.2.2:8080") {
    private val http = OkHttpClient.Builder()
        .callTimeout(15, TimeUnit.SECONDS)
        .build()

    @Throws(IOException::class)
    suspend fun health(): String = withContext(Dispatchers.IO) {
        val req = Request.Builder().url("$baseUrl/api/health").build()
        http.newCall(req).execute().use { it.body?.string() ?: "empty body" }
    }
}
