package com.radarsense.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.radarsense.app.ui.theme.RadarSenseTheme
import java.io.IOException
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val api = ApiClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            RadarSenseTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    var result by remember { mutableStateOf("not called yet") }
                    val scope = rememberCoroutineScope()
                    Column(Modifier.padding(innerPadding).padding(24.dp)) {
                        Button(onClick = {
                            scope.launch {
                                result = try {
                                    api.health()
                                } catch (e: IOException) {
                                    "error: ${e::class.simpleName}: ${e.message}"
                                }
                            }
                        }) { Text("Check server") }
                        Text(result)
                    }
                }
            }
        }
    }
}
