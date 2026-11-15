package com.radarsense.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.radarsense.app.ui.stream.ConnectScreen
import com.radarsense.app.ui.stream.LiveScreen
import com.radarsense.app.ui.stream.StreamViewModel
import com.radarsense.app.ui.theme.RadarSenseTheme

class MainActivity : ComponentActivity() {
    private val streamViewModel: StreamViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            RadarSenseTheme {
                RadarSenseApp(streamViewModel)
            }
        }
    }
}

private enum class Screen(val title: String) { Connect("Connect"), Live("Live") }

/** Two screens for Week 5. Navigation Compose comes with History and Settings in Week 7. */
@Composable
private fun RadarSenseApp(viewModel: StreamViewModel) {
    var screen by rememberSaveable { mutableStateOf(Screen.Connect) }
    val state by viewModel.state.collectAsStateWithLifecycle()

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Column(Modifier.padding(innerPadding)) {
            Row(Modifier.padding(horizontal = 16.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (s in Screen.entries) {
                    if (s == screen) {
                        Button(onClick = {}) { Text(s.title) }
                    } else {
                        OutlinedButton(onClick = { screen = s }) { Text(s.title) }
                    }
                }
            }
            when (screen) {
                Screen.Connect -> ConnectScreen(
                    state = state,
                    onServerUrlChange = viewModel::setServerUrl,
                    onUserChange = viewModel::setUser,
                    onGestureChange = viewModel::setGesture,
                    onStart = {
                        viewModel.start()
                        screen = Screen.Live
                    },
                    onStop = viewModel::stop,
                )
                Screen.Live -> LiveScreen(state = state, frames = viewModel.latestFrame)
            }
        }
    }
}
