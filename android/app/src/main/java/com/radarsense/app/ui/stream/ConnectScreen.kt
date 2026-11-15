package com.radarsense.app.ui.stream

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ConnectScreen(
    state: StreamUiState,
    onServerUrlChange: (String) -> Unit,
    onUserChange: (String) -> Unit,
    onGestureChange: (String) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val running = state.status != StreamStatus.Idle
    Column(
        modifier = modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Replay source", style = MaterialTheme.typography.titleLarge)
        OutlinedTextField(
            value = state.serverUrl,
            onValueChange = onServerUrlChange,
            label = { Text("Server URL") },
            singleLine = true,
            enabled = !running,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.user,
            onValueChange = onUserChange,
            label = { Text("User") },
            supportingText = { Text("\"any\" plays the validation users, or name one, e.g. user3") },
            singleLine = true,
            enabled = !running,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = state.gesture,
            onValueChange = onGestureChange,
            label = { Text("Gesture") },
            supportingText = { Text("\"any\", \"background\" or a class name, e.g. label2") },
            singleLine = true,
            enabled = !running,
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = onStart) { Text(if (running) "Restart" else "Start") }
            OutlinedButton(onClick = onStop, enabled = running) { Text("Stop") }
        }
        Text("Status: ${statusText(state.status)}")
        if (state.classes.isNotEmpty()) {
            Text("Classes from the server: ${state.classes.joinToString()}")
        }
    }
}

fun statusText(status: StreamStatus): String = when (status) {
    StreamStatus.Idle -> "stopped"
    StreamStatus.Connecting -> "connecting"
    StreamStatus.Streaming -> "streaming"
    is StreamStatus.Reconnecting -> "reconnecting (attempt ${status.attempt}): ${status.reason}"
    is StreamStatus.Failed -> "error: ${status.message}"
}
