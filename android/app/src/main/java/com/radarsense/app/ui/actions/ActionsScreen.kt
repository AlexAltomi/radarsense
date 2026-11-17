package com.radarsense.app.ui.actions

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.radarsense.app.domain.ActionDispatcher
import com.radarsense.app.domain.ActionState

/** The deck, the play/pause indicator and the two lights, all driven by [ActionState]. Plain on purpose. */
@Composable
fun ActionsScreen(state: ActionState, modifier: Modifier = Modifier) {
    Column(modifier = modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text("Slide ${state.slide} of ${ActionDispatcher.SLIDE_COUNT}", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (number in 1..ActionDispatcher.SLIDE_COUNT) {
                val current = number == state.slide
                Box(
                    Modifier
                        .size(width = 48.dp, height = 36.dp)
                        .clip(RoundedCornerShape(4.dp))
                        .background(if (current) Color(0xFF1565C0) else Color(0xFFBDBDBD)),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("$number", color = if (current) Color.White else Color.Black)
                }
            }
        }
        Text(if (state.playing) "Playing" else "Paused", style = MaterialTheme.typography.titleLarge)
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp), verticalAlignment = Alignment.CenterVertically) {
            state.lights.forEachIndexed { index, on ->
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .size(32.dp)
                            .clip(CircleShape)
                            .background(if (on) Color(0xFFFFD600) else Color(0xFF424242)),
                    )
                    Text("Light ${index + 1}: ${if (on) "on" else "off"}")
                }
            }
        }
    }
}
