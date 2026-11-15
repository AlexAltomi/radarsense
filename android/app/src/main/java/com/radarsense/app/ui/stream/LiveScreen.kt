package com.radarsense.app.ui.stream

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.radarsense.app.domain.Frame
import kotlinx.coroutines.flow.StateFlow

@Composable
fun LiveScreen(
    state: StreamUiState,
    frames: StateFlow<Frame?>,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Heatmap(frames, Modifier.fillMaxWidth())
        Text("Doppler across, range up (near at the bottom). Channel 0 only.", style = MaterialTheme.typography.bodySmall)
        Text("Expected: ${state.expected}", style = MaterialTheme.typography.headlineMedium)
        Text("Status: ${statusText(state.status)}")
        Text("%.1f fps · %d frames · %d buffer resets after lost frames".format(
            state.fps, state.framesReceived, state.lostFrameResets,
        ))
    }
}

/**
 * Draws channel 0 of the newest frame. The bitmap and the colour array are created once; each frame only refills
 * them. The frame is read inside the draw block, so a new frame redraws the canvas without recomposing the screen.
 */
@Composable
fun Heatmap(frames: StateFlow<Frame?>, modifier: Modifier = Modifier) {
    val renderer = remember { HeatmapRenderer() }
    val frame by frames.collectAsStateWithLifecycle()
    Canvas(modifier.aspectRatio(1f)) {
        frame?.let(renderer::update)
        drawImage(
            image = renderer.image,
            dstSize = IntSize(size.width.toInt(), size.height.toInt()),
            filterQuality = FilterQuality.None, // sharp cells, no blur
        )
    }
}

private class HeatmapRenderer {
    private val bitmap = Bitmap.createBitmap(Frame.SIZE, Frame.SIZE, Bitmap.Config.ARGB_8888)
    private val argb = IntArray(Frame.CHANNEL_PIXELS)
    private var last: Frame? = null
    val image: ImageBitmap = bitmap.asImageBitmap()

    init {
        bitmap.eraseColor(HeatmapPalette.lut[0])
    }

    fun update(frame: Frame) {
        if (frame === last) return
        HeatmapPalette.fillChannel0(frame.pixels, argb)
        bitmap.setPixels(argb, 0, Frame.SIZE, 0, 0, Frame.SIZE, Frame.SIZE)
        last = frame
    }
}
