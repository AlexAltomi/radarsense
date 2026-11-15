package com.radarsense.app.ui.stream

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.radarsense.app.data.net.FrameStreamClient
import com.radarsense.app.data.net.ServerMessage
import com.radarsense.app.data.net.StartRequest
import com.radarsense.app.data.net.StreamEvent
import com.radarsense.app.data.net.StreamProtocol
import com.radarsense.app.domain.Frame
import com.radarsense.app.domain.FrameBuffer
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.retryWhen
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

sealed interface StreamStatus {
    data object Idle : StreamStatus
    data object Connecting : StreamStatus
    data object Streaming : StreamStatus
    data class Reconnecting(val attempt: Long, val reason: String) : StreamStatus
    data class Failed(val message: String) : StreamStatus
}

data class StreamUiState(
    val serverUrl: String = StreamProtocol.DEFAULT_URL,
    val user: String = "any",
    val gesture: String = "any",
    val status: StreamStatus = StreamStatus.Idle,
    val classes: List<String> = emptyList(),
    /** Expected gesture of the newest frame, from the header's true label. */
    val expected: String = "-",
    val fps: Double = 0.0,
    val framesReceived: Long = 0,
    val lostFrameResets: Int = 0,
)

/**
 * Shared by the Connect and Live screens. Owns the stream and the frame buffer.
 *
 * The stream is collected on Dispatchers.Default. The frame buffer is touched only from that coroutine. The UI
 * reads [state], which changes a few times per second, and [latestFrame], which changes on every frame.
 */
class StreamViewModel(
    private val client: FrameStreamClient = FrameStreamClient(),
) : ViewModel() {
    private val _state = MutableStateFlow(StreamUiState())
    val state: StateFlow<StreamUiState> = _state.asStateFlow()

    private val _latestFrame = MutableStateFlow<Frame?>(null)
    val latestFrame: StateFlow<Frame?> = _latestFrame.asStateFlow()

    /** The newest 32 frames. Week 6 runs the classifier on it. */
    private val buffer = FrameBuffer()
    private var job: Job? = null

    // Frame rate, measured over about one second.
    private var rateWindowStart = 0L
    private var rateWindowFrames = 0
    private var framesReceived = 0L

    fun setServerUrl(value: String) = _state.update { it.copy(serverUrl = value) }
    fun setUser(value: String) = _state.update { it.copy(user = value) }
    fun setGesture(value: String) = _state.update { it.copy(gesture = value) }

    fun start() {
        val settings = _state.value
        val request = StartRequest(
            user = settings.user.trim().ifEmpty { "any" },
            gesture = settings.gesture.trim().ifEmpty { "any" },
            loop = true,
        )
        val url = settings.serverUrl.trim()
        val previous = job
        job = viewModelScope.launch(Dispatchers.Default) {
            previous?.cancelAndJoin()
            resetCounters()
            _state.update { it.copy(status = StreamStatus.Connecting, expected = "-", fps = 0.0) }
            client.stream(url, request)
                .retryWhen { cause, attempt ->
                    if (cause !is IOException) return@retryWhen false
                    val reason = cause.message ?: cause::class.java.simpleName
                    _state.update { it.copy(status = StreamStatus.Reconnecting(attempt + 1, reason), fps = 0.0) }
                    delay(StreamProtocol.reconnectDelayMs(attempt))
                    true
                }
                .catch { e ->
                    val message = e.message ?: e::class.java.simpleName
                    _state.update { it.copy(status = StreamStatus.Failed(message), fps = 0.0) }
                }
                .collect { onEvent(it) }
        }
    }

    fun stop() {
        val running = job ?: return
        job = null
        viewModelScope.launch {
            running.cancelAndJoin() // cancelling the collection sends "stop" and closes the socket
            _state.update { it.copy(status = StreamStatus.Idle, fps = 0.0, expected = "-") }
        }
    }

    private fun resetCounters() {
        buffer.clear()
        framesReceived = 0
        rateWindowStart = 0L
        rateWindowFrames = 0
        _state.update { it.copy(framesReceived = 0, lostFrameResets = 0) }
    }

    private fun onEvent(event: StreamEvent) {
        when (event) {
            is StreamEvent.Connected -> Unit // the session message follows
            is StreamEvent.FrameReceived -> onFrame(event.frame)
            is StreamEvent.Message -> when (val m = event.message) {
                is ServerMessage.Session ->
                    _state.update { it.copy(status = StreamStatus.Streaming, classes = m.classes) }
                is ServerMessage.Error ->
                    _state.update { it.copy(status = StreamStatus.Failed("Server: ${m.message}")) }
                ServerMessage.Ended, ServerMessage.Stopped ->
                    _state.update { it.copy(status = StreamStatus.Idle, fps = 0.0) }
                is ServerMessage.Unknown -> Unit
            }
        }
    }

    private fun onFrame(frame: Frame) {
        buffer.add(frame)
        framesReceived++
        _latestFrame.value = frame

        val expected = labelName(frame.label)
        if (expected != _state.value.expected) _state.update { it.copy(expected = expected) }

        if (rateWindowStart == 0L) rateWindowStart = frame.receivedAtNanos
        rateWindowFrames++
        val elapsed = frame.receivedAtNanos - rateWindowStart
        if (elapsed >= 1_000_000_000L) {
            val fps = rateWindowFrames * 1e9 / elapsed
            _state.update {
                it.copy(fps = fps, framesReceived = framesReceived, lostFrameResets = buffer.lostFrameResets)
            }
            rateWindowStart = frame.receivedAtNanos
            rateWindowFrames = 0
        }
    }

    private fun labelName(label: Int): String =
        if (label == Frame.UNKNOWN_LABEL) "unknown" else _state.value.classes.getOrNull(label) ?: "class $label"
}
