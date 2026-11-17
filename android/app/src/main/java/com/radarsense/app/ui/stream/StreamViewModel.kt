package com.radarsense.app.ui.stream

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.radarsense.app.data.net.FrameSource
import com.radarsense.app.data.net.FrameStreamClient
import com.radarsense.app.data.net.ServerMessage
import com.radarsense.app.data.net.StartRequest
import com.radarsense.app.data.net.StreamEvent
import com.radarsense.app.data.net.StreamProtocol
import com.radarsense.app.domain.Frame
import com.radarsense.app.domain.FrameBuffer
import com.radarsense.app.domain.GestureClassifier
import com.radarsense.app.domain.GestureDetector
import com.radarsense.app.domain.ModelClasses
import com.radarsense.app.domain.WindowDecoder
import java.io.IOException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
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
    /** Class name of the newest prediction (neutral model-card names), or "-" before the first one. */
    val prediction: String = "-",
    /** Softmax probability of the newest prediction, 0..1. */
    val confidence: Float = 0f,
    /** Class name of the gesture that fired last, or "-". */
    val fired: String = "-",
    /** How many times a gesture has fired since the stream started. Lets the UI see a repeat of the same gesture. */
    val fireCount: Int = 0,
    /** Time from the arrival of the window-completing frame to the detector firing, for the last fire. Null if none. */
    val latencyMs: Double? = null,
    /** Windows that were due while the classifier was still busy and were skipped. */
    val droppedWindows: Long = 0,
    /** Windows that the classifier has finished. */
    val windowsClassified: Long = 0,
)

/**
 * Shared by the Connect and Live screens. Owns the stream, the frame buffer and the classification of windows.
 *
 * Two coroutines work side by side:
 * - Ingestion (the stream collector) reads frames, fills the frame buffer, and every 4th frame after the buffer is
 *   full it decodes the newest window and hands it over. It never waits for the classifier.
 * - Inference takes the handed-over window, runs the classifier, and gives the probabilities to the detector.
 *   It owns the detector. It never runs on the main thread.
 *
 * Only one window can be in the classifier at a time. If a window is due while the classifier is still busy, that
 * window is dropped and counted. Windows are never queued, so latency cannot grow without bound.
 *
 * The UI reads [state], which changes a few times per second, and [latestFrame], which changes on every frame.
 */
class StreamViewModel(
    /** Makes the classifier. Called on the inference thread when a stream starts, and the result is closed when it ends. */
    private val classifierProvider: () -> GestureClassifier,
    private val source: FrameSource = FrameStreamClient(),
    /** Nanosecond clock for the detector's cooldown and for latency. Tests pass a fake one. */
    private val clock: () -> Long = System::nanoTime,
    private val ingestDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val inferenceDispatcher: CoroutineDispatcher = Dispatchers.Default,
) : ViewModel() {
    private val _state = MutableStateFlow(StreamUiState())
    val state: StateFlow<StreamUiState> = _state.asStateFlow()

    private val _latestFrame = MutableStateFlow<Frame?>(null)
    val latestFrame: StateFlow<Frame?> = _latestFrame.asStateFlow()

    /** The newest 32 frames. Touched only from the ingestion coroutine. */
    private val buffer = FrameBuffer()
    private var job: Job? = null

    // Frame rate, measured over about one second.
    private var rateWindowStart = 0L
    private var rateWindowFrames = 0
    private var framesReceived = 0L

    // Hand-over between ingestion and inference.
    private val decoder = WindowDecoder() // ingestion writes decoder.output, but only while inference is idle
    private val busy = AtomicBoolean(false) // true from hand-over until the classifier is done with the window
    private val work = Channel<Unit>(1) // "a window is ready"; never holds more than one, because of [busy]
    private val clearEpoch = AtomicInteger(0) // goes up every time the frame buffer is emptied
    @Volatile private var windowEpoch = 0 // clearEpoch when the window was decoded
    @Volatile private var windowReceivedAtNanos = 0L // arrival time of the frame that completed the window
    private var droppedWindows = 0L // ingestion only

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
        job = viewModelScope.launch(ingestDispatcher) {
            previous?.cancelAndJoin()
            resetCounters()
            _state.update { it.copy(status = StreamStatus.Connecting, expected = "-", fps = 0.0) }
            try {
                coroutineScope {
                    val inference = launch(inferenceDispatcher) { runInference() }
                    source.stream(url, request)
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
                    inference.cancel()
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // For example the model file could not be loaded. Without a classifier the stream is no use.
                val message = e.message ?: e::class.java.simpleName
                _state.update { it.copy(status = StreamStatus.Failed("Classifier: $message"), fps = 0.0) }
            }
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
        clearEpoch.incrementAndGet()
        while (work.tryReceive().isSuccess) Unit // forget a window handed over by an earlier stream
        busy.set(false)
        droppedWindows = 0
        framesReceived = 0
        rateWindowStart = 0L
        rateWindowFrames = 0
        _state.update {
            it.copy(
                framesReceived = 0, lostFrameResets = 0,
                prediction = "-", confidence = 0f, fired = "-", fireCount = 0, latencyMs = null,
                droppedWindows = 0, windowsClassified = 0,
            )
        }
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

    /** Ingestion. Runs for every frame, so apart from the occasional state update it allocates nothing. */
    private fun onFrame(frame: Frame) {
        // A cleared buffer means the next window holds frames from a new clip. Inference resets the detector when
        // it sees the new epoch; it owns the detector, so it is not touched from here.
        if (buffer.add(frame)) clearEpoch.incrementAndGet()
        framesReceived++
        if (buffer.isWindowDue()) handOverWindow(frame)
        _latestFrame.value = frame // last, so a reader that sees this frame knows it was fully handled

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

    /** A window is due. Gives it to inference, or drops it if the classifier is still busy with the last one. */
    private fun handOverWindow(completingFrame: Frame) {
        if (!busy.compareAndSet(false, true)) {
            droppedWindows++
            val dropped = droppedWindows
            _state.update { it.copy(droppedWindows = dropped) }
            return
        }
        decoder.decode(buffer) // safe: inference is idle, so nobody reads decoder.output now
        windowEpoch = clearEpoch.get()
        windowReceivedAtNanos = completingFrame.receivedAtNanos
        if (!work.trySend(Unit).isSuccess) busy.set(false) // cannot happen: at most one window is in flight
    }

    /** Inference. Takes windows one at a time, runs the classifier and feeds the detector. */
    private suspend fun runInference() {
        val detector = GestureDetector() // the project's settings: 0.8, 2 in a row, 600 ms cooldown
        var appliedEpoch = clearEpoch.get()
        classifierProvider().use { classifier ->
            for (windowReady in work) {
                val epoch = windowEpoch
                val receivedAt = windowReceivedAtNanos
                val prediction = try {
                    classifier.classify(decoder.output) // probabilities, after softmax
                } finally {
                    busy.set(false) // the window is not read any more; the next one may be handed over
                }
                val now = clock()

                // The buffer was emptied since the last window: the detector forgets its old predictions.
                val currentEpoch = clearEpoch.get()
                if (currentEpoch != appliedEpoch) {
                    detector.reset()
                    appliedEpoch = currentEpoch
                }
                // The buffer was emptied while this window was in the classifier. Its frames belong to the old
                // clip, so its result is thrown away and does not reach the detector.
                if (epoch != currentEpoch) continue

                val firedClass = detector.onPrediction(prediction.classIndex, prediction.confidence, now)
                val latencyMs = if (firedClass != null) (clock() - receivedAt) / 1e6 else null
                _state.update {
                    it.copy(
                        prediction = ModelClasses.name(prediction.classIndex),
                        confidence = prediction.confidence,
                        windowsClassified = it.windowsClassified + 1,
                        fired = if (firedClass != null) ModelClasses.name(firedClass) else it.fired,
                        fireCount = if (firedClass != null) it.fireCount + 1 else it.fireCount,
                        latencyMs = latencyMs ?: it.latencyMs,
                    )
                }
            }
        }
    }

    private fun labelName(label: Int): String =
        if (label == Frame.UNKNOWN_LABEL) "unknown" else _state.value.classes.getOrNull(label) ?: "class $label"
}
