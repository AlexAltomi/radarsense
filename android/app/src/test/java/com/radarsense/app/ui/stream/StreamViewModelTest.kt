package com.radarsense.app.ui.stream

import com.radarsense.app.data.net.FrameSource
import com.radarsense.app.data.net.StartRequest
import com.radarsense.app.data.net.StreamEvent
import com.radarsense.app.domain.Frame
import com.radarsense.app.domain.GestureClassifier
import com.radarsense.app.domain.Prediction
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Tests the wiring in [StreamViewModel] with a fake frame source and a fake classifier. The detector itself is
 * tested in GestureDetectorTest; here the question is whether the view model feeds it correctly.
 *
 * Frames go in one at a time. After every frame that completes a window, the test waits until the window has been
 * classified (or dropped), so the results do not depend on thread timing.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class StreamViewModelTest {
    private val none = 0
    private val gestureA = 2 // "label2"

    private class FakeSource : FrameSource {
        val events = Channel<StreamEvent>(Channel.UNLIMITED)
        override fun stream(url: String, start: StartRequest): Flow<StreamEvent> = events.receiveAsFlow()
    }

    /** Returns the scripted predictions in order, and the last one again after that. */
    private class FakeClassifier(private val script: List<Prediction>) : GestureClassifier {
        val calls = AtomicInteger(0)
        @Volatile var closed = false
        @Volatile var gate: CountDownLatch? = null // when set, classify() waits for it
        val entered = CountDownLatch(1)
        @Volatile var atEntry: FloatArray? = null // copy of the input when classify() started
        @Volatile var atExit: FloatArray? = null // copy of the input just before classify() returned

        override fun classify(window: FloatArray): Prediction {
            val index = calls.getAndIncrement()
            if (index == 0) atEntry = window.copyOf()
            entered.countDown()
            gate?.await(10, TimeUnit.SECONDS)
            if (index == 0) atExit = window.copyOf()
            return script[minOf(index, script.lastIndex)]
        }

        override fun close() {
            closed = true
        }
    }

    private fun prediction(classIndex: Int, confidence: Float = 0.9f) =
        Prediction(classIndex, confidence, FloatArray(GestureClassifier.NUM_CLASSES))

    private val baseNanos = 1_000_000_000L
    private val clockNanos = AtomicLong(0)
    private val source = FakeSource()
    private var nextSeq = 0L
    private var framesSinceClear = 0
    private var windowsDue = 0L
    private lateinit var classifier: FakeClassifier
    private lateinit var vm: StreamViewModel

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
    }

    @After
    fun tearDown() {
        classifier.gate?.countDown()
        vm.stop()
        Dispatchers.resetMain()
    }

    private fun startWith(vararg script: Prediction) {
        classifier = FakeClassifier(script.toList())
        vm = StreamViewModel(classifierProvider = { classifier }, source = source, clock = clockNanos::get)
        vm.start()
    }

    private fun waitUntil(what: String, condition: () -> Boolean) {
        val deadline = System.nanoTime() + 5_000_000_000L
        while (!condition()) {
            if (System.nanoTime() > deadline) throw AssertionError("Timed out waiting for: $what")
            Thread.sleep(2)
        }
    }

    /** Sends one frame and waits until the view model has handled it. */
    private fun send(clipStart: Boolean = false) {
        val seq = nextSeq++
        val frame = Frame(
            seq = seq, serverTimeMs = 0L, label = Frame.UNKNOWN_LABEL,
            flags = if (clipStart) Frame.FLAG_CLIP_START else 0,
            // Every frame has its own pixel value, so a window decoded from other frames would look different.
            pixels = ByteArray(Frame.PAYLOAD_BYTES) { ((seq % 250) + 1).toByte() },
            receivedAtNanos = baseNanos + seq * 1_000_000L, // one millisecond between frames
        )
        if (clipStart) framesSinceClear = 0
        framesSinceClear++
        if (framesSinceClear >= 32 && (framesSinceClear - 32) % 4 == 0) windowsDue++
        source.events.trySend(StreamEvent.FrameReceived(frame))
        waitUntil("frame $seq handled") { vm.latestFrame.value === frame }
    }

    /** Sends frames until one window is due, then waits until it is classified or dropped. Returns the clock to use. */
    private fun sendWindow(clockAtDecision: Long) {
        clockNanos.set(clockAtDecision)
        val target = windowsDue + 1
        while (windowsDue < target) send()
        waitUntil("window $target classified or dropped") {
            vm.state.value.windowsClassified + vm.state.value.droppedWindows >= target
        }
    }

    private val state get() = vm.state.value

    @Test
    fun resetsTheDetectorWhenTheFrameBufferIsCleared() {
        // Without a reset, two windows of gesture A in a row would fire. A new clip between them must stop that.
        startWith(prediction(gestureA), prediction(gestureA))
        sendWindow(baseNanos) // first window: gesture A, no fire yet
        assertEquals(0, state.fireCount)

        send(clipStart = true) // the buffer is cleared; 31 more frames make the next window
        repeat(31) { send() }
        waitUntil("second window classified") { state.windowsClassified >= 2 }

        assertEquals("both windows were classified", 2, classifier.calls.get())
        assertEquals("no fire: the detector forgot the first prediction", 0, state.fireCount)
        assertEquals("-", state.fired)
    }

    @Test
    fun withoutAClearTheSamePredictionsDoFire() {
        // The control for the test above.
        startWith(prediction(gestureA), prediction(gestureA))
        sendWindow(baseNanos)
        sendWindow(baseNanos)
        assertEquals(1, state.fireCount)
    }

    @Test
    fun aWindowDueWhileTheClassifierIsBusyIsDroppedCountedAndNotQueued() {
        startWith(prediction(none))
        classifier.gate = CountDownLatch(1)

        repeat(32) { send() } // the first window goes to the classifier, which now waits at the gate
        assertTrue(classifier.entered.await(5, TimeUnit.SECONDS))
        repeat(4) { send() } // a window is due: dropped
        repeat(4) { send() } // another: dropped
        assertEquals(2L, state.droppedWindows)
        assertEquals("only the first window reached the classifier", 1, classifier.calls.get())

        classifier.gate!!.countDown() // the classifier finishes the first window
        waitUntil("first window classified") { state.windowsClassified >= 1 }
        Thread.sleep(150) // time for a queued window to show up, if there were one
        assertEquals("dropped windows were not queued", 1, classifier.calls.get())
        assertEquals(1L, state.windowsClassified)
        assertEquals(2L, state.droppedWindows)

        // The classifier is free again, so the next window is classified.
        sendWindow(baseNanos)
        assertEquals(2, classifier.calls.get())
        assertEquals(2L, state.droppedWindows)
    }

    @Test
    fun theClassifiersInputIsNotOverwrittenWhileItIsBlocked() {
        startWith(prediction(none))
        classifier.gate = CountDownLatch(1)

        repeat(32) { send() } // the first window is handed over; the classifier blocks inside classify()
        assertTrue(classifier.entered.await(5, TimeUnit.SECONDS))
        val before = classifier.atEntry!!
        assertTrue("the test needs a window with different values in it", before.distinct().size > 1)

        repeat(12) { send() } // three more windows are due while the classifier is blocked
        assertEquals(3L, state.droppedWindows)
        assertEquals("nothing was queued", 1, classifier.calls.get())

        classifier.gate!!.countDown()
        waitUntil("classifier finished the first window") { classifier.atExit != null && state.windowsClassified >= 1 }
        assertTrue(
            "the array changed while the classifier was reading it",
            before.contentEquals(classifier.atExit!!),
        )
        Thread.sleep(150)
        assertEquals("dropped windows were never classified", 1, classifier.calls.get())
        assertEquals(3L, state.droppedWindows)
    }

    @Test
    fun aPredictionFromBeforeANewSessionDoesNotCountTowardFiringInTheNextOne() {
        startWith(prediction(gestureA), prediction(gestureA))
        sendWindow(baseNanos) // session 1: one prediction of gesture A, no fire
        assertEquals(0, state.fireCount)

        vm.start() // session 2: the counters and the detector start from zero
        waitUntil("state reset") { state.windowsClassified == 0L && state.prediction == "-" }
        framesSinceClear = 0
        windowsDue = 0
        sendWindow(baseNanos) // session 2: one prediction of gesture A. Alone, it must not fire.

        assertEquals("both windows were classified", 2, classifier.calls.get())
        assertEquals("the old prediction was counted", 0, state.fireCount)
        assertEquals("-", state.fired)
    }

    @Test
    fun firesAfterTwoAgreeingPredictionsAtOrAboveTheThreshold() {
        startWith(prediction(gestureA, 0.9f), prediction(gestureA, 0.8f))
        sendWindow(baseNanos)
        assertEquals("label2", state.prediction)
        assertEquals(0, state.fireCount)

        // The second window completes with frame seq 35, which arrived at baseNanos + 35 ms. The detector fires
        // 50 ms later by the fake clock.
        sendWindow(baseNanos + 35_000_000L + 50_000_000L)
        assertEquals(1, state.fireCount)
        assertEquals("label2", state.fired)
        assertNotNull(state.latencyMs)
        assertEquals(50.0, state.latencyMs!!, 0.001)
        assertEquals(0.8f, state.confidence, 0f)
    }

    @Test
    fun doesNotFireBelowTheThreshold() {
        startWith(prediction(gestureA, 0.9f), prediction(gestureA, 0.79f))
        sendWindow(baseNanos)
        sendWindow(baseNanos)
        assertEquals(0, state.fireCount)
    }

    @Test
    fun noSecondFireInsideTheCooldown() {
        startWith(prediction(gestureA), prediction(gestureA), prediction(gestureA), prediction(gestureA))
        sendWindow(baseNanos)
        sendWindow(baseNanos) // fires
        assertEquals(1, state.fireCount)
        sendWindow(baseNanos + 100_000_000L) // 100 ms later: inside the 600 ms cooldown
        sendWindow(baseNanos + 200_000_000L)
        assertEquals(1, state.fireCount)
        assertEquals(4L, state.windowsClassified)
    }

    @Test
    fun armsAgainAfterANonePrediction() {
        startWith(
            prediction(gestureA), prediction(gestureA), // fires
            prediction(none), // after the cooldown: re-arms
            prediction(gestureA), prediction(gestureA), // fires again
        )
        sendWindow(baseNanos)
        sendWindow(baseNanos)
        assertEquals(1, state.fireCount)
        sendWindow(baseNanos + 700_000_000L) // cooldown (600 ms) is over
        sendWindow(baseNanos + 800_000_000L)
        assertEquals("one gesture is not enough right after re-arming", 1, state.fireCount)
        sendWindow(baseNanos + 900_000_000L)
        assertEquals(2, state.fireCount)
        assertEquals("label2", state.fired)
    }

    @Test
    fun theClassifierIsClosedWhenTheStreamStops() {
        startWith(prediction(none))
        sendWindow(baseNanos)
        vm.stop()
        waitUntil("classifier closed") { classifier.closed }
    }
}
