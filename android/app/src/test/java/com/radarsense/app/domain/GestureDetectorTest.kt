package com.radarsense.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GestureDetectorTest {
    private val g = 2 // a gesture class
    private val h = 3 // another gesture class
    private val none = 0

    private fun ms(value: Long) = value * 1_000_000L

    @Test
    fun firesOnTwoEqualConfidentPredictions() {
        val d = GestureDetector()
        assertNull(d.onPrediction(g, 0.95f, ms(0)))
        assertEquals(g, d.onPrediction(g, 0.95f, ms(100)))
    }

    @Test
    fun doesNotFireOnTwoDifferentGestures() {
        val d = GestureDetector()
        assertNull(d.onPrediction(g, 0.95f, ms(0)))
        assertNull(d.onPrediction(h, 0.95f, ms(100)))
        assertNull(d.onPrediction(g, 0.95f, ms(200)))
    }

    @Test
    fun doesNotFireBelowTheThreshold() {
        val d = GestureDetector()
        assertNull(d.onPrediction(g, 0.79f, ms(0)))
        assertNull(d.onPrediction(g, 0.79f, ms(100)))
    }

    @Test
    fun oneWeakPredictionInThePairIsEnoughToStopIt() {
        val d = GestureDetector()
        assertNull(d.onPrediction(g, 0.79f, ms(0)))
        assertNull(d.onPrediction(g, 0.95f, ms(100))) // the earlier one was weak
        assertNull(d.onPrediction(g, 0.79f, ms(200))) // this one is weak
        assertNull(d.onPrediction(g, 0.95f, ms(300))) // the pair is (weak, confident)
        assertEquals(g, d.onPrediction(g, 0.95f, ms(400))) // now the last two are both confident
    }

    @Test
    fun theThresholdItselfIsEnough() {
        val d = GestureDetector()
        assertNull(d.onPrediction(g, 0.8f, ms(0)))
        assertEquals(g, d.onPrediction(g, 0.8f, ms(100)))
    }

    @Test
    fun noneNeverFires() {
        val d = GestureDetector()
        for (t in 0L until 10L) assertNull(d.onPrediction(none, 0.99f, ms(t * 100)))
    }

    @Test
    fun firesOnlyOnceForARunOfTheSameGesture() {
        val d = GestureDetector()
        val fired = (0L until 4L).map { d.onPrediction(g, 0.95f, ms(it * 100)) }
        assertEquals(listOf(null, g, null, null), fired)
    }

    @Test
    fun ignoresEverythingDuringTheCooldown() {
        val d = GestureDetector()
        d.onPrediction(g, 0.95f, ms(0))
        assertEquals(g, d.onPrediction(g, 0.95f, ms(100))) // fires at 100 ms, the cooldown ends at 700 ms
        assertNull(d.onPrediction(h, 0.95f, ms(200)))
        assertNull(d.onPrediction(h, 0.95f, ms(300)))
        assertNull(d.onPrediction(none, 0.95f, ms(400))) // a "none" inside the cooldown does not re-arm
        assertNull(d.onPrediction(h, 0.95f, ms(500)))
        assertNull(d.onPrediction(h, 0.95f, ms(699)))
        // Because the "none" at 400 ms did not count, the detector is still not armed after the cooldown.
        assertNull(d.onPrediction(h, 0.95f, ms(700)))
        assertNull(d.onPrediction(h, 0.95f, ms(800)))
    }

    @Test
    fun doesNotFireAfterTheCooldownUntilANonePrediction() {
        val d = GestureDetector()
        d.onPrediction(g, 0.95f, ms(0))
        assertEquals(g, d.onPrediction(g, 0.95f, ms(100)))
        for (t in 700L..1500L step 100L) assertNull("at $t ms", d.onPrediction(g, 0.99f, ms(t)))
    }

    @Test
    fun firesAgainAfterNoneThenTheSameGestureTwice() {
        val d = GestureDetector()
        d.onPrediction(g, 0.95f, ms(0))
        assertEquals(g, d.onPrediction(g, 0.95f, ms(100)))
        assertNull(d.onPrediction(none, 0.90f, ms(800))) // after the cooldown: re-arms
        assertNull(d.onPrediction(h, 0.95f, ms(900)))
        assertEquals(h, d.onPrediction(h, 0.95f, ms(1000)))
    }

    @Test
    fun theNoneThatRearmsIsNotPartOfTheNextPair() {
        val d = GestureDetector()
        d.onPrediction(g, 0.95f, ms(0))
        d.onPrediction(g, 0.95f, ms(100))
        assertNull(d.onPrediction(none, 0.90f, ms(800)))
        assertNull(d.onPrediction(g, 0.95f, ms(900))) // the first G after the "none" is not enough
        assertEquals(g, d.onPrediction(g, 0.95f, ms(1000)))
    }

    @Test
    fun resetForgetsTheHistory() {
        val d = GestureDetector()
        assertNull(d.onPrediction(g, 0.95f, ms(0)))
        d.reset() // the frame buffer was cleared between the two predictions
        assertNull(d.onPrediction(g, 0.95f, ms(100)))
        assertEquals(g, d.onPrediction(g, 0.95f, ms(200))) // two new predictions fire as usual
    }

    @Test
    fun resetKeepsTheCooldown() {
        val d = GestureDetector()
        d.onPrediction(g, 0.95f, ms(0))
        assertEquals(g, d.onPrediction(g, 0.95f, ms(100))) // the cooldown ends at 700 ms
        d.reset()
        assertNull(d.onPrediction(none, 0.95f, ms(300)))
        assertNull(d.onPrediction(g, 0.95f, ms(400)))
        assertNull(d.onPrediction(g, 0.95f, ms(500)))
        // Still not armed: the "none" came inside the cooldown. A "none" after it arms the detector again.
        assertNull(d.onPrediction(none, 0.95f, ms(800)))
        assertNull(d.onPrediction(g, 0.95f, ms(900)))
        assertEquals(g, d.onPrediction(g, 0.95f, ms(1000)))
    }

    @Test
    fun thresholdAndCooldownAreConstructorParameters() {
        val d = GestureDetector(threshold = 0.5f, cooldownNanos = ms(100))
        assertNull(d.onPrediction(g, 0.6f, ms(0)))
        assertEquals(g, d.onPrediction(g, 0.6f, ms(10)))
        assertNull(d.onPrediction(none, 0.9f, ms(109))) // inside the 100 ms cooldown
        assertNull(d.onPrediction(none, 0.9f, ms(110))) // the cooldown is over: re-arms
        assertNull(d.onPrediction(g, 0.6f, ms(120)))
        assertEquals(g, d.onPrediction(g, 0.6f, ms(130)))
    }

    @Test
    fun defaultsAre0Point8And600Milliseconds() {
        assertEquals(0.8f, GestureDetector.DEFAULT_THRESHOLD, 0f)
        assertEquals(600_000_000L, GestureDetector.DEFAULT_COOLDOWN_NANOS)
    }
}
