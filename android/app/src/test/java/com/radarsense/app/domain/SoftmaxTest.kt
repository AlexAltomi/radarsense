package com.radarsense.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class SoftmaxTest {
    @Test
    fun probabilitiesSumToOne() {
        val p = Softmax.apply(floatArrayOf(1f, 2f, 3f, -4f, 0.5f, 0f))
        assertEquals(1f, p.sum(), 1e-6f)
        assertTrue(p.all { it in 0f..1f })
    }

    @Test
    fun equalLogitsGiveEqualProbabilities() {
        val p = Softmax.apply(FloatArray(6))
        for (v in p) assertEquals(1f / 6f, v, 1e-6f)
    }

    @Test
    fun matchesTheTextbookValue() {
        // exp(0) / (exp(0) + exp(ln 3)) gives 0.25 and 0.75
        val p = Softmax.apply(floatArrayOf(0f, Math.log(3.0).toFloat()))
        assertEquals(0.25f, p[0], 1e-6f)
        assertEquals(0.75f, p[1], 1e-6f)
    }

    @Test
    fun hugeLogitsDoNotOverflow() {
        // The logits are unbounded. A naive exp(1000) is infinity and gives NaN.
        val p = Softmax.apply(floatArrayOf(1000f, 0f, -1000f))
        assertEquals(1f, p[0], 0f)
        assertEquals(0f, p[1], 0f)
        assertEquals(0f, p[2], 0f)
        assertTrue(p.none { it.isNaN() })
    }

    @Test
    fun addingTheSameNumberToEveryLogitChangesNothing() {
        val a = Softmax.apply(floatArrayOf(1f, 2f, 3f))
        val b = Softmax.apply(floatArrayOf(501f, 502f, 503f))
        for (i in a.indices) assertEquals(a[i], b[i], 1e-6f)
    }

    @Test
    fun writesIntoTheGivenArray() {
        val out = FloatArray(3)
        assertSame(out, Softmax.apply(floatArrayOf(0f, 0f, 0f), out))
        assertEquals(1f / 3f, out[0], 1e-6f)
    }

    @Test
    fun rejectsEmptyInputAndWrongOutputSize() {
        assertThrows(IllegalArgumentException::class.java) { Softmax.apply(FloatArray(0)) }
        assertThrows(IllegalArgumentException::class.java) { Softmax.apply(FloatArray(3), FloatArray(2)) }
    }

    @Test
    fun predictionTakesTheLargestProbability() {
        val p = Prediction.fromLogits(floatArrayOf(0f, 5f, 1f, 0f, 0f, 0f))
        assertEquals(1, p.classIndex)
        assertEquals(p.probabilities[1], p.confidence, 0f)
        assertTrue(p.confidence > 0.9f)
    }

    @Test
    fun logitsAreNotProbabilities() {
        // Logits are not probabilities: 2.0 vs five zeros is about 0.60, so it must not pass the 0.8 check.
        val p = Prediction.fromLogits(floatArrayOf(0f, 2f, 0f, 0f, 0f, 0f))
        assertTrue(p.confidence < GestureDetector.DEFAULT_THRESHOLD)
    }
}
