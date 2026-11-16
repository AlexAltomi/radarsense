package com.radarsense.app.domain

/**
 * Decides when a gesture counts, from one prediction per window.
 *
 * The rules (docs/project-rules.md, "Output to the app"):
 * - A gesture G is accepted when the last 2 predictions are both G, G is not "none" (class 0), and both have a
 *   confidence of at least [threshold].
 * - It fires once. Then everything is ignored for [cooldownNanos].
 * - After that the detector is armed again by one "none" prediction that arrives after the cooldown, or by [reset].
 *
 * Time is passed in by the caller as nanoseconds, so tests do not need a clock. Use one time source that never goes
 * backwards, for example System.nanoTime().
 *
 * Not thread-safe. Use it from one coroutine only.
 */
class GestureDetector(
    private val threshold: Float = DEFAULT_THRESHOLD,
    private val cooldownNanos: Long = DEFAULT_COOLDOWN_NANOS,
) {
    private var armed = true
    private var cooldownEnd = Long.MIN_VALUE // fires are ignored while now < cooldownEnd
    private var hasPrevious = false
    private var previousClass = NONE
    private var previousConfident = false

    /**
     * Gives the detector one prediction. Returns the class index of the gesture if it fires on this prediction, and
     * null otherwise.
     */
    fun onPrediction(classIndex: Int, confidence: Float, nowNanos: Long): Int? {
        if (nowNanos < cooldownEnd) return null // cooldown: ignore everything, also "none"

        val confident = confidence >= threshold
        val agrees = hasPrevious && previousClass == classIndex && previousConfident
        hasPrevious = true
        previousClass = classIndex
        previousConfident = confident

        if (!armed) {
            if (classIndex == NONE) armed = true // the first "none" after the cooldown re-arms
            return null
        }
        if (classIndex != NONE && confident && agrees) {
            armed = false
            cooldownEnd = nowNanos + cooldownNanos
            hasPrevious = false
            return classIndex
        }
        return null
    }

    /**
     * Forgets the earlier predictions and re-arms the detector. Call it when the frame buffer is cleared: the next
     * prediction then comes from 32 new frames (about 1 s later), so the gesture that fired before cannot fire again,
     * and a gesture early in the next clip must not be lost because the gesture of the previous clip came late.
     * The cooldown still applies: nothing fires until it is over.
     */
    fun reset() {
        hasPrevious = false
        armed = true
    }

    companion object {
        const val NONE = 0
        const val DEFAULT_THRESHOLD = 0.8f
        const val DEFAULT_COOLDOWN_NANOS = 600_000_000L // 600 ms
    }
}
