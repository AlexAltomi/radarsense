package com.radarsense.app.domain

/**
 * Decides when a gesture counts, from one prediction per window.
 *
 * The rules (docs/project-rules.md, "Output to the app"):
 * - A gesture G is accepted when the last 2 predictions are both G, G is not "none" (class 0), and both have a
 *   confidence of at least [threshold].
 * - It fires once. Then everything is ignored for [cooldownNanos].
 * - After that the detector is armed again only by one "none" prediction that arrives after the cooldown.
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
     * Forgets the earlier predictions. Call it when the frame buffer is cleared, because the next prediction belongs
     * to a different stretch of frames. The cooldown and the armed state stay as they are.
     */
    fun reset() {
        hasPrevious = false
    }

    companion object {
        const val NONE = 0
        const val DEFAULT_THRESHOLD = 0.8f
        const val DEFAULT_COOLDOWN_NANOS = 600_000_000L // 600 ms
    }
}
