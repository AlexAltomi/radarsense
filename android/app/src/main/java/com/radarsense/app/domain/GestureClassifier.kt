package com.radarsense.app.domain

/** The result of one classifier run: the winning class, how sure the model is, and all class probabilities. */
class Prediction(
    /** Index into the model's class list. 0 is "none". */
    val classIndex: Int,
    /** Softmax probability of [classIndex], 0..1. This is the number the detector compares with 0.8. */
    val confidence: Float,
    /** Softmax probabilities of all classes, summing to 1. */
    val probabilities: FloatArray,
) {
    companion object {
        /** Applies softmax to [logits] and picks the largest probability. */
        fun fromLogits(logits: FloatArray): Prediction {
            val p = Softmax.apply(logits)
            var best = 0
            for (i in p.indices) if (p[i] > p[best]) best = i
            return Prediction(best, p[best], p)
        }
    }
}

/**
 * Turns one window of 32 decoded frames into a [Prediction]. Blocking: call it from a background thread, never from
 * the main thread. Not thread-safe: one caller at a time.
 */
interface GestureClassifier : AutoCloseable {
    /**
     * [window] is the output of [WindowDecoder.decode]: [WindowDecoder.OUTPUT_SIZE] floats in
     * (frame, channel, range, Doppler) order. It is read, not kept, so the caller may reuse the array afterwards.
     */
    fun classify(window: FloatArray): Prediction

    companion object {
        /** Classes of the model: none, label1, label2, label3, label6, label7 (model_card.json). */
        const val NUM_CLASSES = 6
    }
}
