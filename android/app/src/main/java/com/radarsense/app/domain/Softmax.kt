package com.radarsense.app.domain

/**
 * The model outputs logits, which are unbounded. The detector's 0.8 threshold only makes sense on probabilities,
 * so every prediction goes through [apply] first.
 */
object Softmax {
    /**
     * Returns the probabilities for [logits]: all in 0..1 and summing to 1. The largest logit is subtracted first,
     * so large values cannot overflow exp(). Writes into [out] (a new array by default) and returns it.
     */
    fun apply(logits: FloatArray, out: FloatArray = FloatArray(logits.size)): FloatArray {
        require(logits.isNotEmpty()) { "softmax of no values" }
        require(out.size == logits.size) { "out has ${out.size} values, logits has ${logits.size}." }
        var max = logits[0]
        for (v in logits) if (v > max) max = v
        var sum = 0.0
        val exps = DoubleArray(logits.size)
        for (i in logits.indices) {
            exps[i] = Math.exp((logits[i] - max).toDouble())
            sum += exps[i]
        }
        for (i in out.indices) out[i] = (exps[i] / sum).toFloat()
        return out
    }
}
