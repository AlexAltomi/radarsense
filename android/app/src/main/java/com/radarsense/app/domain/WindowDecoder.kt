package com.radarsense.app.domain

/**
 * Turns the 32 frames of a full [FrameBuffer] into the float input of the classifier.
 *
 * The output has 32 * 3 * 32 * 32 = 98,304 values in (frame, channel, range, Doppler) order, with the oldest frame
 * first. The decode formulas are the ones in docs/project-rules.md:
 * - channel 0: u / 255, which gives 0..1
 * - channels 1 and 2: u / 255 * 2 - 1, which gives -1..1
 *
 * The output array is created once, and [decode] overwrites it on every call, so decoding allocates nothing. The
 * caller must finish using the array before it calls [decode] again.
 *
 * Not thread-safe. Use one decoder from one coroutine only.
 */
class WindowDecoder {
    /** The array that [decode] fills. It is the same object on every call. */
    val output = FloatArray(OUTPUT_SIZE)

    /** Fills [output] from the 32 frames of [buffer] and returns it. The buffer must be full. */
    fun decode(buffer: FrameBuffer): FloatArray {
        require(buffer.isFull && buffer.size == FrameBuffer.WINDOW_FRAMES) {
            "A window needs ${FrameBuffer.WINDOW_FRAMES} frames, but the buffer holds ${buffer.size}."
        }
        var out = 0
        for (f in 0 until FrameBuffer.WINDOW_FRAMES) {
            val pixels = buffer[f].pixels
            for (i in 0 until Frame.CHANNEL_PIXELS) output[out++] = MAGNITUDE[pixels[i].toInt() and 0xFF]
            for (i in Frame.CHANNEL_PIXELS until Frame.PAYLOAD_BYTES) output[out++] = ANGLE[pixels[i].toInt() and 0xFF]
        }
        return output
    }

    companion object {
        const val OUTPUT_SIZE = FrameBuffer.WINDOW_FRAMES * Frame.PAYLOAD_BYTES // 98,304

        // One lookup table per formula, built once. Index = the uint8 value.
        private val MAGNITUDE = FloatArray(256) { it / 255f }
        private val ANGLE = FloatArray(256) { it / 255f * 2f - 1f }
    }
}
