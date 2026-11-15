package com.radarsense.app.domain

/**
 * One frame from the replay stream: the header fields and 3 x 32 x 32 uint8 pixels.
 *
 * The pixels are in [channel, range, Doppler] order, exactly as the replay server sends them. Channel 0 is the
 * magnitude (0..255 stands for 0..1). Channels 1 and 2 are the angle channels of axis A and axis B (0..255 stands
 * for -1..1). This class does not convert them; the classifier does that in Week 6.
 *
 * This is a plain class, not a data class: two frames are the same only if they are the same object.
 */
class Frame(
    /** Sequence number from the server, an unsigned 32-bit value (0..4,294,967,295). */
    val seq: Long,
    /** Server clock in milliseconds when the frame was sent. Never compared with the device clock. */
    val serverTimeMs: Long,
    /** True class index of this frame in the recording, or [UNKNOWN_LABEL]. */
    val label: Int,
    /** Flags byte from the header. Only bit 0 ([FLAG_CLIP_START]) is used. */
    val flags: Int,
    /** The 3,072 pixel bytes. Do not modify. */
    val pixels: ByteArray,
    /** Device time (System.nanoTime) when the frame arrived. Used for latency in Week 6. */
    val receivedAtNanos: Long,
) {
    init {
        require(pixels.size == PAYLOAD_BYTES) { "A frame has $PAYLOAD_BYTES pixel bytes, got ${pixels.size}." }
    }

    /** True for the first frame of a clip. Frames before it do not belong with frames after it. */
    val isClipStart: Boolean
        get() = flags and FLAG_CLIP_START != 0

    /** The value of one pixel, 0..255. */
    fun pixel(channel: Int, range: Int, doppler: Int): Int =
        pixels[(channel * SIZE + range) * SIZE + doppler].toInt() and 0xFF

    companion object {
        const val CHANNELS = 3
        const val SIZE = 32
        const val CHANNEL_PIXELS = SIZE * SIZE // 1,024
        const val PAYLOAD_BYTES = CHANNELS * CHANNEL_PIXELS // 3,072
        const val HEADER_BYTES = 16
        const val FRAME_BYTES = HEADER_BYTES + PAYLOAD_BYTES // 3,088
        const val UNKNOWN_LABEL = 255
        const val FLAG_CLIP_START = 0x01
    }
}
