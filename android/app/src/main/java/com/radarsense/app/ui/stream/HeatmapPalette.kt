package com.radarsense.app.ui.stream

import com.radarsense.app.domain.Frame

/**
 * Turns channel 0 of a frame into ARGB colours for the heatmap. Plain Kotlin, so it can be tested on the laptop.
 *
 * The output is a 32 x 32 image with Doppler on the x axis and range on the y axis, near range at the bottom.
 * Nothing is allocated per frame: the caller passes in the array to fill.
 */
object HeatmapPalette {
    /** 256 colours from dark (0) to bright (255), interpolated between a few fixed colours. */
    val lut: IntArray = buildLut(
        intArrayOf(0x000004, 0x3B0F70, 0x8C2981, 0xDE4968, 0xFE9F6D, 0xFCFDBF),
    )

    /** Writes channel 0 of [pixels] (3,072 frame bytes) into [out] (1,024 ARGB values). */
    fun fillChannel0(pixels: ByteArray, out: IntArray) {
        require(pixels.size == Frame.PAYLOAD_BYTES) { "Expected ${Frame.PAYLOAD_BYTES} bytes, got ${pixels.size}." }
        require(out.size == Frame.CHANNEL_PIXELS) { "Expected ${Frame.CHANNEL_PIXELS} values, got ${out.size}." }
        val n = Frame.SIZE
        for (range in 0 until n) {
            val src = range * n // channel 0 starts at byte 0
            val dst = (n - 1 - range) * n // flip so that near range is at the bottom
            for (doppler in 0 until n) {
                out[dst + doppler] = lut[pixels[src + doppler].toInt() and 0xFF]
            }
        }
    }

    private fun buildLut(stops: IntArray): IntArray = IntArray(256) { i ->
        val pos = i / 255.0 * (stops.size - 1)
        val k = minOf(pos.toInt(), stops.size - 2)
        val t = pos - k
        fun mix(shift: Int): Int {
            val a = (stops[k] shr shift) and 0xFF
            val b = (stops[k + 1] shr shift) and 0xFF
            return Math.round(a + (b - a) * t).toInt()
        }
        (0xFF shl 24) or (mix(16) shl 16) or (mix(8) shl 8) or mix(0)
    }
}
