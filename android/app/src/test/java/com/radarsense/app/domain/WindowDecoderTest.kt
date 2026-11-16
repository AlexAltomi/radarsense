package com.radarsense.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class WindowDecoderTest {
    private fun frame(seq: Long, fill: (Int) -> Int) = Frame(
        seq = seq,
        serverTimeMs = 0L,
        label = 0,
        flags = 0,
        pixels = ByteArray(Frame.PAYLOAD_BYTES) { fill(it).toByte() },
        receivedAtNanos = 0L,
    )

    private fun fullBuffer(first: Long = 0L, count: Int = 32, fill: (Long, Int) -> Int): FrameBuffer {
        val buffer = FrameBuffer()
        for (s in first until first + count) buffer.add(frame(s) { i -> fill(s, i) })
        return buffer
    }

    private fun index(frame: Int, channel: Int, range: Int, doppler: Int) =
        ((frame * 3 + channel) * 32 + range) * 32 + doppler

    @Test
    fun outputHas98304Values() {
        assertEquals(98_304, WindowDecoder.OUTPUT_SIZE)
        assertEquals(98_304, WindowDecoder().output.size)
    }

    @Test
    fun decodesZero128And255OnEveryChannel() {
        val decoder = WindowDecoder()
        for ((u, magnitude, angle) in listOf(
            Triple(0, 0f, -1f),
            Triple(128, 128f / 255f, 128f / 255f * 2f - 1f),
            Triple(255, 1f, 1f),
        )) {
            val out = decoder.decode(fullBuffer { _, _ -> u })
            for (f in listOf(0, 17, 31)) {
                assertEquals("u=$u channel 0", magnitude, out[index(f, 0, 5, 9)], 1e-6f)
                assertEquals("u=$u channel 1", angle, out[index(f, 1, 5, 9)], 1e-6f)
                assertEquals("u=$u channel 2", angle, out[index(f, 2, 31, 31)], 1e-6f)
            }
            // The two end points of the ranges, and the exact value of 128 on the angle channels.
            assertEquals(magnitude, out.first(), 1e-6f)
            assertEquals(angle, out.last(), 1e-6f)
        }
        assertEquals(0.0039215686f, decoder.decode(fullBuffer { _, _ -> 128 })[index(0, 1, 0, 0)], 1e-6f)
    }

    @Test
    fun everyValueIsInItsOwnRange() {
        val out = WindowDecoder().decode(fullBuffer { s, i -> ((s * 31 + i * 7) % 256).toInt() })
        for (f in 0 until 32) {
            for (i in 0 until 1024) {
                assertTrue(out[index(f, 0, 0, 0) + i] in 0f..1f)
                assertTrue(out[index(f, 1, 0, 0) + i] in -1f..1f)
                assertTrue(out[index(f, 2, 0, 0) + i] in -1f..1f)
            }
        }
    }

    @Test
    fun orderIsFrameChannelRangeDopplerWithTheOldestFrameFirst() {
        // 40 frames go in, so the window holds the frames 8..39. Every pixel value depends on its frame and position.
        val buffer = fullBuffer(count = 40) { s, i -> ((s * 37 + i * 11 + i / 1024 * 5) % 256).toInt() }
        val out = WindowDecoder().decode(buffer)
        for (f in 0 until 32) {
            val frame = buffer[f]
            assertEquals(8L + f, frame.seq)
            for (c in 0 until 3) for (r in 0 until 32) for (d in 0 until 32) {
                val u = frame.pixel(c, r, d)
                val expected = if (c == 0) u / 255f else u / 255f * 2f - 1f
                assertEquals("frame $f channel $c range $r Doppler $d", expected, out[index(f, c, r, d)], 1e-6f)
            }
        }
    }

    @Test
    fun theFirstValueBelongsToTheOldestFrame() {
        val buffer = fullBuffer(count = 33) { s, _ -> if (s == 1L) 255 else 0 }
        val out = WindowDecoder().decode(buffer) // holds the frames 1..32, and frame 1 is the oldest
        assertEquals(1f, out[index(0, 0, 0, 0)], 0f)
        assertEquals(0f, out[index(1, 0, 0, 0)], 0f)
    }

    @Test
    fun refusesABufferThatIsNotFull() {
        val decoder = WindowDecoder()
        assertThrows(IllegalArgumentException::class.java) { decoder.decode(FrameBuffer()) }
        assertThrows(IllegalArgumentException::class.java) { decoder.decode(fullBuffer(count = 31) { _, _ -> 0 }) }
        decoder.decode(fullBuffer(count = 32) { _, _ -> 0 }) // 32 frames are enough
    }

    @Test
    fun reusesTheSameArrayOnEveryCall() {
        val decoder = WindowDecoder()
        val first = decoder.decode(fullBuffer { _, _ -> 0 })
        val second = decoder.decode(fullBuffer { _, _ -> 255 })
        assertSame(first, second)
        assertSame(decoder.output, second)
        assertEquals(1f, second[0], 0f) // the second call overwrote the values of the first
    }
}
