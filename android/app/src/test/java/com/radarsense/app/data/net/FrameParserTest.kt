package com.radarsense.app.data.net

import com.radarsense.app.domain.Frame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameParserTest {
    // Produced in Python with struct.pack("<IQBB2x", 0x01020304, 1731715200123, 3, 1), the format of
    // backend/replay/replay/codec.py. If the server header changes, this test must change with it.
    private val pythonHeader = byteArrayOf(
        0x04, 0x03, 0x02, 0x01, 0x7B, 0xF4.toByte(), 0x44, 0x32,
        0x93.toByte(), 0x01, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00,
    )

    private fun message(header: ByteArray = pythonHeader, pixels: ByteArray = ByteArray(Frame.PAYLOAD_BYTES)) =
        header + pixels

    @Test
    fun decodesTheHeaderWrittenByPython() {
        val frame = FrameParser.parse(message(), receivedAtNanos = 42L)
        assertEquals(0x01020304L, frame.seq)
        assertEquals(1731715200123L, frame.serverTimeMs)
        assertEquals(3, frame.label)
        assertEquals(1, frame.flags)
        assertTrue(frame.isClipStart)
        assertEquals(42L, frame.receivedAtNanos)
    }

    @Test
    fun readsTheSequenceNumberAsUnsigned() {
        val header = pythonHeader.copyOf().also { for (i in 0..3) it[i] = 0xFF.toByte() }
        assertEquals(4_294_967_295L, FrameParser.parse(message(header)).seq)
    }

    @Test
    fun readsUnknownLabelAndClearedFlags() {
        val header = pythonHeader.copyOf().also { it[12] = 0xFF.toByte(); it[13] = 0 }
        val frame = FrameParser.parse(message(header))
        assertEquals(Frame.UNKNOWN_LABEL, frame.label)
        assertFalse(frame.isClipStart)
    }

    @Test
    fun ignoresOtherFlagBits() {
        val header = pythonHeader.copyOf().also { it[13] = 0xFE.toByte() }
        assertFalse(FrameParser.parse(message(header)).isClipStart)
    }

    @Test
    fun keepsPixelsInChannelRangeDopplerOrder() {
        // Same order as numpy's reshape to (3, 32, 32) on the server.
        val pixels = ByteArray(Frame.PAYLOAD_BYTES) { (it % 251).toByte() }
        val frame = FrameParser.parse(message(pixels = pixels))
        for ((c, r, d) in listOf(Triple(0, 0, 0), Triple(0, 5, 17), Triple(1, 31, 2), Triple(2, 31, 31))) {
            val flat = (c * 32 + r) * 32 + d
            assertEquals(flat % 251, frame.pixel(c, r, d))
        }
    }

    @Test
    fun readsPixelsAsUnsignedBytes() {
        val pixels = ByteArray(Frame.PAYLOAD_BYTES).also { it[0] = 0xFF.toByte(); it[1] = 0x80.toByte() }
        val frame = FrameParser.parse(message(pixels = pixels))
        assertEquals(255, frame.pixel(0, 0, 0))
        assertEquals(128, frame.pixel(0, 0, 1))
    }

    @Test
    fun copiesThePixelsOutOfTheMessage() {
        val msg = message()
        val frame = FrameParser.parse(msg)
        msg[Frame.HEADER_BYTES] = 99
        assertEquals(0, frame.pixel(0, 0, 0))
    }

    @Test(expected = FrameFormatException::class)
    fun rejectsAShortMessage() {
        FrameParser.parse(ByteArray(Frame.FRAME_BYTES - 1))
    }

    @Test(expected = FrameFormatException::class)
    fun rejectsALongMessage() {
        FrameParser.parse(ByteArray(Frame.FRAME_BYTES + 1))
    }

    @Test(expected = FrameFormatException::class)
    fun rejectsAnEmptyMessage() {
        FrameParser.parse(ByteArray(0))
    }
}
