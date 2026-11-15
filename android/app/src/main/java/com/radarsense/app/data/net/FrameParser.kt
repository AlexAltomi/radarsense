package com.radarsense.app.data.net

import com.radarsense.app.domain.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Thrown when a binary message is not a valid frame. The stream stops instead of guessing. */
class FrameFormatException(message: String) : IllegalArgumentException(message)

/**
 * Decodes one binary WebSocket message into a [Frame].
 *
 * The layout matches backend/replay/replay/codec.py: a 16-byte little-endian header (Python struct format
 * "<IQBB2x": uint32 sequence number, uint64 server time in ms, uint8 true label, uint8 flags, 2 reserved bytes)
 * followed by 3,072 pixel bytes. A message of any other length is rejected.
 */
object FrameParser {
    fun parse(message: ByteArray, receivedAtNanos: Long = System.nanoTime()): Frame {
        if (message.size != Frame.FRAME_BYTES) {
            throw FrameFormatException("A frame has ${Frame.FRAME_BYTES} bytes, got ${message.size}.")
        }
        val header = ByteBuffer.wrap(message, 0, Frame.HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        return Frame(
            seq = header.getInt(0).toLong() and 0xFFFF_FFFFL,
            serverTimeMs = header.getLong(4),
            label = message[12].toInt() and 0xFF,
            flags = message[13].toInt() and 0xFF,
            pixels = message.copyOfRange(Frame.HEADER_BYTES, Frame.FRAME_BYTES),
            receivedAtNanos = receivedAtNanos,
        )
    }
}
