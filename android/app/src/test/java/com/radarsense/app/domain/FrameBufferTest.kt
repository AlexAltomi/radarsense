package com.radarsense.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameBufferTest {
    private fun frame(seq: Long, clipStart: Boolean = false) = Frame(
        seq = seq,
        serverTimeMs = 0L,
        label = 0,
        flags = if (clipStart) Frame.FLAG_CLIP_START else 0,
        pixels = ByteArray(Frame.PAYLOAD_BYTES),
        receivedAtNanos = 0L,
    )

    private fun seqs(buffer: FrameBuffer) = List(buffer.size) { buffer[it].seq }

    @Test
    fun startsEmpty() {
        val buffer = FrameBuffer()
        assertEquals(0, buffer.size)
        assertFalse(buffer.isFull)
        assertNull(buffer.newest())
    }

    @Test
    fun keepsFramesOldestFirstBeforeFull() {
        val buffer = FrameBuffer()
        for (s in 0L until 5L) buffer.add(frame(s))
        assertEquals(listOf(0L, 1L, 2L, 3L, 4L), seqs(buffer))
        assertEquals(4L, buffer.newest()!!.seq)
    }

    @Test
    fun alwaysHoldsTheNewest32InOrder() {
        val buffer = FrameBuffer()
        for (s in 0L until 100L) buffer.add(frame(s))
        assertTrue(buffer.isFull)
        assertEquals((68L until 100L).toList(), seqs(buffer))
        assertEquals(100L, buffer.addedSinceClear)
    }

    @Test
    fun clipStartEmptiesTheBufferFirst() {
        val buffer = FrameBuffer()
        for (s in 0L until 40L) buffer.add(frame(s))
        val reset = buffer.add(frame(40L, clipStart = true))
        assertTrue(reset)
        assertEquals(listOf(40L), seqs(buffer))
        assertEquals(0, buffer.lostFrameResets) // a clip start is not a loss
    }

    @Test
    fun clipStartOnTheVeryFirstFrameIsFine() {
        val buffer = FrameBuffer()
        buffer.add(frame(7L, clipStart = true))
        buffer.add(frame(8L))
        assertEquals(listOf(7L, 8L), seqs(buffer))
    }

    @Test
    fun aSequenceGapEmptiesTheBufferAndIsCounted() {
        val buffer = FrameBuffer()
        for (s in 0L until 10L) buffer.add(frame(s))
        assertTrue(buffer.add(frame(12L))) // 10 and 11 were lost
        assertEquals(listOf(12L), seqs(buffer))
        assertEquals(1, buffer.lostFrameResets)
    }

    @Test
    fun aRepeatedSequenceNumberCountsAsAGap() {
        val buffer = FrameBuffer()
        buffer.add(frame(5L))
        assertTrue(buffer.add(frame(5L)))
    }

    @Test
    fun sequenceWrapAroundIsNotAGap() {
        val buffer = FrameBuffer()
        buffer.add(frame(4_294_967_295L))
        assertFalse(buffer.add(frame(0L)))
        assertEquals(2, buffer.size)
    }

    @Test
    fun clearForgetsTheLastSequenceNumber() {
        val buffer = FrameBuffer()
        buffer.add(frame(3L))
        buffer.clear()
        assertFalse(buffer.add(frame(100L))) // nothing to compare with after a clear
        assertEquals(0, buffer.lostFrameResets)
    }

    @Test
    fun windowIsDueOnThe32ndFrameThenEvery4() {
        val buffer = FrameBuffer()
        val due = mutableListOf<Long>()
        for (s in 0L until 45L) {
            buffer.add(frame(s))
            if (buffer.isWindowDue()) due += buffer.addedSinceClear
        }
        assertEquals(listOf(32L, 36L, 40L, 44L), due)
    }

    @Test
    fun windowCadenceRestartsAfterAClipStart() {
        val buffer = FrameBuffer()
        for (s in 0L until 34L) buffer.add(frame(s))
        buffer.add(frame(34L, clipStart = true))
        var firstDue = -1L
        for (s in 35L until 80L) {
            buffer.add(frame(s))
            if (buffer.isWindowDue()) { firstDue = buffer.newest()!!.seq; break }
        }
        assertEquals(65L, firstDue) // frames 34..65 are the first full window of the new clip
    }

    @Test
    fun getReturnsTheStoredObjects() {
        val buffer = FrameBuffer(capacity = 3)
        val frames = (0L until 5L).map { frame(it) }
        frames.forEach { buffer.add(it) }
        assertSame(frames[2], buffer[0])
        assertSame(frames[4], buffer[2])
    }

    @Test(expected = IndexOutOfBoundsException::class)
    fun getOutsideTheFilledPartFails() {
        val buffer = FrameBuffer()
        buffer.add(frame(0L))
        buffer[1]
    }
}
