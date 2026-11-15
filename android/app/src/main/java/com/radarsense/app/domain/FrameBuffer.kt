package com.radarsense.app.domain

/**
 * Holds the newest [capacity] frames of one continuous stretch of a recording, oldest first.
 *
 * The buffer empties itself before adding a frame when:
 * - the frame starts a new clip (flags bit 0), as the project rules require, or
 * - its sequence number does not follow the previous one, which means frames were lost on the way.
 * Either way, a window never mixes frames that were not next to each other in the recording.
 *
 * Not thread-safe. Use it from one coroutine only.
 */
class FrameBuffer(val capacity: Int = WINDOW_FRAMES) {
    private val slots = arrayOfNulls<Frame>(capacity)
    private var next = 0 // slot that the next frame goes into
    private var lastSeq = NO_SEQ

    /** Number of frames held, 0..[capacity]. */
    var size = 0
        private set

    /** Frames added since the buffer was last emptied. Keeps counting after the buffer is full. */
    var addedSinceClear = 0L
        private set

    /** How many times the buffer was emptied because of lost frames (not counting clip starts). */
    var lostFrameResets = 0
        private set

    val isFull: Boolean
        get() = size == capacity

    init {
        require(capacity > 0) { "capacity must be positive, got $capacity." }
    }

    /** Adds a frame. Returns true if the buffer was emptied first. */
    fun add(frame: Frame): Boolean {
        val lostFrames = lastSeq != NO_SEQ && frame.seq != ((lastSeq + 1) and 0xFFFF_FFFFL)
        val reset = frame.isClipStart || lostFrames
        if (reset) clear()
        if (lostFrames && !frame.isClipStart) lostFrameResets++

        slots[next] = frame
        next = (next + 1) % capacity
        if (size < capacity) size++
        addedSinceClear++
        lastSeq = frame.seq
        return reset
    }

    fun clear() {
        slots.fill(null)
        next = 0
        size = 0
        addedSinceClear = 0
        lastSeq = NO_SEQ
    }

    /** The frame at [index], where 0 is the oldest and size - 1 the newest. */
    operator fun get(index: Int): Frame {
        if (index !in 0 until size) throw IndexOutOfBoundsException("index $index, size $size")
        return slots[(next - size + index + capacity) % capacity]!!
    }

    /** The newest frame, or null when empty. */
    fun newest(): Frame? = if (size == 0) null else get(size - 1)

    /**
     * True when the buffer is full and a whole number of [stride] frames has arrived since it filled up:
     * on the 32nd frame, then the 36th, 40th and so on. This is when the classifier runs (Week 6).
     */
    fun isWindowDue(stride: Int = INFERENCE_STRIDE): Boolean =
        isFull && (addedSinceClear - capacity) % stride == 0L

    companion object {
        const val WINDOW_FRAMES = 32
        const val INFERENCE_STRIDE = 4
        private const val NO_SEQ = -1L
    }
}
