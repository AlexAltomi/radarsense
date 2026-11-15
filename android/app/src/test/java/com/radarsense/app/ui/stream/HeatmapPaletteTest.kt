package com.radarsense.app.ui.stream

import com.radarsense.app.domain.Frame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HeatmapPaletteTest {
    private val lut = HeatmapPalette.lut

    @Test
    fun lutHas256OpaqueColoursFromDarkToBright() {
        assertEquals(256, lut.size)
        assertTrue(lut.all { (it ushr 24) == 0xFF })
        assertEquals(0xFF000004.toInt(), lut[0])
        assertEquals(0xFFFCFDBF.toInt(), lut[255])
    }

    @Test
    fun lutGetsBrighterOverall() {
        fun brightness(c: Int) = ((c shr 16) and 0xFF) + ((c shr 8) and 0xFF) + (c and 0xFF)
        assertTrue(brightness(lut[0]) < brightness(lut[128]))
        assertTrue(brightness(lut[128]) < brightness(lut[255]))
    }

    @Test
    fun drawsChannel0WithNearRangeAtTheBottom() {
        val pixels = ByteArray(Frame.PAYLOAD_BYTES)
        pixels[0 * 32 + 3] = 255.toByte() // range 0, Doppler 3
        pixels[31 * 32 + 0] = 128.toByte() // range 31, Doppler 0
        val out = IntArray(Frame.CHANNEL_PIXELS)
        HeatmapPalette.fillChannel0(pixels, out)
        assertEquals(lut[255], out[31 * 32 + 3]) // bottom row
        assertEquals(lut[128], out[0 * 32 + 0]) // top row
        assertEquals(lut[0], out[10 * 32 + 10])
    }

    @Test
    fun ignoresTheAngleChannels() {
        val pixels = ByteArray(Frame.PAYLOAD_BYTES)
        for (i in Frame.CHANNEL_PIXELS until Frame.PAYLOAD_BYTES) pixels[i] = 200.toByte()
        val out = IntArray(Frame.CHANNEL_PIXELS)
        HeatmapPalette.fillChannel0(pixels, out)
        assertTrue(out.all { it == lut[0] })
    }
}
