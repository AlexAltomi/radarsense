package com.radarsense.app.data.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.radarsense.app.domain.Frame
import com.radarsense.app.domain.FrameBuffer
import com.radarsense.app.domain.WindowDecoder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Checks that the app gives the same logits as the Python float32 model on the same windows.
 *
 * The data comes from golden.npz (validation users only) and is not part of the repository. Make it with
 * scripts/make_parity_subset.py and push it to the app's external files folder:
 *
 *   adb push data/models/cnn-v1-e12/parity/. /sdcard/Android/data/com.radarsense.app/files/
 *
 * The test runs the real path: uint8 frames -> FrameBuffer -> WindowDecoder -> LiteRT classifier.
 * The full 1,000-window check is a Week 9 task.
 */
@RunWith(AndroidJUnit4::class)
class ClassifierParityTest {
    private val windowBytes = WindowDecoder.OUTPUT_SIZE // uint8: one byte per value

    private fun dataFile(name: String): File {
        val dir = InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null)
        val file = File(dir, name)
        check(file.isFile) {
            "Missing $file. Run scripts/make_parity_subset.py, then: " +
                "adb push data/models/cnn-v1-e12/parity/. /sdcard/Android/data/com.radarsense.app/files/"
        }
        return file
    }

    @Test
    fun logitsMatchPythonFloat32() {
        val windows = dataFile("parity_u8.bin").readBytes()
        val expectedBytes = dataFile("parity_logits.bin").readBytes()
        val count = windows.size / windowBytes
        assertEquals("parity_u8.bin is not a whole number of windows", 0, windows.size % windowBytes)
        assertTrue("no windows", count > 0)
        assertEquals("logits file does not match the windows", count * 6 * 4, expectedBytes.size)
        val expected = ByteBuffer.wrap(expectedBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val decoder = WindowDecoder()
        var worst = 0f
        var disagreements = 0
        LiteRtGestureClassifier.fromAssets(context).use { classifier ->
            for (w in 0 until count) {
                val buffer = FrameBuffer()
                for (f in 0 until FrameBuffer.WINDOW_FRAMES) {
                    val start = w * windowBytes + f * Frame.PAYLOAD_BYTES
                    buffer.add(
                        Frame(
                            seq = f.toLong(), serverTimeMs = 0L, label = Frame.UNKNOWN_LABEL, flags = 0,
                            pixels = windows.copyOfRange(start, start + Frame.PAYLOAD_BYTES),
                            receivedAtNanos = 0L,
                        ),
                    )
                }
                val prediction = classifier.classify(decoder.decode(buffer))
                val logits = classifier.lastLogits()
                var expectedBest = 0
                for (c in 0 until 6) {
                    val want = expected.get(w * 6 + c)
                    worst = maxOf(worst, kotlin.math.abs(logits[c] - want))
                    if (want > expected.get(w * 6 + expectedBest)) expectedBest = c
                }
                if (prediction.classIndex != expectedBest) disagreements++
                assertTrue("probabilities must sum to 1", kotlin.math.abs(prediction.probabilities.sum() - 1f) < 1e-5f)
            }
        }
        println("parity: $count windows, worst logit difference $worst, top-1 disagreements $disagreements")
        assertEquals("top-1 disagreements", 0, disagreements)
        assertTrue("worst logit difference $worst", worst < TOLERANCE)
    }

    @Test
    fun theModelInTheAppIsTheFloat32Model() {
        // 269,544 bytes is model_fp32.tflite in model_card.json (sizes_bytes). The int8 and fp16 files differ.
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val size = context.assets.open(LiteRtGestureClassifier.MODEL_ASSET).use { it.readBytes().size }
        assertEquals(269_544, size)
    }

    private companion object {
        const val TOLERANCE = 1e-3f
    }
}
