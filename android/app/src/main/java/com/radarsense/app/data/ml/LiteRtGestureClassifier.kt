package com.radarsense.app.data.ml

import android.content.Context
import com.radarsense.app.domain.GestureClassifier
import com.radarsense.app.domain.Prediction
import com.radarsense.app.domain.WindowDecoder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter

/**
 * Runs the cnn-v1-e12 float32 model with LiteRT on the CPU.
 *
 * The model takes one window of shape [1, 32, 3, 32, 32] (float32) and returns 6 logits. The constructor checks both
 * shapes, so a wrong model file fails when the app starts and not with wrong predictions later. The input and output
 * buffers are made once and reused, so [classify] allocates only the returned [Prediction].
 */
class LiteRtGestureClassifier(model: ByteBuffer) : GestureClassifier {
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(NUM_THREADS))
    private val input: ByteBuffer = ByteBuffer.allocateDirect(WindowDecoder.OUTPUT_SIZE * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(GestureClassifier.NUM_CLASSES) }

    init {
        try {
            val inputShape = interpreter.getInputTensor(0).shape().toList()
            val outputShape = interpreter.getOutputTensor(0).shape().toList()
            require(inputShape == INPUT_SHAPE) { "Model input is $inputShape, expected $INPUT_SHAPE." }
            require(outputShape == OUTPUT_SHAPE) { "Model output is $outputShape, expected $OUTPUT_SHAPE." }
        } catch (e: Throwable) {
            interpreter.close()
            throw e
        }
    }

    override fun classify(window: FloatArray): Prediction {
        require(window.size == WindowDecoder.OUTPUT_SIZE) {
            "A window has ${WindowDecoder.OUTPUT_SIZE} values, got ${window.size}."
        }
        input.clear()
        input.asFloatBuffer().put(window)
        interpreter.run(input, output)
        return Prediction.fromLogits(output[0])
    }

    /** The raw logits of the last [classify] call, as a copy. Used by the parity test. */
    fun lastLogits(): FloatArray = output[0].copyOf()

    override fun close() = interpreter.close()

    companion object {
        const val MODEL_ASSET = "model_fp32.tflite"
        private const val NUM_THREADS = 2
        private val INPUT_SHAPE = listOf(1, 32, 3, 32, 32)
        private val OUTPUT_SHAPE = listOf(1, GestureClassifier.NUM_CLASSES)

        /** Loads the model that is packed in the app's assets. */
        fun fromAssets(context: Context, asset: String = MODEL_ASSET): LiteRtGestureClassifier {
            val bytes = context.assets.open(asset).use { it.readBytes() }
            val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            buffer.put(bytes).rewind()
            return LiteRtGestureClassifier(buffer)
        }
    }
}
