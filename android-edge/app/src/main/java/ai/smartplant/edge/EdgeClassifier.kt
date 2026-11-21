package ai.smartplant.edge

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlin.math.exp
import kotlin.math.roundToInt

data class Prediction(val name: String, val probability: Float)

class EdgeClassifier(private val context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val labels: List<String>

    init {
        val model = context.assets.open("smartplant.onnx").use { it.readBytes() }
        val manifest = JSONObject(context.assets.open("model_manifest.json").bufferedReader().use { it.readText() })
        val digest = MessageDigest.getInstance("SHA-256").digest(model)
            .joinToString("") { "%02x".format(it) }
        require(digest == manifest.getString("sha256")) { "Model checksum mismatch" }
        val json = JSONArray(context.assets.open("labels.json").bufferedReader().use { it.readText() })
        labels = (0 until json.length()).map { json.getString(it) }
        require(labels.size == 29) { "Expected 29 classes" }
        session = env.createSession(model)
        require(session.inputInfo.size == 1 && session.outputInfo.size == 1)
    }

    fun predict(bitmap: Bitmap): List<Prediction> {
        val resized = preprocess(bitmap)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(resized), longArrayOf(1, 3, 160, 160)).use { tensor ->
            session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                val logits = (result[0].value as Array<*>)[0] as FloatArray
                require(logits.size == labels.size)
                val max = logits.maxOrNull() ?: error("Empty model output")
                val exponentials = logits.map { exp((it - max).toDouble()) }
                val sum = exponentials.sum()
                return labels.indices.map { Prediction(labels[it], (exponentials[it] / sum).toFloat()) }
                    .sortedByDescending { it.probability }.take(5)
            }
        }
    }

    private fun preprocess(original: Bitmap): FloatArray {
        // Match torchvision Resize(176), CenterCrop(160), ToTensor(), ImageNet Normalize().
        val w = original.width
        val h = original.height
        require(w > 0 && h > 0)
        val scale = 176.0 / minOf(w, h)
        val nw = (w * scale).toInt()
        val nh = (h * scale).toInt()
        val scaled = Bitmap.createScaledBitmap(original, nw, nh, true)
        val crop = Bitmap.createBitmap(scaled, (nw - 160) / 2, (nh - 160) / 2, 160, 160)
        val pixels = IntArray(160 * 160)
        crop.getPixels(pixels, 0, 160, 0, 0, 160, 160)
        val mean = floatArrayOf(.485f, .456f, .406f)
        val std = floatArrayOf(.229f, .224f, .225f)
        val data = FloatArray(3 * 160 * 160)
        for (i in pixels.indices) {
            val p = pixels[i]
            val channels = intArrayOf((p shr 16) and 255, (p shr 8) and 255, p and 255)
            for (c in 0..2) data[c * 25600 + i] = (channels[c] / 255f - mean[c]) / std[c]
        }
        if (crop !== scaled) crop.recycle()
        if (scaled !== original) scaled.recycle()
        return data
    }

    override fun close() { session.close() }
}
