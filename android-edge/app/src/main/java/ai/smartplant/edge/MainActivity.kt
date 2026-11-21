package ai.smartplant.edge

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var output: TextView
    private lateinit var preview: ImageView
    private var photoUri: Uri? = null

    private val camera = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) photoUri?.let { analyze(it) } else output.text = "Camera cancelled"
    }
    private val gallery = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) analyze(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
        }
        output = TextView(this).apply { text = "Offline SmartPlant AI • 29-class CNN\nSelect a leaf photo." }
        preview = ImageView(this).apply { adjustViewBounds = true; maxHeight = 800 }
        root.addView(Button(this).apply {
            text = "Take leaf photo"
            setOnClickListener {
                val folder = File(cacheDir, "images").apply { mkdirs() }
                val file = File.createTempFile("leaf_", ".jpg", folder)
                photoUri = FileProvider.getUriForFile(this@MainActivity, "${packageName}.fileprovider", file)
                camera.launch(photoUri)
            }
        })
        root.addView(Button(this).apply {
            text = "Choose existing photo"
            setOnClickListener { gallery.launch("image/*") }
        })
        root.addView(preview)
        root.addView(output)
        setContentView(root)
    }

    private fun analyze(uri: Uri) {
        output.text = "Running CNN locally on your phone..."
        executor.execute {
            try {
                val bitmap = contentResolver.openInputStream(uri).use { stream ->
                    requireNotNull(stream) { "Cannot open photo" }
                    BitmapFactory.decodeStream(stream) ?: error("Invalid image")
                }
                val results = EdgeClassifier(this).use { it.predict(bitmap) }
                runOnUiThread {
                    preview.setImageBitmap(bitmap)
                    output.text = results.joinToString("\n") {
                        "${it.name}: ${"%.1f".format(it.probability * 100)}%"
                    } + "\n\nModel probabilities are not diagnostic confidence."
                }
            } catch (e: Exception) {
                runOnUiThread { output.text = "Inference unavailable: ${e.message}" }
            }
        }
    }

    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }
}
