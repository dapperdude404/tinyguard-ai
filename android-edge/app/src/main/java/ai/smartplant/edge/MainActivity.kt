package ai.smartplant.edge

import android.app.AlertDialog
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.exifinterface.media.ExifInterface
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * SmartPlant Edge V2: offline camera / gallery + ONNX inference.
 * The app deliberately has no android.permission.INTERNET.
 * Scan history contains only class labels, scores and timestamps, never image bytes.
 */
class MainActivity : ComponentActivity() {
    private val bg = Color.rgb(245, 249, 246)
    private val forest = Color.rgb(23, 79, 53)
    private val emerald = Color.rgb(27, 128, 81)
    private val ink = Color.rgb(25, 45, 36)
    private val muted = Color.rgb(92, 111, 101)
    private val border = Color.rgb(219, 233, 224)
    private val executor = Executors.newSingleThreadExecutor()
    private val history by lazy { getSharedPreferences("smartplant_edge_scans", MODE_PRIVATE) }

    private lateinit var status: TextView
    private lateinit var preview: ImageView
    private lateinit var placeholder: TextView
    private lateinit var resultPanel: LinearLayout
    private lateinit var historyPanel: LinearLayout
    private lateinit var cameraButton: Button
    private lateinit var galleryButton: Button
    private var photoUri: Uri? = null
    private var cameraFile: File? = null

    private val camera = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) {
            photoUri?.let { analyze(it) }
        } else {
            cameraFile?.delete()
            cameraFile = null
            status.text = "Camera cancelled. You can try again."
        }
    }
    private val gallery = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) analyze(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = bg
        window.navigationBarColor = bg
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR

        val scroll = ScrollView(this).apply {
            isFillViewport = true
            setBackgroundColor(bg)
            clipToPadding = false
        }
        val content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        scroll.addView(content, ViewGroup.LayoutParams(-1, -2))
        ViewCompat.setOnApplyWindowInsetsListener(scroll) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(dp(18), bars.top + dp(16), dp(18), bars.bottom + dp(24))
            insets
        }
        setContentView(scroll)

        val heading = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val titleGroup = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        titleGroup.addView(label("SmartPlant AI", 27f, forest, true))
        titleGroup.addView(label("EDGE AI  /  PLANT HEALTH", 11f, emerald, true))
        heading.addView(titleGroup, LinearLayout.LayoutParams(0, -2, 1f))
        heading.addView(label("● OFFLINE", 12f, forest, true).apply {
            background = rounded(Color.rgb(226, 243, 233), 20)
            setPadding(dp(11), dp(8), dp(11), dp(8))
            contentDescription = "Offline inference, no internet connection needed"
        })
        content.addView(heading)
        space(content, 16)

        val welcome = card().apply {
            background = GradientDrawable(GradientDrawable.Orientation.TL_BR,
                intArrayOf(forest, Color.rgb(25, 116, 76))).apply { cornerRadius = dp(22).toFloat() }
            addView(ImageView(this@MainActivity).apply {
                setImageResource(R.drawable.ic_smartplant_foreground)
                contentDescription = "SmartPlant AI leaf and neural circuit logo"
                scaleType = ImageView.ScaleType.FIT_CENTER
            }, LinearLayout.LayoutParams(dp(66), dp(66)))
            space(this, 6)
            addView(label("Scan a leaf, understand its patterns.", 20f, Color.WHITE, true))
            space(this, 8)
            addView(label("Capture a leaf photo or choose one from your phone. Our 29-class CNN runs entirely on your device.", 14f,
                Color.rgb(224, 245, 232)))
            space(this, 12)
            addView(label("PRIVATE BY DESIGN  ·  NO CLOUD UPLOAD", 10f, Color.rgb(200, 237, 213), true))
        }
        content.addView(welcome)
        space(content, 18)

        content.addView(section("01", "Capture your leaf"))
        space(content, 10)
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        cameraButton = actionButton("📷  Take photo", emerald, Color.WHITE)
        galleryButton = actionButton("▣  Gallery", Color.WHITE, forest)
        galleryButton.background = rounded(Color.WHITE, 14, border)
        actions.addView(cameraButton, LinearLayout.LayoutParams(0, dp(54), 1f))
        val separator = View(this)
        actions.addView(separator, LinearLayout.LayoutParams(dp(10), 1))
        actions.addView(galleryButton, LinearLayout.LayoutParams(0, dp(54), 1f))
        content.addView(actions)

        cameraButton.setOnClickListener {
            try {
                val dir = File(cacheDir, "images").apply { mkdirs() }
                val file = File.createTempFile("leaf_", ".jpg", dir)
                cameraFile = file
                val uri = FileProvider.getUriForFile(this, packageName + ".fileprovider", file)
                photoUri = uri
                camera.launch(uri)
            } catch (e: Exception) {
                status.text = "Unable to launch camera: " + (e.localizedMessage ?: "unknown error")
            }
        }
        galleryButton.setOnClickListener { gallery.launch("image/*") }

        space(content, 12)
        val imageCard = card().apply { setPadding(dp(10), dp(10), dp(10), dp(10)) }
        placeholder = label("Your leaf preview will appear here", 14f, muted).apply {
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(58), dp(8), dp(58))
        }
        imageCard.addView(placeholder, LinearLayout.LayoutParams(-1, -2))
        preview = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            contentDescription = "Selected plant leaf photograph"
            background = rounded(Color.rgb(234, 243, 236), 13)
            clipToOutline = true
            visibility = View.GONE
        }
        imageCard.addView(preview, LinearLayout.LayoutParams(-1, dp(245)))
        content.addView(imageCard)
        space(content, 9)
        status = label("Ready to scan · Model runs on your phone", 13f, muted)
        status.setPadding(dp(4), 0, dp(4), 0)
        content.addView(status)
        space(content, 22)

        content.addView(section("02", "AI scan result"))
        space(content, 10)
        resultPanel = card()
        content.addView(resultPanel)
        showEmptyResult()
        space(content, 21)

        val historyHeading = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        historyHeading.addView(section("03", "Recent scans"), LinearLayout.LayoutParams(0, -2, 1f))
        val clear = label("Clear", 13f, emerald, true).apply {
            setPadding(dp(14), dp(10), 0, dp(10))
            contentDescription = "Clear saved scan history"
            setOnClickListener {
                AlertDialog.Builder(this@MainActivity)
                    .setTitle("Clear scan history?")
                    .setMessage("This deletes saved labels, scores and dates on this device.")
                    .setNegativeButton("Cancel", null)
                    .setPositiveButton("Clear") { _, _ ->
                        history.edit().remove("records").apply()
                        renderHistory()
                    }.show()
            }
        }
        historyHeading.addView(clear)
        content.addView(historyHeading)
        space(content, 10)
        historyPanel = card()
        content.addView(historyPanel)
        renderHistory()
        space(content, 16)
        content.addView(label("Research use only. Model probabilities are not verified diagnostic confidence, and results should not be used alone for crop treatment decisions.", 12f, muted))
    }

    private fun analyze(uri: Uri) {
        setBusy(true)
        status.text = "Analyzing locally · Please keep the app open"
        executor.execute {
            var capture: File? = null
            try {
                val bitmap = decodePhoto(uri)
                val predictions = EdgeClassifier(applicationContext).use { it.predict(bitmap) }
                capture = cameraFile
                runOnUiThread {
                    preview.setImageBitmap(bitmap)
                    preview.visibility = View.VISIBLE
                    placeholder.visibility = View.GONE
                    showPredictions(predictions)
                    saveHistory(predictions.first())
                    renderHistory()
                    status.text = "Analysis complete · No image uploaded to the cloud"
                    setBusy(false)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    status.text = "Could not analyze image: " + (e.localizedMessage ?: "invalid image or model")
                    setBusy(false)
                }
            } finally {
                capture?.delete()
                if (capture != null && cameraFile == capture) cameraFile = null
            }
        }
    }

    private fun decodePhoto(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri).use { stream ->
            requireNotNull(stream) { "Photo not accessible" }
            BitmapFactory.decodeStream(stream, null, bounds)
        }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Invalid image" }
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1600) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val decoded = contentResolver.openInputStream(uri).use { stream ->
            requireNotNull(stream) { "Photo not accessible" }
            BitmapFactory.decodeStream(stream, null, opts) ?: error("Image decoding failed")
        }
        val orientation = contentResolver.openInputStream(uri).use { stream ->
            if (stream == null) ExifInterface.ORIENTATION_NORMAL
            else ExifInterface(stream).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.postRotate(90f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.postRotate(270f); matrix.postScale(-1f, 1f) }
        }
        if (matrix.isIdentity) return decoded
        return Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true).also {
            if (it !== decoded) decoded.recycle()
        }
    }

    private fun showEmptyResult() {
        resultPanel.removeAllViews()
        resultPanel.addView(label("Awaiting first scan", 17f, ink, true))
        space(resultPanel, 7)
        resultPanel.addView(label("Choose a leaf photo to see predicted categories and probability bars.", 13f, muted))
    }

    private fun showPredictions(predictions: List<Prediction>) {
        resultPanel.removeAllViews()
        val top = predictions.first()
        resultPanel.addView(label("TOP PREDICTED CATEGORY", 11f, emerald, true))
        space(resultPanel, 7)
        resultPanel.addView(label(top.name.replace("|", " · "), 21f, forest, true))
        space(resultPanel, 7)
        resultPanel.addView(label(String.format(Locale.US, "%.1f%% model probability", top.probability * 100f), 15f, emerald, true))
        space(resultPanel, 15)
        resultPanel.addView(label("TOP 5 CATEGORIES", 11f, muted, true))
        space(resultPanel, 6)
        for (p in predictions) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
            row.addView(label(p.name.replace("|", " · "), 12f, ink), LinearLayout.LayoutParams(0, -2, 1f))
            row.addView(label(String.format(Locale.US, "%.1f%%", p.probability * 100f), 12f, muted, true))
            resultPanel.addView(row)
            space(resultPanel, 5)
            val bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 1000
                progress = (p.probability * 1000).roundToInt().coerceIn(0, 1000)
                progressTintList = ColorStateList.valueOf(emerald)
                progressBackgroundTintList = ColorStateList.valueOf(Color.rgb(225, 238, 230))
                contentDescription = p.name + " probability " + String.format(Locale.US, "%.1f percent", p.probability * 100f)
            }
            resultPanel.addView(bar, LinearLayout.LayoutParams(-1, dp(9)))
            space(resultPanel, 12)
        }
        resultPanel.addView(label("Probabilities are model scores, not independently validated diagnosis confidence.", 11f, muted))
    }

    private fun saveHistory(top: Prediction) {
        val records = try { JSONArray(history.getString("records", "[]")) } catch (_: Exception) { JSONArray() }
        val updated = JSONArray()
        updated.put(JSONObject().put("name", top.name).put("score", top.probability.toDouble()).put("time", System.currentTimeMillis()))
        for (i in 0 until minOf(records.length(), 19)) updated.put(records.get(i))
        history.edit().putString("records", updated.toString()).apply()
    }

    private fun renderHistory() {
        historyPanel.removeAllViews()
        val records = try { JSONArray(history.getString("records", "[]")) } catch (_: Exception) { JSONArray() }
        if (records.length() == 0) {
            historyPanel.addView(label("No scans yet. Your recent results will appear here.", 13f, muted))
            return
        }
        val dateFormat = SimpleDateFormat("dd MMM · HH:mm", Locale.getDefault())
        for (i in 0 until records.length()) {
            val obj = records.optJSONObject(i) ?: continue
            val name = obj.optString("name", "Unknown")
            val score = obj.optDouble("score", 0.0)
            val time = obj.optLong("time", 0L)
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
            val info = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
            info.addView(label(name.replace("|", " · "), 13f, ink, true))
            info.addView(label(dateFormat.format(Date(time)), 11f, muted))
            row.addView(info, LinearLayout.LayoutParams(0, -2, 1f))
            row.addView(label(String.format(Locale.US, "%.1f%%", score * 100), 13f, emerald, true))
            historyPanel.addView(row)
            if (i < records.length() - 1) {
                space(historyPanel, 11)
                historyPanel.addView(View(this).apply { setBackgroundColor(border) }, LinearLayout.LayoutParams(-1, dp(1)))
                space(historyPanel, 11)
            }
        }
    }

    private fun setBusy(value: Boolean) {
        cameraButton.isEnabled = !value
        galleryButton.isEnabled = !value
        cameraButton.alpha = if (value) .5f else 1f
        galleryButton.alpha = if (value) .5f else 1f
    }

    private fun section(number: String, title: String): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(label(number, 12f, emerald, true).apply {
                background = rounded(Color.rgb(227, 243, 234), 11)
                setPadding(dp(10), dp(7), dp(10), dp(7))
            })
            addView(label("  " + title, 18f, ink, true))
        }

    private fun card(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(17), dp(17), dp(17), dp(17))
        background = rounded(Color.WHITE, 20, border)
        elevation = dp(2).toFloat()
    }

    private fun label(text: String, size: Float, color: Int, bold: Boolean = false): TextView =
        TextView(this).apply {
            this.text = text
            textSize = size
            setTextColor(color)
            if (bold) setTypeface(null, Typeface.BOLD)
            includeFontPadding = true
        }

    private fun actionButton(text: String, backgroundColor: Int, textColor: Int): Button =
        Button(this).apply {
            this.text = text
            textSize = 13f
            isAllCaps = false
            setTextColor(textColor)
            background = rounded(backgroundColor, 14)
            elevation = dp(1).toFloat()
        }

    private fun rounded(color: Int, radiusDp: Int, stroke: Int? = null): GradientDrawable =
        GradientDrawable().apply {
            setColor(color)
            cornerRadius = dp(radiusDp).toFloat()
            if (stroke != null) setStroke(dp(1), stroke)
        }

    private fun space(parent: LinearLayout, heightDp: Int) {
        parent.addView(View(this), LinearLayout.LayoutParams(1, dp(heightDp)))
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).roundToInt()

    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }
}
