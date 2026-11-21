package ai.smartplant.edge

/** Advisory review only: human-provided crop labels do not replace model predictions. */
data class CropAssessment(
    val selectedCrop: String,
    val modelTopCrop: String,
    val mismatch: Boolean,
    val withinCrop: List<Prediction>
)

object CropReview {
    fun cropOf(label: String): String = label.substringBefore("|").trim()

    fun assess(predictions: List<Prediction>, selectedCrop: String?): CropAssessment? {
        if (selectedCrop.isNullOrBlank() || selectedCrop == "Auto / unknown" || predictions.isEmpty()) return null
        val overallCrop = cropOf(predictions.first().name)
        val withinCrop = predictions.filter {
            cropOf(it.name).equals(selectedCrop, ignoreCase = true)
        }.sortedByDescending { it.probability }.take(3)
        return CropAssessment(
            selectedCrop = selectedCrop,
            modelTopCrop = overallCrop,
            mismatch = !overallCrop.equals(selectedCrop, ignoreCase = true),
            withinCrop = withinCrop
        )
    }
}
