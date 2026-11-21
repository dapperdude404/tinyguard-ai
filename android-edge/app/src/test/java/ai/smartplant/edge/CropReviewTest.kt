package ai.smartplant.edge

import org.junit.Assert.*
import org.junit.Test

class CropReviewTest {
    private val predictions = listOf(
        Prediction("Strawberry | Leaf Scorch", .884f),
        Prediction("Potato | Early Blight", .062f),
        Prediction("Strawberry | Healthy", .049f),
        Prediction("Potato | Late Blight", .005f)
    )

    @Test fun warnsOnKnownCropMismatch() {
        val report = CropReview.assess(predictions, "Potato")!!
        assertTrue(report.mismatch)
        assertEquals("Strawberry", report.modelTopCrop)
        assertEquals("Potato | Early Blight", report.withinCrop.first().name)
        assertEquals(.062f, report.withinCrop.first().probability, .00001f)
    }

    @Test fun doesNotForceCropPrediction() {
        val report = CropReview.assess(predictions, "Potato")!!
        assertNotEquals(report.modelTopCrop, report.selectedCrop)
        assertEquals(.062f, report.withinCrop.first().probability, .00001f)
    }

    @Test fun unknownCropSkipsCropReview() {
        assertNull(CropReview.assess(predictions, "Auto / unknown"))
    }

    @Test fun matchingCropHasNoMismatch() {
        val report = CropReview.assess(predictions, "Strawberry")!!
        assertFalse(report.mismatch)
    }

    @Test fun handlesMissingCropCandidates() {
        val report = CropReview.assess(predictions, "Grape")!!
        assertTrue(report.withinCrop.isEmpty())
    }
}
