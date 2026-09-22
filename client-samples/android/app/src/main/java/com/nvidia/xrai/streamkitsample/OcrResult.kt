package com.nvidia.xrai.streamkitsample

import org.json.JSONObject

data class OcrDetection(
    val text: String,
    val confidence: Double,
)

data class OcrImageResult(
    val imageIndex: Int,
    val detections: List<OcrDetection>,
)

data class OcrResult(
    val requestId: String,
    val status: String,
    val images: List<OcrImageResult>,
)

data class ExtractedOcrFields(
    val bedId: String?,
    val bedIdConfidence: Double?,

    val patientName: String?,
    val patientNameConfidence: Double?,

    val drugs: List<String>,
    val drugConfidences: List<Double>,

    val dose: String?,
    val doseConfidence: Double?,

    val route: String?,
    val routeConfidence: Double?,

    val medicationTime: String?,
    val medicationTimeConfidence: Double?,
)

private val BED_ID_REGEX =
    Regex("""\d+[A-Z]-\d+""")

//private val BIRTHDAY_REGEX =
//    Regex("""生日[：:]\s*(\d{6,8})""")

private val MEDICATION_TIME_REGEX =
    Regex("""\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}""")

private fun OcrDetection.isLabel(label: String): Boolean =
    text.trim().equals(label, ignoreCase = true)

private fun confidenceOf(
    detections: List<OcrDetection>
): Double? =
    detections.minOfOrNull { it.confidence }

fun parseOcrResult(json: String): OcrResult {
    val root = JSONObject(json)

    val imagesJson = root.getJSONArray("images")

    val images = buildList {
        for (i in 0 until imagesJson.length()) {
            val imageJson = imagesJson.getJSONObject(i)

            val detectionsJson =
                imageJson.getJSONArray("detections")

            val detections = buildList {
                for (j in 0 until detectionsJson.length()) {
                    val detectionJson =
                        detectionsJson.getJSONObject(j)

                    add(
                        OcrDetection(
                            text = detectionJson.getString("text"),
                            confidence =
                                detectionJson.getDouble("confidence"),
                        )
                    )
                }
            }

            add(
                OcrImageResult(
                    imageIndex =
                        imageJson.getInt("image_index"),
                    detections = detections,
                )
            )
        }
    }

    return OcrResult(
        requestId = root.getString("request_id"),
        status = root.getString("status"),
        images = images,
    )
}

fun extractOcrFields(
    image: OcrImageResult
): ExtractedOcrFields {

    val detections = image.detections

    // Bed ID
    val bedDetection = detections.firstOrNull {
        BED_ID_REGEX.containsMatchIn(it.text)
    }

    val bedId = bedDetection
        ?.let { BED_ID_REGEX.find(it.text)?.value }

    // Locate field labels
    val patientIndex =
        detections.indexOfFirst { it.isLabel("Patient") }

    val drugIndex =
        detections.indexOfFirst { it.isLabel("Drug") }

    val doseIndex =
        detections.indexOfFirst { it.isLabel("Dose") }

    val routeIndex =
        detections.indexOfFirst { it.isLabel("Route") }

    val timeIndex =
        detections.indexOfFirst { it.isLabel("Time") }

    // Patient
    val patientDetections =
        if (
            patientIndex >= 0 &&
            drugIndex > patientIndex
        ) {
            detections.subList(
                patientIndex + 1,
                drugIndex,
            )
        } else {
            emptyList()
        }

    val patientName =
        patientDetections
            .joinToString(" ") {
                it.text.trim()
            }
            .ifBlank { null }

    val patientNameConfidence =
        confidenceOf(patientDetections)

    // Drug
    val drugDetections =
        if (
            drugIndex >= 0 &&
            doseIndex > drugIndex
        ) {
            detections.subList(
                drugIndex + 1,
                doseIndex,
            )
        } else {
            emptyList()
        }

    // Dose
    val doseDetections =
        if (
            doseIndex >= 0 &&
            routeIndex > doseIndex
        ) {
            detections.subList(
                doseIndex + 1,
                routeIndex,
            )
        } else {
            emptyList()
        }

    // Route
    val routeDetections =
        if (
            routeIndex >= 0 &&
            timeIndex > routeIndex
        ) {
            detections.subList(
                routeIndex + 1,
                timeIndex,
            )
        } else {
            emptyList()
        }

    // Time
    val timeDetection =
        if (timeIndex >= 0) {
            detections
                .drop(timeIndex + 1)
                .firstOrNull {
                    MEDICATION_TIME_REGEX.containsMatchIn(
                        it.text
                    )
                }
        } else {
            null
        }

    val medicationTime =
        timeDetection?.let {
            MEDICATION_TIME_REGEX
                .find(it.text)
                ?.value
        }

    return ExtractedOcrFields(
        bedId = bedId,
        bedIdConfidence =
            bedDetection?.confidence,

        patientName =
            patientName,
        patientNameConfidence =
            patientNameConfidence,

        drugs =
            drugDetections.map {
                it.text.trim()
            },
        drugConfidences =
            drugDetections.map {
                it.confidence
            },

        dose =
            doseDetections
                .joinToString(" ") {
                    it.text.trim()
                }
                .ifBlank { null },
        doseConfidence =
            confidenceOf(doseDetections),

        route =
            routeDetections
                .joinToString(" ") {
                    it.text.trim()
                }
                .ifBlank { null },
        routeConfidence =
            confidenceOf(routeDetections),

        medicationTime =
            medicationTime,
        medicationTimeConfidence =
            timeDetection?.confidence,
    )
}