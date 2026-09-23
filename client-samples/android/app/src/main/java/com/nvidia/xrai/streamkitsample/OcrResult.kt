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

    val birthday: String?,
    val birthdayConfidence: Double?,

    val drugs: List<String>,
    val drugConfidences: List<Double>,

    val dose: String?,
    val doseConfidence: Double?,

    val route: String?,
    val routeConfidence: Double?,

    val medicationTime: String?,
    val medicationTimeConfidence: Double?,
)

data class MergedOcrFields(
    val bedId: String?,
    val bedIdConfidence: Double?,

    val patientName: String?,
    val patientNameConfidence: Double?,

    val birthday: String?,
    val birthdayConfidence: Double?,

    val drugs: List<String>,
    val drugConfidences: List<Double>,

    val dose: String?,
    val doseConfidence: Double?,

    val route: String?,
    val routeConfidence: Double?,

    val medicationTime: String?,
    val medicationTimeConfidence: Double?,
)

enum class OcrValidationStatus {
    OK,
    LOW_CONFIDENCE,
    MISSING_REQUIRED_FIELD,
}

data class OcrValidationResult(
    val identityStatus: OcrValidationStatus,
    val medicationStatus: OcrValidationStatus,

    val identityLowConfidenceFields: List<String>,
    val identityMissingFields: List<String>,

    val medicationLowConfidenceFields: List<String>,
    val medicationMissingFields: List<String>,
)

private val BED_ID_REGEX =
    Regex("""\d+[A-Z]-\d+""")

private val BIRTHDAY_REGEX =
    Regex("""生日[：:]\s*(\d{6,8})""")

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

    val birthdayDetection =
        detections.firstOrNull {
            BIRTHDAY_REGEX.containsMatchIn(it.text)
        }

    val birthday =
        birthdayDetection?.let {
            BIRTHDAY_REGEX
                .find(it.text)
                ?.groupValues
                ?.get(1)
        }

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

        birthday = birthday,
        birthdayConfidence =
            birthdayDetection?.confidence,

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

fun mergeOcrFields(
    fields: List<ExtractedOcrFields>
): MergedOcrFields {

    require(fields.isNotEmpty()) {
        "At least one OCR result is required"
    }

    if (fields.size == 1) {
        val field = fields.first()

        return MergedOcrFields(
            bedId = field.bedId,
            bedIdConfidence = field.bedIdConfidence,

            patientName = field.patientName,
            patientNameConfidence = field.patientNameConfidence,

            birthday = field.birthday,
            birthdayConfidence = field.birthdayConfidence,

            drugs = field.drugs,
            drugConfidences = field.drugConfidences,

            dose = field.dose,
            doseConfidence = field.doseConfidence,

            route = field.route,
            routeConfidence = field.routeConfidence,

            medicationTime = field.medicationTime,
            medicationTimeConfidence = field.medicationTimeConfidence,
        )
    }

    val first = fields[0]
    val second = fields[1]

    val bed = chooseHigherConfidence(
        first.bedId,
        first.bedIdConfidence,
        second.bedId,
        second.bedIdConfidence,
    )

    val patient = chooseHigherConfidence(
        first.patientName,
        first.patientNameConfidence,
        second.patientName,
        second.patientNameConfidence,
    )

    val birthday =
        chooseHigherConfidence(
            first.birthday,
            first.birthdayConfidence,
            second.birthday,
            second.birthdayConfidence,
        )

    val dose = chooseHigherConfidence(
        first.dose,
        first.doseConfidence,
        second.dose,
        second.doseConfidence,
    )

    val route = chooseHigherConfidence(
        first.route,
        first.routeConfidence,
        second.route,
        second.routeConfidence,
    )

    val medicationTime = chooseHigherConfidence(
        first.medicationTime,
        first.medicationTimeConfidence,
        second.medicationTime,
        second.medicationTimeConfidence,
    )

    val firstDrugScore =
        averageConfidence(first.drugConfidences)

    val secondDrugScore =
        averageConfidence(second.drugConfidences)

    val bestDrugFields =
        if (firstDrugScore >= secondDrugScore) {
            first
        } else {
            second
        }

    return MergedOcrFields(
        bedId = bed.first,
        bedIdConfidence = bed.second,

        patientName = patient.first,
        patientNameConfidence = patient.second,

        birthday = birthday.first,
        birthdayConfidence = birthday.second,

        drugs = bestDrugFields.drugs,
        drugConfidences = bestDrugFields.drugConfidences,

        dose = dose.first,
        doseConfidence = dose.second,

        route = route.first,
        routeConfidence = route.second,

        medicationTime = medicationTime.first,
        medicationTimeConfidence = medicationTime.second,
    )
}

fun validateMergedOcrFields(
    fields: MergedOcrFields,
    confidenceThreshold: Double = 0.9,
): OcrValidationResult {

    val identityLowConfidenceFields =
        mutableListOf<String>()

    val identityMissingFields =
        mutableListOf<String>()

    val medicationLowConfidenceFields =
        mutableListOf<String>()

    val medicationMissingFields =
        mutableListOf<String>()

    fun checkField(
        name: String,
        value: String?,
        confidence: Double?,
        lowConfidenceFields: MutableList<String>,
        missingFields: MutableList<String>,
    ) {
        if (value.isNullOrBlank()) {
            missingFields += name
        } else if (
            confidence == null ||
            confidence < confidenceThreshold
        ) {
            lowConfidenceFields += name
        }
    }

    // Identity
    checkField(
        "bedId",
        fields.bedId,
        fields.bedIdConfidence,
        identityLowConfidenceFields,
        identityMissingFields,
    )

    checkField(
        "patientName",
        fields.patientName,
        fields.patientNameConfidence,
        identityLowConfidenceFields,
        identityMissingFields,
    )

    // Medication details
    if (fields.drugs.isEmpty()) {
        medicationMissingFields += "drugs"
    } else if (
        fields.drugConfidences.isEmpty() ||
        fields.drugConfidences.any {
            it < confidenceThreshold
        }
    ) {
        medicationLowConfidenceFields += "drugs"
    }

    checkField(
        "dose",
        fields.dose,
        fields.doseConfidence,
        medicationLowConfidenceFields,
        medicationMissingFields,
    )

    checkField(
        "route",
        fields.route,
        fields.routeConfidence,
        medicationLowConfidenceFields,
        medicationMissingFields,
    )

    checkField(
        "medicationTime",
        fields.medicationTime,
        fields.medicationTimeConfidence,
        medicationLowConfidenceFields,
        medicationMissingFields,
    )

    val identityStatus = when {
        identityMissingFields.isNotEmpty() ->
            OcrValidationStatus.MISSING_REQUIRED_FIELD

        identityLowConfidenceFields.isNotEmpty() ->
            OcrValidationStatus.LOW_CONFIDENCE

        else ->
            OcrValidationStatus.OK
    }

    val medicationStatus = when {
        medicationMissingFields.isNotEmpty() ->
            OcrValidationStatus.MISSING_REQUIRED_FIELD

        medicationLowConfidenceFields.isNotEmpty() ->
            OcrValidationStatus.LOW_CONFIDENCE

        else ->
            OcrValidationStatus.OK
    }

    return OcrValidationResult(
        identityStatus = identityStatus,
        medicationStatus = medicationStatus,

        identityLowConfidenceFields =
            identityLowConfidenceFields,

        identityMissingFields =
            identityMissingFields,

        medicationLowConfidenceFields =
            medicationLowConfidenceFields,

        medicationMissingFields =
            medicationMissingFields,
    )
}

private fun chooseHigherConfidence(
    value1: String?,
    confidence1: Double?,
    value2: String?,
    confidence2: Double?,
): Pair<String?, Double?> {
    if (value1 == null && value2 == null) {
        return null to null
    }

    if (value1 == null) {
        return value2 to confidence2
    }

    if (value2 == null) {
        return value1 to confidence1
    }

    val c1 = confidence1 ?: 0.0
    val c2 = confidence2 ?: 0.0

    return if (c1 >= c2) {
        value1 to confidence1
    } else {
        value2 to confidence2
    }
}

private fun averageConfidence(
    confidences: List<Double>
): Double =
    if (confidences.isEmpty()) {
        0.0
    } else {
        confidences.average()
    }