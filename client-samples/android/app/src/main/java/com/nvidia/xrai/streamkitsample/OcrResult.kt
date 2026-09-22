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
    val birthday: String?,
    val birthdayConfidence: Double?,
)

private val BED_ID_REGEX =
    Regex("""\d+[A-Z]-\d+""")

private val BIRTHDAY_REGEX =
    Regex("""生日[：:]\s*(\d{6,8})""")

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

    var bedId: String? = null
    var bedIdConfidence: Double? = null

    var birthday: String? = null
    var birthdayConfidence: Double? = null

    for (detection in image.detections) {
        if (bedId == null) {
            val match =
                BED_ID_REGEX.find(detection.text)

            if (match != null) {
                bedId = match.value
                bedIdConfidence = detection.confidence
            }
        }

        if (birthday == null) {
            val match =
                BIRTHDAY_REGEX.find(detection.text)

            if (match != null) {
                birthday = match.groupValues[1]
                birthdayConfidence = detection.confidence
            }
        }
    }

    return ExtractedOcrFields(
        bedId = bedId,
        bedIdConfidence = bedIdConfidence,
        birthday = birthday,
        birthdayConfidence = birthdayConfidence,
    )
}