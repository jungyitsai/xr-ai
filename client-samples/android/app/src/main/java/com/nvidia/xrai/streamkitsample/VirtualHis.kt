package com.nvidia.xrai.streamkitsample

enum class HisMatchStatus {
    MATCH,
    MISMATCH,
    NOT_FOUND,
}

data class HisPatientRecord(
    val bedId: String,
    val patientName: String,
    val birthday: String?,

    val drugs: List<String>,
    val dose: String,
    val route: String,
    val medicationTime: String,
)

data class HisMatchResult(
    val status: HisMatchStatus,
    val expected: HisPatientRecord?,
    val actual: MergedOcrFields?,
    val matchedFields: List<String>,
    val mismatchedFields: List<String>,
)

private val VIRTUAL_HIS = listOf(
    HisPatientRecord(
        bedId = "6A-252",
        patientName = "Deson",
        birthday = null,

        drugs = listOf(
            "Chlorzoxazone",
            "Ginkgo biloba ext.",
            "Vit B complex Tab",
        ),
        dose = "1 tab each",
        route = "PO",
        medicationTime = "2026-09-07 13:00",
    ),
)

fun compareWithVirtualHis(
    fields: MergedOcrFields,
): HisMatchResult {

    val bedId = fields.bedId
    val patientName = fields.patientName

    if (
        bedId.isNullOrBlank() ||
        patientName.isNullOrBlank()
    ) {
        return HisMatchResult(
            status = HisMatchStatus.NOT_FOUND,
            expected = null,
            actual = fields,
            matchedFields = emptyList(),
            mismatchedFields = emptyList(),
        )
    }

    val expected =
        VIRTUAL_HIS.firstOrNull {
            it.bedId.equals(
                bedId,
                ignoreCase = true,
            )
        }

    if (expected == null) {
        return HisMatchResult(
            status = HisMatchStatus.NOT_FOUND,
            expected = null,
            actual = fields,
            matchedFields = emptyList(),
            mismatchedFields = listOf("bedId"),
        )
    }

    val matchedFields =
        mutableListOf<String>()

    val mismatchedFields =
        mutableListOf<String>()

    fun compareField(
        name: String,
        expectedValue: String,
        actualValue: String?,
    ) {
        if (
            actualValue != null &&
            expectedValue.trim().equals(
                actualValue.trim(),
                ignoreCase = true,
            )
        ) {
            matchedFields += name
        } else {
            mismatchedFields += name
        }
    }

    compareField(
        name = "bedId",
        expectedValue = expected.bedId,
        actualValue = fields.bedId,
    )

    compareField(
        name = "patientName",
        expectedValue = expected.patientName,
        actualValue = fields.patientName,
    )

    if (expected.drugs == fields.drugs) {
        matchedFields += "drugs"
    } else {
        mismatchedFields += "drugs"
    }

    compareField(
        name = "dose",
        expectedValue = expected.dose,
        actualValue = fields.dose,
    )

    compareField(
        name = "route",
        expectedValue = expected.route,
        actualValue = fields.route,
    )

    compareField(
        name = "medicationTime",
        expectedValue = expected.medicationTime,
        actualValue = fields.medicationTime,
    )

    return HisMatchResult(
        status =
            if (mismatchedFields.isEmpty()) {
                HisMatchStatus.MATCH
            } else {
                HisMatchStatus.MISMATCH
            },
        expected = expected,
        actual = fields,
        matchedFields = matchedFields,
        mismatchedFields = mismatchedFields,
    )
}