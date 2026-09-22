import json
from pathlib import Path

from xr_ai_models import ChatMessage

_PROMPT = (
    Path(__file__).resolve().parent
    / "prompts"
    / "ocr_structuring.txt"
).read_text(encoding="utf-8").strip()


async def structure_ocr(
    llm,
    request_results: dict[int, list[dict]],
) -> dict:
    ocr_input = {
        "images": [
            {
                "image_index": image_index,
                "detections": detections,
            }
            for image_index, detections
            in sorted(request_results.items())
        ]
    }

    prompt = _PROMPT.replace(
        "__OCR_RESULTS__",
        json.dumps(
            ocr_input,
            ensure_ascii=False,
        ),
    )

    response = await llm.chat(
        [
            ChatMessage(
                role="user",
                content=prompt,
            )
        ],
        temperature=0.0,
        max_tokens=1000,
        timeout=120.0,
    )

    return json.loads(response.content.strip())