# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the simple VLM assistant from shared SDK primitives."""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import nemo_relay
from loguru import logger
from PIL import Image
from xr_ai_logging import setup_logging
from xr_ai_models import VLMService, load_models_config, make_stt, make_tts, make_vlm
from xr_ai_runtime import AgentRuntime
from xr_ai_tools.current_frame import CurrentFrameTool
from xr_ai_tools.image import ImageRegistry
from xr_ai_tools.vision import StreamingImageQueryTool
from xr_ai_voice import HubVoiceTransport, VadConfig, VoiceAgent
from xr_ai_voicegate import load_voice_gate_config

import msgpack
from xr_ai_hub import DataMessage

import base64
import httpx
import json
import time
from functools import partial


_OCR_RESULT_TOPIC = "medical.ocr.result"
_OCR_API_URL = "http://127.0.0.1:8102/v1/ocr"

_ocr_pending: dict[str, dict[int, list[dict]]] = {}

from .agent import (
    INTERRUPTED_TOPIC,
    PARTICIPANT_LEFT_TOPIC,
    USER_QUERY_TOPIC,
    SimpleVlmAgent,
)
from .config import WorkerConfig

_VLM_WARMUP_SIZE = (1280, 720)
_VLM_WARMUP_MAX_TOKENS = 4
_VLM_WARMUP_TIMEOUT_S = 120.0


_OCR_IMAGE_TOPIC = "medical.ocr.image"

async def _run_nemotron_ocr(
    image_bytes: bytes,
    mime_type: str,
) -> dict:
    encoded = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:{mime_type};base64,{encoded}",
            }
        ],
        "merge_levels": ["word"],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _OCR_API_URL,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

async def _on_ocr_image(
    msg: DataMessage,
    *,
    transport: HubVoiceTransport,
) -> None:
    if msg.topic != _OCR_IMAGE_TOPIC:
        return

    try:
        payload = msgpack.unpackb(
            msg.data,
            raw=False,
        )

        request_id = payload.get("request_id", "")
        image_index = int(payload.get("image_index", 0))
        image_count = int(payload.get("image_count", 0))
        mime_type = payload.get("mime_type", "")
        image_bytes = payload.get("image", b"")

        logger.info(
            "OCR image received: participant={!r} "
            "request_id={!r} image_index={}/{} "
            "mime_type={!r} size={} bytes",
            msg.participant_id,
            request_id,
            image_index,
            image_count,
            mime_type,
            len(image_bytes),
        )

        result = await _run_nemotron_ocr(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        logger.info(
            "Nemotron OCR completed: participant={!r} "
            "request_id={!r} image_index={}/{}",
            msg.participant_id,
            request_id,
            image_index,
            image_count,
        )

        detections: list[dict] = []

        for item in result.get("data", []):
            for detection in item.get("text_detections", []):
                prediction = detection.get(
                    "text_prediction",
                    {},
                )

                text = prediction.get("text", "")
                confidence = float(
                    prediction.get("confidence", 0.0)
                )

                detections.append(
                    {
                        "text": text,
                        "confidence": confidence,
                    }
                )

                logger.info(
                    "OCR detection: request_id={!r} "
                    "image_index={} text={!r} confidence={:.4f}",
                    request_id,
                    image_index,
                    text,
                    confidence,
                )

        request_results = _ocr_pending.setdefault(
            request_id,
            {},
        )

        request_results[image_index] = detections

        received_count = len(request_results)

        if received_count < image_count:
            logger.info(
                "OCR request pending: request_id={!r} "
                "received={}/{}",
                request_id,
                received_count,
                image_count,
            )
            return

        logger.info(
            "OCR request complete: request_id={!r} "
            "images={}",
            request_id,
            sorted(request_results.keys()),
        )

        for index in sorted(request_results):
            logger.info(
                "OCR grouped result: request_id={!r} "
                "image_index={} detections={}",
                request_id,
                index,
                request_results[index],
            )
        
        result_payload = {
            "request_id": request_id,
            "status": "completed",
            "images": [
                {
                    "image_index": index,
                    "detections": request_results[index],
                }
                for index in sorted(request_results)
            ],
        }

        result_bytes = json.dumps(
            result_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        await transport.send_return_data(
            DataMessage(
                participant_id=msg.participant_id,
                topic=_OCR_RESULT_TOPIC,
                pts_us=time.time_ns() // 1_000,
                data=result_bytes,
            )
        )

        logger.info(
            "OCR result returned: participant={!r} "
            "request_id={!r} images={} size={} bytes",
            msg.participant_id,
            request_id,
            len(request_results),
            len(result_bytes),
        )

        del _ocr_pending[request_id]

    except Exception:
        logger.exception(
            "Failed to process OCR image: participant={!r}",
            msg.participant_id,
        )

def _vlm_warmup_jpeg() -> bytes:
    buffer = io.BytesIO()
    with Image.new("RGB", _VLM_WARMUP_SIZE, color=(128, 128, 128)) as image:
        image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


_VLM_WARMUP_IMAGE = _vlm_warmup_jpeg()


async def _warm_vlm(vlm: VLMService) -> bool:
    """Confirm readiness by exercising the production multimodal stream."""

    try:
        if not await vlm.health():
            return False
        started_at = time.monotonic()
        logger.info(
            "VLM warmup request started image={}x{} max_tokens={}",
            *_VLM_WARMUP_SIZE,
            _VLM_WARMUP_MAX_TOKENS,
        )
        first_token_at: float | None = None
        async for _ in vlm.stream_images(
            [_VLM_WARMUP_IMAGE],
            "What is the dominant color?",
            system_prompt="Answer with one word.",
            max_tokens=_VLM_WARMUP_MAX_TOKENS,
            timeout=_VLM_WARMUP_TIMEOUT_S,
        ):
            if first_token_at is None:
                first_token_at = time.monotonic()
                logger.info(
                    "VLM warmup first token latency_ms={:.1f}",
                    (first_token_at - started_at) * 1000,
                )
        logger.info(
            "VLM warmup completed total_ms={:.1f}",
            (time.monotonic() - started_at) * 1000,
        )
        return True
    except Exception:
        logger.exception("VLM warmup failed; readiness will retry")
        return False


@asynccontextmanager
async def _relay_event_log(log_file: Path) -> AsyncIterator[Path]:
    event_path = log_file.parent / "relay-events.jsonl"
    sink = event_path.open("w", encoding="utf-8")
    lock = Lock()
    subscriber = "simple-vlm-compact-event-log"

    def write_event(event: nemo_relay.Event) -> None:
        if event.kind == "mark" and event.name == "llm.chunk":
            return
        with lock:
            sink.write(event.to_json())
            sink.write("\n")
            sink.flush()

    try:
        nemo_relay.subscribers.register(subscriber, write_event)
    except Exception:
        sink.close()
        raise
    try:
        yield event_path
    finally:
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscriber)
        sink.close()


async def run_app(
    config: WorkerConfig,
    *,
    ready_file: Path | None = None,
) -> None:
    """Run the worker until the voice session shuts down."""

    log_file = setup_logging("worker")
    models = load_models_config(config.models_config)
    voice_gate = load_voice_gate_config(config.voice_gate_yaml)
    stt = make_stt(models, "stt")
    vlm = make_vlm(models, "vlm")
    tts = make_tts(models, "tts")

    transport = HubVoiceTransport()
    
    transport.endpoint.on_data(
        partial(
            _on_ocr_image,
            transport=transport,
        )
    )
    
    voice = VoiceAgent(
        query_topic=USER_QUERY_TOPIC,
        stt=stt,
        tts=tts,
        vad=VadConfig(
            silence_duration=config.silence_duration,
            min_speech=config.min_speech,
            silero_threshold=config.silero_threshold,
        ),
        voice_gate=voice_gate,
        probes={"vlm": lambda: _warm_vlm(vlm)},
        ready_file=ready_file,
        closeables=(vlm,),
        text_topic="vlm.response",
        idle_timeout_secs=config.idle_timeout_secs,
        transport=transport,
        participant_left_topic=PARTICIPANT_LEFT_TOPIC,
        interrupted_topic=INTERRUPTED_TOPIC,
        interrupt_on_supersede=True,
    )

    runtime = AgentRuntime()
    images = ImageRegistry()
    simple_vlm = runtime.register(
        "simple-vlm",
        SimpleVlmAgent(
            lambda: (
                CurrentFrameTool(
                    endpoint=transport.endpoint,
                    images=images,
                    frame_max_age_s=config.frame_max_age_s,
                    frame_timeout_s=config.frame_timeout_s,
                ),
                StreamingImageQueryTool(
                    images=images,
                    vlm=vlm,
                    system_prompt=config.system_prompt,
                ),
            ),
            transport.endpoint.set_status,
        ),
    )
    runtime.register("voice", voice)

    logger.info("Relay events → {}", log_file.parent / "relay-events.jsonl")
    logger.info("simple-vlm-example starting")
    async with _relay_event_log(log_file):
        async with runtime:
            try:
                await voice.run(runtime)
            finally:
                await simple_vlm.stop()
    logger.info("simple-vlm-example stopped")
