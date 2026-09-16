# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming Riva gRPC client for STT services.

This adapter is intended for streaming-only Riva/NIM ASR services such as
Nemotron ASR 3.5 Streaming.

It preserves the XR_AI STTService interface:

    transcribe(audio: bytes) -> str

while using Riva StreamingRecognize internally.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ._riva_grpc import (
    _channel_ready,
    _import_riva,
    _make_auth,
    _parse_wav,
)


class RivaStreamingSTT:
    """Riva ASR client using StreamingRecognize over gRPC."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str | None = None,
        function_id: str | None = None,
        use_ssl: bool = False,
        language: str = "en-US",
        timeout: float = 30.0,
        health_check: bool = True,
    ) -> None:
        self._rc = _import_riva()
        self._language = language
        self._timeout = timeout
        self._health_check = health_check

        api_key = os.environ.get(api_key_env) if api_key_env else None

        self._auth = _make_auth(
            self._rc,
            base_url,
            use_ssl=use_ssl,
            api_key=api_key,
            function_id=function_id,
        )

        self._asr = self._rc.ASRService(self._auth)

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        timeout: float | None = None,
    ) -> str:
        """Recognize audio through Riva StreamingRecognize.

        XR_AI's VAD already provides a completed utterance. The utterance is
        split into smaller PCM chunks and sent through Riva's streaming API.
        """

        if sample_rate is None:
            sample_rate, channels, audio = _parse_wav(audio)

        config = self._rc.RecognitionConfig(
            encoding=self._rc.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=sample_rate,
            language_code=self._language,
            audio_channel_count=channels,
            max_alternatives=1,
        )

        streaming_config = self._rc.StreamingRecognitionConfig(
            config=config,
            interim_results=False,
        )

        # LINEAR_PCM is signed 16-bit PCM:
        # 2 bytes per sample.
        #
        # Split the completed XR_AI utterance into approximately 100 ms
        # chunks before sending it to StreamingRecognize.
        bytes_per_sample = 2
        chunk_duration_seconds = 0.1

        chunk_size = int(
            sample_rate
            * channels
            * bytes_per_sample
            * chunk_duration_seconds
        )

        def _recognize() -> str:
            audio_chunks = (
                audio[offset:offset + chunk_size]
                for offset in range(0, len(audio), chunk_size)
            )

            responses = self._asr.streaming_response_generator(
                audio_chunks=audio_chunks,
                streaming_config=streaming_config,
            )

            transcripts: list[str] = []

            for response in responses:
                for result in response.results:
                    if result.is_final and result.alternatives:
                        transcripts.append(
                            result.alternatives[0].transcript
                        )

            return "".join(transcripts)

        return await asyncio.wait_for(
            asyncio.to_thread(_recognize),
            timeout or self._timeout,
        )

    async def health(self) -> bool:
        """Whether the Riva gRPC channel is ready."""
        return await _channel_ready(
            self._auth,
            self._health_check,
        )

    async def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._auth.channel.close()

    async def __aenter__(self) -> "RivaStreamingSTT":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()