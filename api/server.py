"""VieNeu-TTS External REST API Server.

Khởi chạy:
    # Dev (không cần API key)
    uvicorn api.server:app --reload --port 8001

    # Production
    VIENEU_API_KEY=secret uvicorn api.server:app --host 0.0.0.0 --port 8001 --workers 4

Endpoints:
    GET  /health                  — health check
    GET  /v1/voices               — danh sách preset voices
    POST /v1/tts                  — tổng hợp giọng nói (trả WAV bytes)
    POST /v1/tts/batch            — xử lý batch nhiều text
    POST /v1/audio/speech         — OpenAI-compatible endpoint
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import tempfile
import time
import unicodedata
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response

from api.engine_pool import configure, get_engine, close_engine
from api.models import (
    AudioFormat,
    BatchResultItem,
    BatchTTSRequest,
    BatchTTSResponse,
    HealthResponse,
    OpenAITTSRequest,
    TTSRequest,
    VoiceInfo,
    VoiceListResponse,
)

# ── Config ────────────────────────────────────────────────────────────────────

_API_KEY: Optional[str] = os.getenv("VIENEU_API_KEY")          # None = no auth
_ENGINE_MODE: str       = os.getenv("VIENEU_ENGINE_MODE", "v3turbo")
# Giọng mặc định khi request không chỉ định `voice`. Pin 1 giọng đơn-accent ở đây
# để chống trộn giọng Bắc/Nam (thay vì để engine tự dùng default). None = giữ
# nguyên default của engine (Ngọc Linh với v3turbo).
_DEFAULT_VOICE: Optional[str] = os.getenv("VIENEU_DEFAULT_VOICE") or None
_VERSION: str           = "3.0.5"
_MIME: dict[AudioFormat, str] = {
    AudioFormat.wav:  "audio/wav",
    AudioFormat.mp3:  "audio/mpeg",
    AudioFormat.flac: "audio/flac",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vieneu.api")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure(mode=_ENGINE_MODE)
    logger.info("VieNeu-TTS API started (mode=%s, auth=%s)", _ENGINE_MODE, bool(_API_KEY))
    yield
    close_engine()
    logger.info("VieNeu-TTS API stopped")


app = FastAPI(
    title="VieNeu-TTS API",
    version=_VERSION,
    description="REST API cho VieNeu-TTS — Vietnamese Text-to-Speech với voice cloning",
    lifespan=lifespan,
)


# ── Auth ──────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Kiểm tra API key nếu VIENEU_API_KEY đã được thiết lập."""
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key không hợp lệ hoặc thiếu header X-Api-Key",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _audio_to_bytes(audio: np.ndarray, sample_rate: int, fmt: AudioFormat) -> bytes:
    """Chuyển numpy array thành bytes theo định dạng yêu cầu."""
    buf = io.BytesIO()
    if fmt == AudioFormat.wav:
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    elif fmt == AudioFormat.flac:
        sf.write(buf, audio, sample_rate, format="FLAC")
    elif fmt == AudioFormat.mp3:
        # soundfile không hỗ trợ MP3; xuất WAV rồi để client tự convert
        # Hoặc dùng pydub nếu được cài. Fallback về WAV.
        try:
            from pydub import AudioSegment
            sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
            buf.seek(0)
            seg = AudioSegment.from_wav(buf)
            buf = io.BytesIO()
            seg.export(buf, format="mp3", bitrate="192k")
        except ImportError:
            logger.warning("pydub không được cài — fallback WAV cho request MP3")
            sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _voice_slug(name: str) -> str:
    """Tên preset tiếng Việt → slug ASCII: 'Trọng Hữu' → 'trong-huu'."""
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("Đ", "D").replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _resolve_voice(engine, requested: Optional[str]) -> Optional[str]:
    """Chuẩn hoá `voice` do client gửi về đúng khóa preset của engine.

    Chấp nhận 3 dạng để client không phải gửi chuỗi có dấu qua các ranh giới
    encoding (console/env/CLI Windows) — nguồn gốc lỗi "Voice 'Tr?ng H?u' not found":
      - tên preset gốc:  "Trọng Hữu"
      - reserved_id:     "13"
      - slug ASCII:      "trong-huu" (không phân biệt hoa thường/gạch nối)
    Trả None nếu không khớp — caller nên trả 400 kèm danh sách hợp lệ.
    """
    if not requested:
        return requested
    presets = getattr(engine, "_preset_voices", None) or {}
    if requested in presets:
        return requested
    req = str(requested).strip()
    for name, data in presets.items():
        if isinstance(data, dict) and str(data.get("reserved_id")) == req:
            return name
    want = _voice_slug(req)
    for name in presets:
        if _voice_slug(name) == want:
            return name
    return None


def _voice_or_400(engine, requested: Optional[str]) -> Optional[str]:
    """Như :func:`_resolve_voice` nhưng ném 400 thay vì để engine ném ValueError.

    Giọng sai là lỗi CLIENT: trả 500 khiến client không phân biệt được "tôi gửi sai"
    với "server chết" nên không biết có nên retry / fallback hay không.
    """
    resolved = _resolve_voice(engine, requested)
    if requested and resolved is None:
        presets = getattr(engine, "_preset_voices", None) or {}
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Voice '{requested}' không tồn tại",
                "hint": "Dùng `slug` hoặc `reserved_id` từ GET /v1/voices — "
                        "tên có dấu dễ hỏng khi đi qua console/env Windows.",
                "available": [
                    {"id": n, "slug": _voice_slug(n),
                     "reserved_id": str(d.get("reserved_id")) if isinstance(d, dict) else None}
                    for n, d in presets.items()
                ],
            },
        )
    return resolved


def _count_chunks(text: str, max_chars: int) -> int:
    """Số chunk engine sẽ cắt text thành — dùng CHÍNH hàm engine dùng nên không lệch.

    Đây là con số chẩn đoán quan trọng nhất khi truy sự cố lệch accent: >1 chunk
    mà voice_lock=False nghĩa là mỗi chunk sinh độc lập và có thể "rút" accent khác.
    Trả -1 nếu không tính được (log không bao giờ được làm hỏng request).
    """
    try:
        from vieneu_utils.core_utils import split_text_into_chunks
        return len(split_text_into_chunks(text, max_chars=max_chars))
    except Exception:
        return -1


def _decode_ref_audio(ref_audio_b64: str) -> str:
    """Decode base64 ref audio → temp WAV file path."""
    try:
        audio_bytes = base64.b64decode(ref_audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="ref_audio không phải base64 hợp lệ")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Kiểm tra trạng thái server."""
    try:
        engine = get_engine()
        device = getattr(engine, "device", "cpu")
    except Exception:
        device = "unknown"
    return HealthResponse(
        status="ok",
        version=_VERSION,
        engine=_ENGINE_MODE,
        device=str(device),
    )


@app.get("/v1/voices", response_model=VoiceListResponse, tags=["Voices"])
def list_voices(_: None = Depends(verify_api_key)) -> VoiceListResponse:
    """Trả về danh sách tất cả preset voices có sẵn."""
    engine = get_engine()
    raw = engine.list_preset_voices()        # [(description, id), ...]
    presets = getattr(engine, "_preset_voices", None) or {}
    voices = [
        VoiceInfo(
            id=vid,
            slug=_voice_slug(vid),
            reserved_id=(str(presets[vid].get("reserved_id"))
                         if isinstance(presets.get(vid), dict)
                         and presets[vid].get("reserved_id") is not None else None),
            description=desc,
            gender="nữ" if any(k in desc for k in ["nữ", "bà", "cô", "chị"]) else "nam",
            sample_rate=getattr(engine, "sample_rate", 48000),
        )
        for desc, vid in raw
    ]
    default_voice = getattr(engine, "_default_voice", None) or (voices[0].id if voices else "")
    return VoiceListResponse(voices=voices, default=default_voice)


@app.post("/v1/tts", tags=["TTS"])
def synthesize(
    req: TTSRequest,
    _: None = Depends(verify_api_key),
) -> Response:
    """Tổng hợp text thành audio.

    - Trả về audio bytes với Content-Type tương ứng.
    - Hỗ trợ preset voice hoặc voice cloning qua `ref_audio` (base64 WAV).
    """
    engine = get_engine()
    tmp_path: Optional[str] = None
    t0 = time.perf_counter()

    try:
        kwargs: dict = {"text": req.text, **req.infer_overrides()}
        voice = _voice_or_400(engine, req.voice or _DEFAULT_VOICE)
        if voice:
            kwargs["voice"] = voice
        if req.ref_audio:
            tmp_path = _decode_ref_audio(req.ref_audio)
            kwargs["ref_audio"] = tmp_path
            if req.ref_text:
                kwargs["ref_text"] = req.ref_text

        audio: np.ndarray = engine.infer(**kwargs)
        sr = getattr(engine, "sample_rate", 48000)
        audio_bytes = _audio_to_bytes(audio, sr, req.format)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        n_chunks = _count_chunks(req.text, req.max_chars)
        audio_s = len(audio) / sr if sr else 0.0
        # chunks/max_chars/lock/seed là bộ tham số quyết định ổn định accent —
        # thiếu chúng thì không chẩn đoán được sự cố "lúc giọng Bắc lúc giọng Nam".
        logger.info(
            "TTS ok | chars=%d chunks=%d max_chars=%d voice=%s clone=%s lock=%s(tail=%.1fs) "
            "temp=%s cb0=%s seed=%s frames_cap=%d format=%s audio=%.1fs latency=%.0fms",
            len(req.text), n_chunks, req.max_chars, voice, bool(req.ref_audio),
            req.voice_lock, req.voice_lock_tail_s, req.temperature, req.cb0_temperature,
            req.seed, req.max_new_frames, req.format, audio_s, elapsed_ms,
        )
        if n_chunks > 1 and not req.voice_lock:
            logger.warning(
                "Accent risk | text cắt thành %d chunk nhưng voice_lock=False → mỗi chunk "
                "sinh độc lập, giọng có thể lệch Bắc/Nam giữa các chunk.", n_chunks,
            )
        return Response(
            content=audio_bytes,
            media_type=_MIME[req.format],
            headers={
                "X-Audio-Duration-Samples": str(len(audio)),
                "X-Latency-Ms": f"{elapsed_ms:.0f}",
            },
        )

    except HTTPException:
        raise
    except HTTPException:
        raise                      # 400 giọng-không-tồn-tại phải giữ nguyên,
                                   # không để khối dưới biến thành 500
    except Exception as exc:
        logger.exception("TTS error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp: {exc}") from exc

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/v1/tts/batch", response_model=BatchTTSResponse, tags=["TTS"])
def synthesize_batch(
    req: BatchTTSRequest,
    _: None = Depends(verify_api_key),
) -> BatchTTSResponse:
    """Xử lý nhiều text cùng lúc.

    Mỗi item trong `items` được xử lý tuần tự; lỗi ở một item không ảnh hưởng item khác.
    Kết quả trả về dưới dạng JSON với audio được mã hoá base64.
    """
    engine = get_engine()
    sr = getattr(engine, "sample_rate", 48000)
    results: list[BatchResultItem] = []

    for idx, item in enumerate(req.items):
        t0 = time.perf_counter()
        tmp_path: Optional[str] = None
        try:
            kwargs: dict = {"text": item.text, **item.infer_overrides()}
            voice = _voice_or_400(engine, item.voice or _DEFAULT_VOICE)
            if voice:
                kwargs["voice"] = voice
            if item.ref_audio:
                tmp_path = _decode_ref_audio(item.ref_audio)
                kwargs["ref_audio"] = tmp_path
                if item.ref_text:
                    kwargs["ref_text"] = item.ref_text

            audio: np.ndarray = engine.infer(**kwargs)
            audio_bytes = _audio_to_bytes(audio, sr, req.format)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append(BatchResultItem(
                index=idx,
                audio_b64=base64.b64encode(audio_bytes).decode(),
                duration_ms=elapsed_ms,
            ))

        except Exception as exc:
            logger.warning("Batch item %d error: %s", idx, exc)
            results.append(BatchResultItem(index=idx, error=str(exc)))

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    succeeded = sum(1 for r in results if r.error is None)
    return BatchTTSResponse(
        results=results,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
def openai_speech(
    req: OpenAITTSRequest,
    _: None = Depends(verify_api_key),
) -> Response:
    """OpenAI-compatible TTS endpoint.

    Map voice aliases: alloy→Bình An, echo→Xuân Vĩnh, fable→Ngọc Linh,
    onyx→Hùng Dũng, nova→Thu Hà, shimmer→Lan Anh.
    """
    engine = get_engine()
    voice = _voice_or_400(engine, req.resolve_voice() or _DEFAULT_VOICE)
    t0 = time.perf_counter()

    try:
        kwargs: dict = {"text": req.input, **req.infer_overrides()}
        if voice:
            kwargs["voice"] = voice
        audio: np.ndarray = engine.infer(**kwargs)
        sr = getattr(engine, "sample_rate", 48000)

        fmt_map = {"wav": AudioFormat.wav, "mp3": AudioFormat.mp3, "flac": AudioFormat.flac}
        fmt = fmt_map.get(req.response_format, AudioFormat.wav)
        audio_bytes = _audio_to_bytes(audio, sr, fmt)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("OpenAI TTS ok | voice=%s latency=%.0fms", voice, elapsed_ms)
        return Response(content=audio_bytes, media_type=_MIME[fmt])

    except HTTPException:
        raise                      # 400 giọng-không-tồn-tại phải giữ nguyên,
                                   # không để khối dưới biến thành 500
    except Exception as exc:
        logger.exception("OpenAI TTS error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=os.getenv("VIENEU_HOST", "0.0.0.0"),
        port=int(os.getenv("VIENEU_PORT", "8001")),
        reload=os.getenv("VIENEU_RELOAD", "").lower() in ("1", "true"),
        log_level="info",
    )


if __name__ == "__main__":
    main()
