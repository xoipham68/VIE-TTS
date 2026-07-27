"""Pydantic request/response models for VieNeu-TTS API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AudioFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"
    flac = "flac"


# ── Requests ──────────────────────────────────────────────────────────────────

class AccentStabilityMixin(BaseModel):
    """Các tham số ổn định accent (chống trộn giọng Bắc/Nam trên văn bản dài).

    Engine (V3TurboVieNeuTTS.infer) đã nhận sẵn các tham số này; API chỉ forward
    xuống. Xem kế hoạch: giảm nhẹ (mitigation), KHÔNG đảm bảo 100%.
    """
    temperature: float = Field(
        0.7, ge=0.1, le=1.5,
        description="Độ ngẫu nhiên sampling. Thấp hơn (0.5–0.7) giảm lệch accent nhưng có thể kém tự nhiên.",
    )
    cb0_temperature: Optional[float] = Field(
        None, ge=0.2, le=1.0,
        description="Temperature riêng cho codebook-0 (mã thô/accent). Để None = tắt. "
                    "Đặt thấp (~0.3) giảm lệch accent NHƯNG có thể gây đoạn câm ~20s trên văn bản dài.",
    )
    max_chars: int = Field(
        256, ge=128, le=512,
        description="Số ký tự tối đa mỗi chunk. Văn bản dài hơn sẽ bị cắt thành nhiều chunk. "
                    "ĐỪNG nâng lên 512 để bớt chunk: đo thực tế 2026-07-20 cho thấy chunk 512 "
                    "ký tự làm model sinh ra tiếng lảm nhảm (3/3 mẫu hỏng, 256 thì nghe được). "
                    "Chỉ số RMS/thời lượng KHÔNG phát hiện được lỗi này — phải nghe.",
    )
    max_new_frames: int = Field(
        300, ge=100, le=2000,
        description="Hard cap số frame sinh ra mỗi chunk (1 frame = 80ms → 300 = 24s). Đủ cho "
                    "chunk 256 ký tự (~16s). Nâng cap chỉ khiến chunk degenerate chạy dài hơn "
                    "trước khi bị chặn, nên chỉ nâng khi đồng thời nâng max_chars.",
    )
    voice_lock: bool = Field(
        False,
        description="Nối đuôi audio chunk trước vào reference chunk sau để giữ giọng nhất quán xuyên chunk.",
    )
    voice_lock_tail_s: float = Field(
        1.5, ge=0.0, le=5.0,
        description="Độ dài (giây) đuôi audio chunk trước dùng làm reference khi voice_lock bật.",
    )
    seed: Optional[int] = Field(
        None,
        description="Seed cho sampling. LƯU Ý: no-op trên backend CPU/ONNX (chỉ có tác dụng trên PyTorch/GPU).",
    )

    def infer_overrides(self) -> dict:
        """Trả về kwargs ổn định accent để truyền vào engine.infer().

        cb0_temperature/seed để None thì bỏ qua (dùng default của engine).
        """
        kw: dict = {
            "temperature": self.temperature,
            "max_chars": self.max_chars,
            "max_new_frames": self.max_new_frames,
            "voice_lock": self.voice_lock,
            "voice_lock_tail_s": self.voice_lock_tail_s,
        }
        if self.cb0_temperature is not None:
            kw["cb0_temperature"] = self.cb0_temperature
        if self.seed is not None:
            kw["seed"] = self.seed
        return kw


class TTSRequest(AccentStabilityMixin):
    text: str = Field(..., min_length=1, max_length=5000, description="Text cần đọc")
    voice: Optional[str] = Field(None, description="Tên preset voice (VD: 'Bình An'). Bỏ trống để dùng giọng mặc định.")
    ref_audio: Optional[str] = Field(None, description="Base64-encoded WAV cho voice cloning")
    ref_text: Optional[str] = Field(None, description="Transcript của ref_audio (tăng chất lượng clone)")
    format: AudioFormat = Field(AudioFormat.wav, description="Định dạng output")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Tốc độ đọc (hiện tại chưa áp dụng vào model)")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text không được để trắng")
        return v.strip()


class BatchItem(AccentStabilityMixin):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None


class BatchTTSRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=20)
    format: AudioFormat = Field(AudioFormat.wav)


# OpenAI-compatible (/v1/audio/speech)
# alloy → None có nghĩa là dùng engine default (Ngọc Linh trong v3turbo)
_OPENAI_VOICE_MAP: dict[str, Optional[str]] = {
    "alloy":   None,
    "echo":    "Xuân Vĩnh",
    "fable":   "Ngọc Linh",
    "onyx":    "Hùng Dũng",
    "nova":    "Thu Hà",
    "shimmer": "Lan Anh",
}

class OpenAITTSRequest(AccentStabilityMixin):
    model: str = Field("vieneu-v3turbo")
    input: str = Field(..., min_length=1, max_length=5000, description="Text cần đọc")
    voice: str = Field("alloy", description="OpenAI voice alias hoặc tên VieNeu trực tiếp")
    response_format: str = Field("wav")
    speed: float = Field(1.0, ge=0.25, le=4.0)

    def resolve_voice(self) -> Optional[str]:
        """Trả về tên VieNeu voice tương ứng.

        - Alias đã biết (alloy/echo/...): dùng mapping, alloy → None (engine default).
        - Voice lạ (tên VieNeu trực tiếp): trả về nguyên tên.
        """
        if self.voice in _OPENAI_VOICE_MAP:
            return _OPENAI_VOICE_MAP[self.voice]
        return self.voice or None


# ── Responses ─────────────────────────────────────────────────────────────────

class VoiceInfo(BaseModel):
    # `id` là tên preset tiếng Việt CÓ DẤU (khóa trong voices_v3_turbo.json). Giữ để
    # tương thích ngược, NHƯNG client nên dùng `slug` hoặc `reserved_id`: chuỗi có dấu
    # đi qua console/env/dòng lệnh Windows biến thành "Tr?ng H?u" và tra cứu fail.
    id: str
    slug: Optional[str] = None          # ASCII ổn định: "trong-huu"
    reserved_id: Optional[str] = None   # speaker token của model: "13"
    description: str
    gender: Optional[str] = None
    sample_rate: int = 48000


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]
    default: str


class BatchResultItem(BaseModel):
    index: int
    audio_b64: Optional[str] = None   # base64-encoded WAV
    error: Optional[str] = None
    duration_ms: Optional[float] = None


class BatchTTSResponse(BaseModel):
    results: list[BatchResultItem]
    total: int
    succeeded: int
    failed: int


class HealthResponse(BaseModel):
    status: str
    version: str
    engine: str
    device: str
