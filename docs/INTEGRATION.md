# VieNeu-TTS — Hướng dẫn Tích hợp

## Tổng quan tích hợp

VieNeu-TTS cung cấp ba phương thức tích hợp chính, mỗi phương thức phù hợp với một tình huống triển khai khác nhau:

| Phương thức | Mô tả | Khi nào dùng |
|-------------|-------|--------------|
| **Python SDK** | Import trực tiếp trong cùng Python process | Scripts, notebooks, ứng dụng Python đơn |
| **REST API Wrapper** | HTTP server tự xây trên SDK | Microservices, đa ngôn ngữ, nhiều client |
| **Remote Mode (LMDeploy)** | Client nhẹ gọi GPU server qua OpenAI-compatible API | GPU tập trung, nhiều client CPU-only |

### Decision Tree — Chọn phương thức tích hợp

```
Bạn có GPU cục bộ không?
├── Có → Ứng dụng Python?
│        ├── Có → Python SDK (mode="v3turbo", device="cuda")
│        └── Không → REST API Wrapper (FastAPI + SDK)
└── Không → Có GPU server riêng không?
             ├── Có → Remote Mode (mode="remote") hoặc REST API Wrapper
             └── Không → Python SDK CPU (mode="v3turbo", ONNX, torch-free)
```

### Yêu cầu chung

- Python >= 3.10 (3.12 khuyến nghị cho production)
- eSpeak NG (phonemizer backend): `apt install espeak-ng` / `winget install eSpeak-NG`
- HuggingFace account (tùy chọn): để tải models private hoặc tăng tốc download

---

## PHẦN 1: Tích hợp Python SDK (trong cùng process)

### 1.1 Cài đặt

```bash
# Minimal — CPU, ONNX, hoàn toàn không cần PyTorch (torch-free)
# Phù hợp cho v3 Turbo trên CPU
pip install vieneu

# Với GPU support — cần CUDA 12.8+, torch, lmdeploy
pip install "vieneu[gpu]"

# Phiên bản cụ thể cho production (tránh breaking changes)
pip install "vieneu==3.0.5"

# Khuyến nghị: dùng uv để quản lý môi trường
uv add vieneu              # CPU
uv add "vieneu[gpu]"       # GPU
```

> **Lưu ý:** Cài đặt `vieneu` không cần `torch` — engine ONNX của v3 Turbo chạy hoàn toàn với `onnxruntime` và `numpy`. Chỉ cài `[gpu]` khi thực sự cần PyTorch hoặc LMDeploy.

### 1.2 Khởi tạo và chọn Engine

Factory function `Vieneu()` là điểm vào duy nhất. Nó tự động chọn backend phù hợp dựa trên tham số `mode`:

```python
from vieneu import Vieneu

# --- V3 Turbo (mặc định) ---
# CPU: tự động dùng ONNX engine (không cần torch)
# GPU: tự động dùng PyTorch engine nếu CUDA khả dụng
tts = Vieneu()
# Tường minh hơn:
tts = Vieneu(mode="v3turbo")

# Ép buộc dùng ONNX dù có GPU (tiết kiệm VRAM)
tts = Vieneu(mode="v3turbo", backend="onnx")

# Ép buộc dùng PyTorch trên GPU
tts = Vieneu(mode="v3turbo", backend="pytorch", device="cuda")

# Chỉ định dtype để tiết kiệm bộ nhớ trên GPU
tts = Vieneu(mode="v3turbo", device="cuda", dtype="bfloat16")

# --- Fast GPU (LMDeploy TurbomindEngine) ---
# Cần: pip install "vieneu[gpu]", CUDA, lmdeploy
tts = Vieneu(
    mode="fast",
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cuda",
    memory_util=0.3,       # Dùng 30% VRAM cho KV cache
    tp=1,                  # Tensor parallel size (1 GPU)
    enable_triton=True,    # Tăng tốc với Triton kernel
    max_batch_size=4       # Tối đa 4 requests song song
)

# --- Standard (PyTorch local, linh hoạt nhất) ---
tts = Vieneu(
    mode="standard",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2"
)

# --- Turbo CPU (GGUF, không cần GPU) ---
tts = Vieneu(mode="turbo")

# --- Turbo GPU (GGUF + ONNX decoder trên GPU) ---
tts = Vieneu(mode="turbo_gpu")

# --- Remote Mode (client nhẹ, không cần model local) ---
tts = Vieneu(
    mode="remote",
    api_base="http://your-server:23333/v1",
    model_name="pnnbao-ump/VieNeu-TTS-v2"
)

# --- Intel XPU (Intel Arc GPU) ---
tts = Vieneu(mode="xpu")
```

**Bảng so sánh engine:**

| Mode | Backend | VRAM | Tốc độ | Voice Cloning | Dùng khi |
|------|---------|------|--------|---------------|----------|
| `v3turbo` (CPU) | ONNX | 0 | Trung bình | Có | CPU-only, production nhẹ |
| `v3turbo` (GPU) | PyTorch | ~4GB | Nhanh | Có | GPU có sẵn |
| `fast` | LMDeploy | ~6GB | Rất nhanh | Có | High-throughput server |
| `turbo` | GGUF llama-cpp | 0 | Trung bình | Không | CPU, không cần GPU |
| `standard` | PyTorch HF | ~4GB | Bình thường | Có | Tùy chỉnh cao |
| `remote` | HTTP client | 0 | Phụ thuộc mạng | Có | Nhiều client, 1 server |

### 1.3 Basic Synthesis

```python
from vieneu import Vieneu
import numpy as np

# Khởi tạo một lần, tái sử dụng nhiều lần
tts = Vieneu()  # v3 Turbo, CPU ONNX

# --- Tổng hợp text đơn giản ---
audio = tts.infer("Xin chào, tôi là trợ lý ảo của bạn.")
tts.save(audio, "output.wav")
# audio là numpy.ndarray float32, sample_rate=48000 (v3turbo)

# --- Với preset voice theo tên ---
audio = tts.infer(
    text="Hôm nay thời tiết rất đẹp, trời trong xanh và gió mát.",
    voice="Xuân Vĩnh"  # Tên string trực tiếp cho v3turbo
)
tts.save(audio, "xuan_vinh.wav")

# --- Với voice object (dùng lại voice data đã load) ---
voice_data = tts.get_preset_voice("Ngọc Linh")
audio = tts.infer(text="Chào buổi sáng!", voice=voice_data)

# --- Với emotion tags (v3 Turbo — thử nghiệm) ---
# Các tag được hỗ trợ: [cười], [thở dài], [hắng giọng]
audio = tts.infer(
    text="Ồ thật sao? [cười] Tôi không ngờ điều đó lại xảy ra! "
         "[thở dài] Thôi được rồi."
)
tts.save(audio, "emotion.wav")

# --- Điều chỉnh tham số sinh ---
audio = tts.infer(
    text="Đây là một câu cần đọc chậm rãi và rõ ràng.",
    voice="Bình An",
    temperature=0.7,           # Thấp hơn → ổn định hơn (mặc định 0.8)
    top_k=25,                  # Số token candidate (mặc định 25)
    top_p=0.95,                # Nucleus sampling (mặc định 0.95)
    repetition_penalty=1.2,    # Tránh lặp từ (mặc định 1.2)
    max_chars=256              # Giới hạn chunk size
)
```

### 1.4 Voice Cloning

V3 Turbo hỗ trợ clone giọng từ file audio reference (3–5 giây, mono, 16kHz hoặc sẽ tự convert):

```python
from vieneu import Vieneu

tts = Vieneu()  # v3turbo, có hỗ trợ cloning

# --- Clone đơn giản: chỉ cần file audio ---
audio = tts.infer(
    text="Đây là giọng được clone từ audio mẫu.",
    ref_audio="reference_voice.wav"
    # v3turbo không yêu cầu ref_text
)
tts.save(audio, "cloned.wav")

# --- Clone với v2 GPU: cần cả audio lẫn transcript ---
# (mode="fast" hoặc mode="standard" với backbone v2)
tts_v2 = Vieneu(
    mode="fast",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2"
)
audio = tts_v2.infer(
    text="Văn bản cần đọc với giọng clone.",
    ref_audio="reference_voice.wav",
    ref_text="Nội dung chính xác của đoạn audio mẫu trên."
)

# --- Encode trước để tái sử dụng (tránh encode lại mỗi lần) ---
# Encode một lần, lưu codes, dùng lại cho nhiều câu
ref_codes = tts.encode_reference("reference_voice.wav")
# ref_codes là numpy array, có thể serialize và lưu lại

import numpy as np
np.save("my_voice_codes.npy", ref_codes)  # Lưu
ref_codes_loaded = np.load("my_voice_codes.npy")  # Load lại

# Dùng lại codes đã encode (nhanh hơn nhiều so với encode từ audio)
for text in ["Câu một.", "Câu hai.", "Câu ba."]:
    audio = tts.infer(
        text=text,
        ref_codes=ref_codes_loaded  # Truyền codes thay vì ref_audio
    )
    tts.save(audio, f"output_{text[:5]}.wav")

# --- Kiểm tra yêu cầu audio reference ---
# Audio lý tưởng: 3–5 giây, ít tiếng ồn nền, nói rõ ràng
# Cảnh báo nếu quá dài (>5.1s): chất lượng clone giảm
```

**Lưu ý về chất lượng clone:**
- Audio 3–5 giây: tốt nhất
- Audio < 1 giây: clone không ổn định
- Audio > 5 giây: chỉ dùng 5 giây đầu, phần còn lại bị cắt bỏ
- Định dạng: WAV/MP3/FLAC đều được, librosa tự convert về 16kHz mono

### 1.5 Streaming (Real-time)

```python
from vieneu import Vieneu
import numpy as np

tts = Vieneu()

# --- Generator-based streaming ---
# Mỗi chunk là numpy array float32, sample_rate=48000
long_text = (
    "Đây là một đoạn văn bản rất dài cần được đọc theo kiểu streaming. "
    "Hệ thống sẽ tự động chia thành các chunk nhỏ, "
    "mỗi chunk không quá 256 ký tự, "
    "và sinh âm thanh tuần tự theo từng chunk."
)

for i, audio_chunk in enumerate(tts.infer_stream(
    text=long_text,
    voice="Ngọc Linh",
    max_chars=256,          # Kích thước chunk tối đa
    silence_p=0.15,         # 150ms silence giữa các chunk (giây)
    crossfade_p=0.0         # Crossfade giữa chunks (0 = không dùng)
)):
    print(f"Chunk {i+1}: {len(audio_chunk)} samples "
          f"({len(audio_chunk)/48000:.2f}s)")
    # Xử lý ngay khi có chunk: lưu file, phát audio, gửi qua WebSocket...
    # Ví dụ: phát qua sounddevice
    # sd.play(audio_chunk, samplerate=48000)
    # sd.wait()

# --- Collect tất cả chunks thành audio hoàn chỉnh ---
chunks = list(tts.infer_stream(long_text))
full_audio = np.concatenate(chunks)
tts.save(full_audio, "full_output.wav")

# --- Streaming qua WebSocket (ví dụ với FastAPI + WebSockets) ---
# Xem thêm ở Phần 2 — FastAPI Wrapper có streaming endpoint

# --- Streaming với pre-buffer (giảm latency cảm nhận) ---
import queue
import threading

def streaming_with_prebuffer(tts, text, prebuffer_chunks=3):
    """
    Đệm trước 3 chunks trước khi bắt đầu phát,
    giống cách Gradio UI xử lý streaming.
    """
    audio_queue = queue.Queue(maxsize=50)
    
    def producer():
        for chunk in tts.infer_stream(text):
            audio_queue.put(chunk)
        audio_queue.put(None)  # Sentinel kết thúc
    
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    
    # Đợi đủ prebuffer_chunks trước khi yield
    buffer = []
    while len(buffer) < prebuffer_chunks:
        chunk = audio_queue.get()
        if chunk is None:
            break
        buffer.append(chunk)
    
    # Yield buffer đã tích lũy
    for chunk in buffer:
        yield chunk
    
    # Tiếp tục yield các chunk còn lại
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        yield chunk

for chunk in streaming_with_prebuffer(tts, long_text):
    # Phát với độ trễ thấp hơn
    pass
```

### 1.6 Batch Processing

```python
from vieneu import Vieneu
import numpy as np
from pathlib import Path

tts = Vieneu()

# --- Batch đơn giản ---
texts = [
    "Đoạn văn bản thứ nhất, nội dung ngắn.",
    "Đoạn văn bản thứ hai, nội dung cũng ngắn.",
    "Đoạn văn bản thứ ba.",
]

# infer_batch trả về list numpy arrays
audios = tts.infer_batch(texts=texts)
for i, audio in enumerate(audios):
    tts.save(audio, f"batch_output_{i+1}.wav")

# --- Batch với voice khác nhau cho mỗi text ---
# Lưu ý: infer_batch của v3turbo không nhận per-item voice trực tiếp
# Nếu cần voice khác nhau, gọi infer() riêng lẻ hoặc dùng vòng lặp
voice_tasks = [
    ("Câu một cần đọc.", "Xuân Vĩnh"),
    ("Câu hai cần đọc.", "Ngọc Linh"),
    ("Câu ba cần đọc.", "Bình An"),
]

output_dir = Path("batch_outputs")
output_dir.mkdir(exist_ok=True)

for i, (text, voice_name) in enumerate(voice_tasks):
    audio = tts.infer(text=text, voice=voice_name)
    tts.save(audio, str(output_dir / f"output_{i+1:03d}.wav"))
    print(f"[{i+1}/{len(voice_tasks)}] {voice_name}: {text[:40]}")

# --- Batch async (Remote Mode) — xử lý song song ---
# Chỉ dùng được với mode="remote"
import asyncio

async def batch_async_example():
    tts_remote = Vieneu(
        mode="remote",
        api_base="http://localhost:23333/v1",
        model_name="pnnbao-ump/VieNeu-TTS-v2"
    )
    
    texts = [f"Đây là câu số {i}." for i in range(1, 11)]
    
    # infer_batch_async: gửi tất cả requests song song,
    # trả về kết quả theo đúng thứ tự input
    audios = await tts_remote.infer_batch_async(
        texts=texts,
        concurrency_limit=10  # Tối đa 10 requests đồng thời
    )
    
    for i, audio in enumerate(audios):
        tts_remote.save(audio, f"async_output_{i+1}.wav")

asyncio.run(batch_async_example())
```

### 1.7 Danh sách Preset Voices

V3 Turbo có 10 preset voices, tất cả được load từ `assets/voices_v3_turbo.json`:

```python
from vieneu import Vieneu

tts = Vieneu()

# Lấy danh sách tất cả voices
voices = tts.list_preset_voices()
for description, voice_id in voices:
    print(f"ID: {voice_id:15s} | {description}")
```

**Danh sách 10 voices có sẵn trong v3 Turbo:**

| Voice ID | Giới tính | reserved_id |
|----------|-----------|-------------|
| Ngoc Lan | Nữ | 13 |
| Gia Bao | Nam | 16 |
| Thai Son | Nam | 17 |
| Duc Tri | Nam | 21 |
| My Duyen | Nữ | 22 |
| Truc Ly | Nữ | 30 |
| Xuan Vinh | Nam | 32 |
| Trong Huu | Nam | 36 |
| Binh An | Nam | 37 |
| Ngoc Linh | Nữ | 41 |

> **Voice mặc định:** Ngoc Linh (ID: 41) — được dùng khi không truyền `voice`.

```python
# Lấy voice data theo tên
voice_data = tts.get_preset_voice("Ngoc Linh")
# voice_data chứa: {"codes": np.ndarray, "reserved_id": int, ...}

# Dùng voice mặc định (không truyền voice)
audio = tts.infer("Xin chào!")

# Truyền tên string trực tiếp (chỉ v3turbo)
audio = tts.infer("Xin chào!", voice="Xuan Vinh")

# Truyền voice data object
audio = tts.infer("Xin chào!", voice=voice_data)
```

### 1.8 Async Integration (asyncio)

VieNeu không phải native async — tất cả inference chạy đồng bộ. Để tích hợp vào ứng dụng asyncio (FastAPI, aiohttp, v.v.), cần wrap trong thread pool:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from vieneu import Vieneu
import numpy as np

class AsyncVieneu:
    """
    Wrapper async-safe cho VieNeu.
    Dùng ThreadPoolExecutor với max_workers=1 vì Vieneu không thread-safe:
    chỉ một thread được gọi inference tại một thời điểm.
    """
    def __init__(self, **kwargs):
        self._tts = Vieneu(**kwargs)
        # max_workers=1: đảm bảo sequential inference, tránh race condition
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    async def infer(self, text: str, **kwargs) -> np.ndarray:
        """Gọi inference bất đồng bộ, không block event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._tts.infer(text, **kwargs)
        )
    
    async def infer_batch(self, texts: list, **kwargs) -> list:
        """Batch inference bất đồng bộ."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._tts.infer_batch(texts, **kwargs)
        )
    
    def save(self, audio: np.ndarray, path: str):
        """Lưu file — đồng bộ, không cần wrap."""
        self._tts.save(audio, path)
    
    async def close(self):
        """Giải phóng tài nguyên khi shutdown."""
        self._executor.shutdown(wait=True)
        if hasattr(self._tts, 'close'):
            self._tts.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()

# --- Ví dụ sử dụng ---
async def main():
    async with AsyncVieneu(mode="v3turbo") as tts:
        # Xử lý nhiều requests đồng thời trong FastAPI
        # (thực tế sẽ queue lại vì max_workers=1)
        tasks = [
            tts.infer("Câu đầu tiên."),
            tts.infer("Câu thứ hai.", voice="Xuan Vinh"),
            tts.infer("Câu thứ ba.", voice="Ngoc Lan"),
        ]
        # gather sẽ chạy tuần tự do executor 1 worker
        results = await asyncio.gather(*tasks)
        
        for i, audio in enumerate(results):
            tts.save(audio, f"async_{i+1}.wav")
        
        print(f"Đã tổng hợp {len(results)} file audio.")

asyncio.run(main())
```

### 1.9 Error Handling và Retry

```python
import time
import logging
from vieneu import Vieneu
from threading import local

logger = logging.getLogger(__name__)

# --- Retry với exponential backoff ---
def infer_with_retry(
    tts: Vieneu,
    text: str,
    max_retries: int = 3,
    **kwargs
):
    """
    Thử lại inference tối đa max_retries lần.
    GPU OOM: clear cache rồi thử lại ngay.
    Lỗi khác: exponential backoff (1s, 2s, 4s, ...).
    """
    for attempt in range(max_retries):
        try:
            return tts.infer(text, **kwargs)
        
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda oom" in error_msg:
                logger.warning(
                    f"GPU OOM tại attempt {attempt+1}/{max_retries}, "
                    "đang clear cache..."
                )
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"GPU OOM sau {max_retries} lần thử. "
                        "Thử giảm max_chars hoặc dùng CPU mode."
                    ) from e
                # Không sleep sau OOM, thử ngay sau khi clear cache
                continue
            
            # RuntimeError khác: retry với backoff
            wait = 2 ** attempt
            logger.error(f"RuntimeError: {e}. Thử lại sau {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise
        
        except Exception as e:
            wait = 2 ** attempt
            logger.error(f"Lỗi không xác định: {e}. Thử lại sau {wait}s...")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise

# --- Thread-local storage (quan trọng cho multi-threaded apps) ---
# Vieneu KHÔNG thread-safe: mỗi thread cần instance riêng
_thread_local = local()

def get_tts_instance(mode: str = "v3turbo", **kwargs) -> Vieneu:
    """
    Trả về Vieneu instance riêng cho mỗi thread.
    Tự động tạo mới nếu chưa có.
    """
    attr_name = f"tts_{mode}"
    if not hasattr(_thread_local, attr_name):
        logger.info(
            f"Tạo Vieneu instance mới cho thread "
            f"{threading.current_thread().name}"
        )
        instance = Vieneu(mode=mode, **kwargs)
        setattr(_thread_local, attr_name, instance)
    return getattr(_thread_local, attr_name)

# --- Context manager pattern ---
# Dùng 'with' để đảm bảo tài nguyên được giải phóng
with Vieneu() as tts:
    audio = tts.infer("Văn bản cần tổng hợp.")
    tts.save(audio, "output.wav")
# tts.close() được gọi tự động khi ra khỏi block

# --- Ví dụ tích hợp đầy đủ ---
import threading

def worker_thread(texts: list, output_dir: str, thread_id: int):
    """Worker chạy trong thread riêng với Vieneu instance riêng."""
    tts = get_tts_instance(mode="v3turbo")
    
    for i, text in enumerate(texts):
        try:
            audio = infer_with_retry(tts, text, max_retries=3)
            tts.save(audio, f"{output_dir}/thread{thread_id}_{i:04d}.wav")
        except Exception as e:
            logger.error(f"[Thread {thread_id}] Lỗi text '{text[:30]}': {e}")

# Tạo 4 worker threads, mỗi thread xử lý batch riêng
all_texts = [f"Câu số {i}." for i in range(100)]
chunk_size = len(all_texts) // 4

import os
os.makedirs("threaded_output", exist_ok=True)

threads = []
for t_id in range(4):
    start = t_id * chunk_size
    end = start + chunk_size if t_id < 3 else len(all_texts)
    t = threading.Thread(
        target=worker_thread,
        args=(all_texts[start:end], "threaded_output", t_id)
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()
print("Hoàn thành xử lý song song.")
```

---

## PHẦN 2: Tích hợp REST API (qua HTTP)

### 2.1 Tại sao cần REST API Wrapper?

VieNeu core không có HTTP server built-in (serve.py chỉ wrap LMDeploy, không expose full TTS API). Bạn cần tự xây REST wrapper khi:
- Ứng dụng viết bằng ngôn ngữ khác (JavaScript, Go, Java, Rust)
- Kiến trúc microservices — TTS là một service độc lập
- Muốn API key auth, rate limiting, request logging
- Nhiều client cần truy cập cùng một model instance

### 2.2 FastAPI Wrapper — Full Implementation

```python
# api_server.py
# Chạy: uvicorn api_server:app --host 0.0.0.0 --port 8000
# Yêu cầu: pip install fastapi uvicorn[standard] soundfile

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, validator
from typing import Optional
import uvicorn
import base64
import io
import os
import tempfile
import logging
import numpy as np
import soundfile as sf
from vieneu import Vieneu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VieNeu-TTS API",
    description="Vietnamese Neural TTS REST API",
    version="1.0.0"
)

# --- Khởi tạo TTS một lần duy nhất khi server start ---
# Với GPU: Vieneu(mode="v3turbo", device="cuda")
# Với CPU: Vieneu(mode="v3turbo") — ONNX, torch-free
tts = Vieneu(mode="v3turbo")
logger.info("VieNeu-TTS engine đã sẵn sàng.")

# --- Pydantic models cho request/response ---

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      description="Văn bản cần tổng hợp")
    voice: Optional[str] = Field(None,
                                  description="Tên preset voice (vd: 'Xuan Vinh')")
    ref_audio_b64: Optional[str] = Field(None,
                                          description="Base64-encoded WAV cho voice cloning")
    ref_text: Optional[str] = Field(None,
                                     description="Transcript của ref audio (tùy chọn với v3)")
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(25, ge=1, le=100)
    top_p: float = Field(0.95, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0)
    max_chars: int = Field(256, ge=50, le=512)
    
    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("text không được để trống")
        return v.strip()

class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS API format."""
    model: str = Field("vieneu-v3turbo")
    input: str = Field(..., min_length=1)
    voice: str = Field("alloy", description="OpenAI voice name, được map sang VieNeu")
    response_format: str = Field("wav")
    speed: float = Field(1.0, ge=0.25, le=4.0)

class VoiceListResponse(BaseModel):
    voices: list
    default_voice: str
    total: int

# --- API Key authentication ---
API_KEY = os.environ.get("VIENEU_API_KEY", "dev-secret-key")

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Kiểm tra API key từ header X-Api-Key."""
    if API_KEY == "dev-secret-key":
        return  # Bỏ qua auth trong dev mode
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="API key không hợp lệ. Truyền qua header X-Api-Key."
        )

def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 48000) -> bytes:
    """Convert numpy array sang WAV bytes để trả về HTTP response."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

# --- Endpoints ---

@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra server còn sống không."""
    return {
        "status": "ok",
        "model": "vieneu-v3turbo",
        "sample_rate": 48000,
        "version": "1.0.0"
    }

@app.get("/v1/voices", tags=["Voices"],
         response_model=VoiceListResponse,
         dependencies=[Depends(verify_api_key)])
def list_voices():
    """Trả về danh sách tất cả preset voices."""
    voices = tts.list_preset_voices()
    return {
        "voices": [{"id": vid, "description": desc} for desc, vid in voices],
        "default_voice": "Ngoc Linh",
        "total": len(voices)
    }

@app.post("/v1/tts",
          tags=["Synthesis"],
          dependencies=[Depends(verify_api_key)],
          response_class=Response)
def synthesize(req: TTSRequest):
    """
    Tổng hợp giọng nói từ văn bản.
    Trả về file WAV dạng binary.
    """
    temp_path = None
    try:
        # Xây dựng kwargs cho tts.infer()
        infer_kwargs = {
            "text": req.text,
            "temperature": req.temperature,
            "top_k": req.top_k,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
            "max_chars": req.max_chars,
        }
        
        # Xử lý voice cloning từ base64 audio
        if req.ref_audio_b64:
            try:
                audio_bytes = base64.b64decode(req.ref_audio_b64)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="ref_audio_b64 không phải base64 hợp lệ."
                )
            # Lưu tạm thời ra file để librosa/soundfile đọc
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as f:
                f.write(audio_bytes)
                temp_path = f.name
            infer_kwargs["ref_audio"] = temp_path
            if req.ref_text:
                infer_kwargs["ref_text"] = req.ref_text
        
        elif req.voice:
            # Preset voice theo tên
            infer_kwargs["voice"] = req.voice
        
        audio = tts.infer(**infer_kwargs)
        wav_bytes = audio_to_wav_bytes(audio, sample_rate=48000)
        
        logger.info(
            f"Tổng hợp thành công: {len(req.text)} chars, "
            f"voice={req.voice or 'default'}, "
            f"audio={len(audio)/48000:.2f}s"
        )
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-Audio-Duration": str(round(len(audio)/48000, 3)),
                "X-Sample-Rate": "48000",
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi TTS: {str(e)}")
    
    finally:
        # Xóa file tạm nếu đã tạo
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/v1/tts/stream",
          tags=["Synthesis"],
          dependencies=[Depends(verify_api_key)])
def synthesize_stream(req: TTSRequest):
    """
    Streaming TTS — trả về audio theo từng chunk.
    Phù hợp cho real-time playback trên client.
    """
    def generate():
        try:
            for chunk in tts.infer_stream(
                text=req.text,
                voice=req.voice,
                temperature=req.temperature,
                max_chars=req.max_chars,
            ):
                # Mỗi chunk là numpy float32 → convert sang PCM bytes
                pcm = (chunk * 32767).astype(np.int16)
                yield pcm.tobytes()
        except Exception as e:
            logger.error(f"Streaming error: {e}")
    
    return StreamingResponse(
        generate(),
        media_type="audio/pcm",  # raw PCM 16-bit, 48kHz
        headers={
            "X-Sample-Rate": "48000",
            "X-Channels": "1",
            "X-Bit-Depth": "16"
        }
    )

@app.post("/v1/audio/speech",
          tags=["OpenAI Compatible"],
          dependencies=[Depends(verify_api_key)],
          response_class=Response)
def openai_tts(req: OpenAITTSRequest):
    """
    OpenAI-compatible TTS endpoint.
    Drop-in replacement cho OpenAI TTS API.
    """
    # Map OpenAI voice names sang VieNeu preset voices
    VOICE_MAP = {
        "alloy":   "Binh An",
        "echo":    "Xuan Vinh",
        "fable":   "Ngoc Linh",
        "onyx":    "Thai Son",
        "nova":    "My Duyen",
        "shimmer": "Truc Ly",
    }
    vieneu_voice = VOICE_MAP.get(req.voice, req.voice)
    
    try:
        audio = tts.infer(text=req.input, voice=vieneu_voice)
        wav_bytes = audio_to_wav_bytes(audio, sample_rate=48000)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

### 2.3 Gọi API từ các ngôn ngữ khác

#### Python (requests)

```python
import requests
import base64

BASE_URL = "http://localhost:8000"
HEADERS = {"x-api-key": "dev-secret-key"}

# Tổng hợp text đơn giản
response = requests.post(
    f"{BASE_URL}/v1/tts",
    headers=HEADERS,
    json={
        "text": "Xin chào, đây là API test!",
        "voice": "Xuan Vinh",
        "temperature": 0.8
    }
)
response.raise_for_status()

with open("output.wav", "wb") as f:
    f.write(response.content)

print(f"Audio duration: {response.headers.get('X-Audio-Duration')}s")

# Voice cloning qua API
with open("reference.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/v1/tts",
    headers=HEADERS,
    json={
        "text": "Đây là giọng clone!",
        "ref_audio_b64": audio_b64,
    }
)
with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

#### JavaScript / Node.js

```javascript
// Node.js với fetch (Node 18+) hoặc node-fetch
const fs = require("fs");

async function synthesize(text, voice = null) {
    const response = await fetch("http://localhost:8000/v1/tts", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-api-key": "dev-secret-key"
        },
        body: JSON.stringify({ text, voice })
    });
    
    if (!response.ok) {
        throw new Error(`TTS API lỗi: ${response.status}`);
    }
    
    const buffer = await response.arrayBuffer();
    return Buffer.from(buffer);
}

// Sử dụng
synthesize("Xin chào từ JavaScript!", "Ngoc Linh")
    .then(wavBuffer => fs.writeFileSync("output.wav", wavBuffer))
    .catch(console.error);

// Browser: phát audio ngay trong trang
async function playTTS(text) {
    const response = await fetch("http://localhost:8000/v1/tts", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-api-key": "dev-secret-key"
        },
        body: JSON.stringify({ text, voice: "Binh An" })
    });
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    await audio.play();
    // Giải phóng URL sau khi phát xong
    audio.onended = () => URL.revokeObjectURL(url);
}
```

#### cURL

```bash
# Tổng hợp text, lưu WAV
curl -X POST http://localhost:8000/v1/tts \
    -H "x-api-key: dev-secret-key" \
    -H "Content-Type: application/json" \
    -d '{"text": "Xin chào!", "voice": "Binh An"}' \
    --output output.wav \
    --silent \
    --show-error

# Kiểm tra health
curl http://localhost:8000/health

# Lấy danh sách voices
curl http://localhost:8000/v1/voices \
    -H "x-api-key: dev-secret-key" \
    | python -m json.tool

# OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "x-api-key: dev-secret-key" \
    -H "Content-Type: application/json" \
    -d '{"model": "vieneu-v3turbo", "input": "Hello from OpenAI API!", "voice": "fable"}' \
    --output openai_output.wav
```

#### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
)

type TTSRequest struct {
    Text        string  `json:"text"`
    Voice       string  `json:"voice,omitempty"`
    Temperature float64 `json:"temperature,omitempty"`
}

func synthesize(text, voice string) ([]byte, error) {
    reqBody, _ := json.Marshal(TTSRequest{
        Text:        text,
        Voice:       voice,
        Temperature: 0.8,
    })
    
    req, err := http.NewRequest(
        "POST",
        "http://localhost:8000/v1/tts",
        bytes.NewBuffer(reqBody),
    )
    if err != nil {
        return nil, err
    }
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-Api-Key", "dev-secret-key")
    
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 {
        return nil, fmt.Errorf("API error: %d", resp.StatusCode)
    }
    
    return io.ReadAll(resp.Body)
}

func main() {
    wavBytes, err := synthesize("Xin chào từ Go!", "Xuan Vinh")
    if err != nil {
        panic(err)
    }
    os.WriteFile("output.wav", wavBytes, 0644)
    fmt.Println("Đã lưu output.wav")
}
```

---

## PHẦN 3: Remote Mode (LMDeploy Server)

### 3.1 Kiến trúc Remote Mode

```
┌─────────────────────────────────────────────────────┐
│                     GPU Server                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  serve.py → lmdeploy api_server             │    │
│  │  Model: VieNeu-TTS-v2 (backbone LLM)        │    │
│  │  Port: 23333                                │    │
│  │  Protocol: OpenAI Chat Completions API       │    │
│  └─────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────┘
                         │ HTTP POST /v1/chat/completions
                         │ (speech tokens <|speech_N|>)
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼────────┐         ┌──────────▼───────┐
│  Client A (CPU)  │         │  Client B (CPU)  │
│  pip install     │         │  pip install     │
│  vieneu          │         │  vieneu          │
│  Codec: ONNX     │         │  Codec: ONNX     │
│  (local, nhẹ)    │         │  (local, nhẹ)    │
└──────────────────┘         └──────────────────┘
```

**Phân chia công việc:**
- **Server (GPU):** Chạy backbone LLM (VieNeu-TTS-v2), sinh speech tokens từ phoneme prompt
- **Client (CPU):** Phonemize text, encode reference audio, giải mã speech tokens bằng ONNX codec

### 3.2 Setup Server

```bash
# --- Cách 1: Docker (khuyến nghị cho production) ---
docker pull pnnbao/vieneu-tts:serve

docker run \
    --gpus all \
    -p 23333:23333 \
    -e HF_HOME=/root/.cache/huggingface \
    -v /data/huggingface:/root/.cache/huggingface \
    pnnbao/vieneu-tts:serve \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --model-name pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --memory-util 0.3

# --- Cách 2: Docker Compose (nhiều profile) ---
# File: docker/docker-compose.yml
MODEL=pnnbao-ump/VieNeu-TTS-v2 \
MEMORY_UTIL=0.5 \
SERVE_PORT=23333 \
docker compose -f docker/docker-compose.yml --profile serve up -d

# --- Cách 3: Từ source (môi trường đã cài vieneu[gpu]) ---
uv run python src/vieneu/serve.py \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --model-name pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --memory-util 0.3 \
    --tp 1

# Với GPU yếu (tiết kiệm VRAM hơn):
uv run python src/vieneu/serve.py \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --memory-util 0.2 \
    --quant-policy 4     # KV cache quantization: 0, 4, hoặc 8

# Với public tunnel (expose qua bore.pub):
uv run python src/vieneu/serve.py \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --tunnel              # Yêu cầu bore CLI đã cài
```

**Kiểm tra server đã hoạt động:**

```bash
# Server LMDeploy expose OpenAI-compatible API tại:
curl http://localhost:23333/v1/models
# Kết quả: {"data": [{"id": "pnnbao-ump/VieNeu-TTS-v2", ...}]}
```

### 3.3 Setup Client

```python
from vieneu import Vieneu
import numpy as np

# --- Client cực nhẹ: chỉ cần pip install vieneu (không cần GPU!) ---
# Codec (ONNX) chạy local, backbone LLM chạy trên server
tts = Vieneu(
    mode="remote",
    api_base="http://server-ip:23333/v1",
    model_name="pnnbao-ump/VieNeu-TTS-v2"
)

# --- Inference cơ bản ---
audio = tts.infer(
    text="Xin chào từ remote server!",
    voice=tts.get_preset_voice("Binh An"),
    temperature=1.0,
    top_k=50,
    repetition_penalty=1.2
)
tts.save(audio, "remote_output.wav")

# --- Voice cloning qua remote ---
# Client tự encode audio locally (codec ONNX), gửi codes lên server
ref_codes = tts.encode_reference("reference.wav")
audio = tts.infer(
    text="Giọng được clone và sinh trên server GPU.",
    ref_codes=ref_codes,
    ref_text="Transcript của audio reference."
)
tts.save(audio, "remote_cloned.wav")

# --- Batch async: gửi song song tất cả requests ---
import asyncio

async def remote_batch():
    texts = [f"Câu số {i} được sinh trên server." for i in range(1, 6)]
    
    audios = await tts.infer_batch_async(
        texts=texts,
        voice=tts.get_preset_voice("Ngoc Linh"),
        concurrency_limit=10    # Tối đa 10 requests song song
    )
    
    for i, audio in enumerate(audios):
        tts.save(audio, f"remote_batch_{i+1}.wav")
    print(f"Xử lý xong {len(audios)} files.")

asyncio.run(remote_batch())

# --- Streaming qua remote ---
for chunk in tts.infer_stream(
    text="Văn bản dài cần đọc theo streaming qua remote server.",
):
    # Mỗi chunk từ remote được overlap-add để ghép mượt
    pass  # xử lý chunk tại đây
```

### 3.4 Load Balancing nhiều server

```python
import random
import asyncio
from vieneu import Vieneu
from typing import List

# Danh sách GPU servers
GPU_SERVERS = [
    "http://gpu-server-1:23333/v1",
    "http://gpu-server-2:23333/v1",
    "http://gpu-server-3:23333/v1",
]

MODEL_NAME = "pnnbao-ump/VieNeu-TTS-v2"

def get_random_client() -> Vieneu:
    """Round-robin đơn giản: chọn server ngẫu nhiên."""
    return Vieneu(
        mode="remote",
        api_base=random.choice(GPU_SERVERS),
        model_name=MODEL_NAME
    )

class LoadBalancedTTS:
    """
    TTS client với load balancing theo round-robin.
    Tự động fallback sang server khác nếu một server lỗi.
    """
    def __init__(self, servers: List[str], model_name: str):
        self._servers = servers
        self._model_name = model_name
        self._idx = 0
        self._clients = [
            Vieneu(mode="remote", api_base=srv, model_name=model_name)
            for srv in servers
        ]
    
    def _next_client(self) -> Vieneu:
        """Round-robin."""
        client = self._clients[self._idx % len(self._clients)]
        self._idx += 1
        return client
    
    def infer(self, text: str, **kwargs) -> np.ndarray:
        """Thử từng server, fallback nếu lỗi."""
        last_error = None
        for _ in range(len(self._clients)):
            client = self._next_client()
            try:
                return client.infer(text, **kwargs)
            except Exception as e:
                last_error = e
                print(f"Server lỗi: {e}, thử server tiếp theo...")
        raise RuntimeError(
            f"Tất cả {len(self._clients)} servers đều lỗi. "
            f"Lỗi cuối: {last_error}"
        )

# Sử dụng
lb_tts = LoadBalancedTTS(GPU_SERVERS, MODEL_NAME)
audio = lb_tts.infer("Văn bản cần tổng hợp.", voice=None)
```

---

## PHẦN 4: Use Cases cụ thể

### 4.1 Web Application (Next.js + FastAPI)

Kiến trúc: Next.js frontend → FastAPI (api_server.py) → VieNeu SDK

```python
# Thêm vào api_server.py: endpoint streaming cho web player

from fastapi.middleware.cors import CORSMiddleware

# Cho phép Next.js frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

```javascript
// pages/api/tts.js (Next.js API route — proxy để che API key)
export default async function handler(req, res) {
    if (req.method !== "POST") {
        return res.status(405).end();
    }
    
    const { text, voice } = req.body;
    
    const response = await fetch("http://localhost:8000/v1/tts", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-api-key": process.env.VIENEU_API_KEY  // Giữ API key server-side
        },
        body: JSON.stringify({ text, voice })
    });
    
    if (!response.ok) {
        return res.status(response.status).json({ error: "TTS failed" });
    }
    
    const wavBuffer = await response.arrayBuffer();
    res.setHeader("Content-Type", "audio/wav");
    res.setHeader("Content-Disposition", "inline; filename=speech.wav");
    res.send(Buffer.from(wavBuffer));
}

// components/TTSPlayer.jsx
import { useState } from "react";

export default function TTSPlayer() {
    const [audioUrl, setAudioUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [text, setText] = useState("");
    
    async function handleSynthesize() {
        setLoading(true);
        try {
            const response = await fetch("/api/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, voice: "Binh An" })
            });
            const blob = await response.blob();
            // Giải phóng URL cũ trước khi tạo mới
            if (audioUrl) URL.revokeObjectURL(audioUrl);
            setAudioUrl(URL.createObjectURL(blob));
        } finally {
            setLoading(false);
        }
    }
    
    return (
        <div>
            <textarea value={text} onChange={e => setText(e.target.value)} />
            <button onClick={handleSynthesize} disabled={loading}>
                {loading ? "Đang tổng hợp..." : "Đọc văn bản"}
            </button>
            {audioUrl && <audio src={audioUrl} controls autoPlay />}
        </div>
    );
}
```

### 4.2 Chatbot TTS (LangChain / LLM pipeline)

```python
# Tích hợp VieNeu làm TTS output cho chatbot LangChain
from vieneu import Vieneu
import numpy as np

class VieneuTTSOutput:
    """
    Wrapper tích hợp VieNeu vào bất kỳ LLM pipeline nào.
    Gọi sau khi LLM sinh ra text response.
    """
    def __init__(self, voice: str = "Ngoc Linh", output_dir: str = "responses"):
        import os
        self.tts = Vieneu(mode="v3turbo")
        self.voice = voice
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._counter = 0
    
    def speak(self, text: str) -> str:
        """
        Tổng hợp text thành audio, lưu file, trả về đường dẫn.
        Dùng sau khi nhận response từ LLM.
        """
        # Làm sạch text: loại bỏ markdown nếu cần
        clean_text = self._strip_markdown(text)
        
        # Chia thành các câu ngắn nếu text quá dài
        audio = self.tts.infer(
            text=clean_text,
            voice=self.voice,
            max_chars=256
        )
        
        self._counter += 1
        output_path = f"{self.output_dir}/response_{self._counter:04d}.wav"
        self.tts.save(audio, output_path)
        
        return output_path
    
    def _strip_markdown(self, text: str) -> str:
        """Loại bỏ ký tự markdown trước khi TTS."""
        import re
        # Loại bỏ bold, italic, code blocks
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'#+\s', '', text)
        return text.strip()

# --- Ví dụ pipeline đơn giản (không phụ thuộc LangChain) ---
def simple_chatbot_with_tts():
    tts_output = VieneuTTSOutput(voice="Binh An")
    
    # Giả lập LLM responses
    llm_responses = [
        "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?",
        "Đây là câu trả lời cho câu hỏi của bạn...",
        "Cảm ơn bạn đã sử dụng dịch vụ của chúng tôi!",
    ]
    
    for response in llm_responses:
        print(f"LLM: {response}")
        audio_path = tts_output.speak(response)
        print(f"Audio: {audio_path}")
        # Phát audio, gửi cho client, v.v.

simple_chatbot_with_tts()
```

### 4.3 Dubbing Pipeline tự động (Video/SRT)

Pipeline hoàn chỉnh: đọc SRT → TTS từng dòng → ghép audio với FFmpeg timing:

```python
# dubbing_pipeline.py
import re
import os
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import List
from vieneu import Vieneu

@dataclass
class SubtitleEntry:
    index: int
    start_ms: int       # Milliseconds
    end_ms: int         # Milliseconds
    text: str
    duration_ms: int    # end - start

def parse_srt_time(time_str: str) -> int:
    """Chuyển '00:01:23,456' thành milliseconds."""
    h, m, rest = time_str.split(":")
    s, ms = rest.split(",")
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)

def parse_srt(srt_path: str) -> List[SubtitleEntry]:
    """Đọc và parse file SRT."""
    with open(srt_path, encoding="utf-8") as f:
        content = f.read()
    
    # Pattern match từng entry SRT
    pattern = (
        r"(\d+)\n"                          # Index
        r"(\d{2}:\d{2}:\d{2},\d{3})"       # Start time
        r" --> "
        r"(\d{2}:\d{2}:\d{2},\d{3})\n"     # End time
        r"((?:.+\n?)+)"                     # Text (có thể nhiều dòng)
    )
    entries = []
    for match in re.finditer(pattern, content.strip()):
        idx, start_str, end_str, text = match.groups()
        start_ms = parse_srt_time(start_str)
        end_ms = parse_srt_time(end_str)
        text = text.strip().replace("\n", " ")
        entries.append(SubtitleEntry(
            index=int(idx),
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            duration_ms=end_ms - start_ms
        ))
    return entries

def fit_audio_to_duration(
    audio: np.ndarray,
    target_duration_ms: int,
    sample_rate: int = 48000
) -> np.ndarray:
    """
    Điều chỉnh độ dài audio khớp với subtitle timing.
    Nếu audio ngắn hơn: pad silence ở cuối.
    Nếu audio dài hơn: cắt bớt (cảnh báo).
    """
    target_samples = int(target_duration_ms * sample_rate / 1000)
    current_samples = len(audio)
    
    if current_samples <= target_samples:
        # Pad silence để khớp timing
        silence = np.zeros(target_samples - current_samples, dtype=np.float32)
        return np.concatenate([audio, silence])
    else:
        # Audio dài hơn subtitle: cắt bớt (có thể cắt giữa câu)
        print(f"Cảnh báo: Audio ({current_samples/sample_rate:.2f}s) dài hơn "
              f"subtitle ({target_duration_ms/1000:.2f}s), sẽ cắt bớt.")
        return audio[:target_samples]

def dub_srt_file(
    srt_path: str,
    output_wav: str,
    voice: str = "Binh An",
    sample_rate: int = 48000
):
    """
    Tổng hợp toàn bộ SRT thành một file WAV với timing chính xác.
    
    Args:
        srt_path: Đường dẫn file .srt
        output_wav: Đường dẫn file WAV output
        voice: Tên preset voice
        sample_rate: Sample rate (48000 cho v3turbo)
    """
    tts = Vieneu(mode="v3turbo")
    entries = parse_srt(srt_path)
    
    if not entries:
        raise ValueError(f"Không tìm thấy subtitle entries trong {srt_path}")
    
    # Tính tổng độ dài audio cần tạo
    total_duration_ms = max(e.end_ms for e in entries)
    total_samples = int(total_duration_ms * sample_rate / 1000)
    
    # Tạo buffer silence cho toàn bộ track
    full_audio = np.zeros(total_samples, dtype=np.float32)
    
    print(f"Tổng hợp {len(entries)} subtitle entries...")
    print(f"Tổng thời lượng: {total_duration_ms/1000:.1f}s")
    
    for entry in entries:
        print(f"[{entry.index:03d}/{len(entries)}] "
              f"{entry.start_ms/1000:.2f}s → {entry.end_ms/1000:.2f}s: "
              f"{entry.text[:50]}")
        
        # Tổng hợp audio cho entry này
        audio = tts.infer(text=entry.text, voice=voice)
        
        # Fit vào thời lượng subtitle
        fitted = fit_audio_to_duration(audio, entry.duration_ms, sample_rate)
        
        # Đặt vào đúng vị trí trong buffer
        start_sample = int(entry.start_ms * sample_rate / 1000)
        end_sample = start_sample + len(fitted)
        
        if end_sample > total_samples:
            fitted = fitted[:total_samples - start_sample]
            end_sample = total_samples
        
        full_audio[start_sample:end_sample] += fitted
    
    # Normalize để tránh clipping
    peak = np.abs(full_audio).max()
    if peak > 1.0:
        full_audio /= peak
    
    # Lưu file
    sf.write(output_wav, full_audio, sample_rate)
    print(f"Đã lưu: {output_wav} ({total_duration_ms/1000:.1f}s, {total_samples} samples)")
    
    return output_wav

# Chạy pipeline
if __name__ == "__main__":
    dub_srt_file(
        srt_path="movie_vi.srt",
        output_wav="dubbed_audio.wav",
        voice="Binh An"
    )
```

### 4.4 Multi-speaker Script (làm phim/podcast)

```python
# multi_speaker.py — Render kịch bản nhiều nhân vật thành audio

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from vieneu import Vieneu

def render_script(script_json_path: str, output_dir: str = "rendered"):
    """
    Đọc script JSON có nhiều nhân vật, render thành audio riêng
    cho từng line và ghép thành track hoàn chỉnh.
    
    Format JSON:
    {
        "characters": {
            "narrator": {"voice": "Binh An"},
            "hero": {"voice": "Xuan Vinh", "ref_audio": "hero.wav"},
            "villain": {"voice": "Thai Son"}
        },
        "script": [
            {"character": "narrator", "text": "..."},
            {"character": "hero", "text": "..."}
        ]
    }
    """
    with open(script_json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    characters = data["characters"]
    script = data["script"]
    
    tts = Vieneu(mode="v3turbo")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Pre-encode reference audios nếu có (một lần, dùng lại)
    ref_codes_cache = {}
    for char_name, char_config in characters.items():
        if "ref_audio" in char_config:
            print(f"Encoding reference audio cho {char_name}...")
            ref_codes_cache[char_name] = tts.encode_reference(
                char_config["ref_audio"]
            )
    
    all_audio_parts = []
    sample_rate = 48000
    
    for i, line in enumerate(script):
        char = line["character"]
        text = line["text"]
        config = characters[char]
        
        infer_kwargs = {"text": text}
        
        # Dùng ref_codes đã encode nếu có
        if char in ref_codes_cache:
            infer_kwargs["ref_codes"] = ref_codes_cache[char]
        elif "voice" in config:
            infer_kwargs["voice"] = config["voice"]
        
        print(f"[{i+1:03d}] {char}: {text[:60]}")
        audio = tts.infer(**infer_kwargs)
        
        # Lưu line riêng lẻ
        line_path = f"{output_dir}/line_{i+1:03d}_{char}.wav"
        tts.save(audio, line_path)
        
        all_audio_parts.append(audio)
        
        # Thêm khoảng lặng giữa các dòng (0.5 giây)
        silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        all_audio_parts.append(silence)
    
    # Ghép toàn bộ
    full_track = np.concatenate(all_audio_parts)
    full_path = f"{output_dir}/full_track.wav"
    sf.write(full_track, full_path, sample_rate)
    print(f"\nHoàn thành! Full track: {full_path} "
          f"({len(full_track)/sample_rate:.1f}s)")

# Ví dụ file script.json
EXAMPLE_SCRIPT = {
    "characters": {
        "narrator": {"voice": "Binh An"},
        "hero":     {"voice": "Xuan Vinh"},
        "villain":  {"voice": "Thai Son"}
    },
    "script": [
        {"character": "narrator", "text": "Câu chuyện bắt đầu vào một buổi sáng mùa đông lạnh giá."},
        {"character": "hero",     "text": "Tôi sẽ bảo vệ mọi người khỏi bóng tối!"},
        {"character": "villain",  "text": "Đừng mơ tưởng! Không ai có thể ngăn cản ta."},
        {"character": "narrator", "text": "Hai bên đối đầu nhau trong im lặng căng thẳng."}
    ]
}

if __name__ == "__main__":
    import tempfile
    # Lưu script mẫu
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(EXAMPLE_SCRIPT, f, ensure_ascii=False, indent=2)
        script_path = f.name
    
    render_script(script_path, output_dir="film_audio")
```

---

## PHẦN 5: Best Practices và Checklist

### 5.1 Thread Safety

VieNeu **không thread-safe** — không được share một instance giữa nhiều thread:

```python
# SAI: Sẽ gây race condition và crash
tts = Vieneu()
threads = [threading.Thread(target=lambda: tts.infer("...")) for _ in range(4)]
# Chạy sẽ fail hoặc cho kết quả sai

# ĐÚNG: Thread-local — mỗi thread tạo instance riêng
from threading import local
_local = local()

def get_tts():
    if not hasattr(_local, "tts"):
        _local.tts = Vieneu(mode="v3turbo")
    return _local.tts

def worker(text):
    tts = get_tts()  # Lấy instance riêng của thread này
    return tts.infer(text)

# ĐÚNG: Queue + single worker (an toàn nhất)
import queue

work_queue = queue.Queue()
result_store = {}

def tts_worker():
    tts = Vieneu()  # Một instance duy nhất trong worker thread
    while True:
        item = work_queue.get()
        if item is None:
            break
        req_id, text, kwargs = item
        result_store[req_id] = tts.infer(text, **kwargs)
        work_queue.task_done()

worker_thread = threading.Thread(target=tts_worker, daemon=True)
worker_thread.start()

# Submit jobs
work_queue.put(("req1", "Câu đầu tiên.", {}))
work_queue.put(("req2", "Câu thứ hai.", {"voice": "Xuan Vinh"}))
work_queue.join()  # Đợi xử lý xong
```

### 5.2 Memory Management

```python
from vieneu import Vieneu

# --- GPU memory ---
tts = Vieneu(mode="v3turbo", device="cuda")

# Sau batch lớn: clear VRAM
import torch
import gc

def cleanup_gpu():
    """Giải phóng VRAM sau inference nặng."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Xử lý 100 texts → cleanup mỗi 10 batch
for i in range(0, 100, 10):
    batch = texts[i:i+10]
    audios = tts.infer_batch(batch)
    # Xử lý audios...
    cleanup_gpu()

# --- Chunk size: tối đa 256 chars ---
# Vượt quá sẽ bị tự động chia nhỏ, nhưng mỗi chunk tốt nhất ≤ 256
long_text = "..." * 100  # Text rất dài
# infer() tự chia thành chunks ≤ max_chars và ghép lại
audio = tts.infer(long_text, max_chars=256)

# --- Tránh load lại model không cần thiết ---
# SAI: Tạo Vieneu mới mỗi lần
def bad_synthesize(text):
    tts = Vieneu()  # Load model từ đầu mỗi lần gọi — rất chậm!
    return tts.infer(text)

# ĐÚNG: Singleton hoặc module-level instance
_global_tts = None

def good_synthesize(text):
    global _global_tts
    if _global_tts is None:
        _global_tts = Vieneu()  # Chỉ load một lần
    return _global_tts.infer(text)
```

### 5.3 Model Caching

```python
import os

# Thiết lập HF_HOME TRƯỚC KHI import vieneu
# để model download về thư mục cố định
os.environ["HF_HOME"] = "/data/huggingface_cache"

from vieneu import Vieneu

# Lần đầu: download model (~800MB cho v3turbo)
tts = Vieneu()

# Lần sau: load từ cache, không cần internet
# Kiểm tra cache:
cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
print(f"Models cached tại: {cache_dir}")

# Offline mode: ngăn mọi network request
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Truyền hf_token nếu model private
tts = Vieneu(
    mode="v3turbo",
    hf_token=os.environ.get("HF_TOKEN")
)
```

### 5.4 Testing với VieNeu

```python
# tests/test_my_tts_feature.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# --- Unit test: mock hoàn toàn, không chạy inference thật ---
def test_synthesize_function():
    """Test logic xung quanh TTS mà không tốn thời gian inference."""
    mock_audio = np.zeros(48000, dtype=np.float32)  # 1 giây silence
    
    with patch("vieneu.Vieneu") as MockVieneu:
        mock_instance = MockVieneu.return_value
        mock_instance.infer.return_value = mock_audio
        mock_instance.list_preset_voices.return_value = [
            ("Bình An (Nam)", "Binh An"),
            ("Ngọc Linh (Nữ)", "Ngoc Linh"),
        ]
        
        # Test code của bạn
        from vieneu import Vieneu
        tts = Vieneu()
        audio = tts.infer("Test text")
        
        assert audio is not None
        assert len(audio) == 48000
        mock_instance.infer.assert_called_once_with("Test text")

# --- Integration test: dùng thật với text ngắn ---
@pytest.mark.integration
def test_real_inference():
    """
    Chạy inference thật, đánh dấu là integration test.
    Chỉ chạy khi có đủ môi trường: pytest -m integration
    """
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")  # CPU ONNX, không cần GPU
    
    # Text ngắn để test nhanh
    audio = tts.infer("Test.")
    
    assert audio is not None
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert len(audio) < 48000 * 5  # Không quá 5 giây cho 1 từ

@pytest.mark.integration
def test_preset_voices():
    """Kiểm tra danh sách voices khớp với expectation."""
    from vieneu import Vieneu
    tts = Vieneu()
    
    voices = tts.list_preset_voices()
    voice_ids = [vid for _, vid in voices]
    
    # Kiểm tra các voices quan trọng đều có mặt
    expected = ["Binh An", "Ngoc Linh", "Xuan Vinh"]
    for voice_id in expected:
        assert any(voice_id in vid for vid in voice_ids), \
            f"Voice '{voice_id}' không tìm thấy trong preset list"

# Chạy test:
# pytest tests/                    # Chỉ unit tests (không cần model)
# pytest tests/ -m integration     # Bao gồm cả integration tests
```

### 5.5 Production Checklist

Trước khi deploy VieNeu-TTS lên production, kiểm tra các mục sau:

**Môi trường và cài đặt:**
- [ ] Set `HF_HOME` để cache models ở thư mục cố định, tránh re-download
- [ ] Đặt `VIENEU_API_KEY` trong environment, không hardcode trong code
- [ ] Cài đủ system dependencies: `espeak-ng` cho phonemization
- [ ] Test offline mode sau khi download models xong

**Inference và hiệu năng:**
- [ ] Giới hạn `max_chars` <= 256 chars mỗi chunk
- [ ] Dùng thread-local Vieneu instances trong multi-threaded app
- [ ] Singleton pattern: không tạo Vieneu mới mỗi request
- [ ] GPU: monitor VRAM usage, set `memory_util` phù hợp (0.3–0.5)

**Độ tin cậy:**
- [ ] Implement retry với exponential backoff
- [ ] Error handling cho GPU OOM (clear cache và retry)
- [ ] Graceful shutdown: đảm bảo `tts.close()` được gọi khi app stop
- [ ] Health check endpoint trả về status engine

**API và bảo mật:**
- [ ] API key rotation mechanism (đổi key không cần restart)
- [ ] Rate limiting để ngăn abuse (vd: max 10 req/s per client)
- [ ] Log latency mỗi request để phát hiện bottleneck
- [ ] Validate và sanitize input text trước khi inference

**Monitoring:**
- [ ] Alert khi GPU memory > 80%
- [ ] Metric: P50/P95/P99 latency per request
- [ ] Log audio duration vs inference time (real-time factor)
- [ ] Track error rate và loại lỗi phổ biến

### 5.6 Watermarking

VieNeu-TTS tích hợp Perth watermark để đánh dấu audio AI-generated:

```python
# perth watermark được nhúng tự động nếu cài package perth
# pip install perth

from vieneu import Vieneu

# Watermark ON (mặc định nếu perth đã cài)
tts = Vieneu()
audio_watermarked = tts.infer(
    "Đây là audio có watermark.",
    apply_watermark=True  # Mặc định True
)

# Tắt watermark (vd: khi cần xử lý audio thêm sau đó)
audio_raw = tts.infer(
    "Audio không có watermark.",
    apply_watermark=False
)

# Kiểm tra xem watermark có được nhúng không:
# (perth không cài hoặc watermarker=None → bỏ qua silently)
print(f"Watermarker active: {tts.watermarker is not None}")
```

**Lưu ý:** Perth watermark là implicit watermark (vô hình với tai người), nhúng trực tiếp vào waveform. Không làm giảm chất lượng audio đáng kể nhưng cho phép phát hiện AI-generated content sau này.
