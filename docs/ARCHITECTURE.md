# VieNeu-TTS — Kiến trúc Kỹ thuật & Chiến lược Hệ thống

> Phiên bản tài liệu: v3.0.5 | Giấy phép: Apache 2.0 | Ngày cập nhật: 2026-06-23

---

## 1. Tổng quan dự án

**VieNeu-TTS** là một hệ thống Text-to-Speech (TTS) tiếng Việt tiên tiến, được thiết kế để chạy trực tiếp trên thiết bị người dùng (on-device) mà không phụ thuộc vào dịch vụ đám mây. Dự án nhắm đến cả hai nhóm người dùng: nhà phát triển tích hợp SDK Python và người dùng cuối thông qua giao diện web Gradio.

### Mục tiêu thiết kế

- **Torch-free trên CPU**: Phiên bản v3 Turbo hoạt động hoàn toàn không cần PyTorch khi chạy trên CPU, chỉ cần ONNX Runtime — giảm đáng kể kích thước cài đặt (vài trăm MB thay vì vài GB).
- **Voice Cloning tức thì**: Người dùng cung cấp một đoạn audio mẫu ngắn (5–30 giây); hệ thống encode sang MOSS codec tokens và tái tạo giọng nói đó cho bất kỳ văn bản nào.
- **Song ngữ (Bilingual)**: Hỗ trợ văn bản hỗn hợp Việt–Anh thông qua pipeline sea-g2p.
- **Streaming real-time**: Xuất âm thanh theo từng chunk nhỏ ngay trong lúc model đang sinh token, giảm thời gian phản hồi cảm nhận của người dùng.
- **Chất lượng 48kHz**: Model v3 Turbo xuất âm thanh 48.000 Hz — tiêu chuẩn phát thanh chuyên nghiệp.

### Thông tin phiên bản

| Thuộc tính | Giá trị |
|---|---|
| Tên package | `vieneu` |
| Phiên bản hiện tại | 3.0.5 |
| Giấy phép | Apache 2.0 |
| Python tối thiểu | 3.10 |
| Nền tảng | Windows, Linux, macOS (Apple Silicon) |
| Trạng thái | Beta (Development Status 4) |

### Điểm mạnh chính

1. **Cài đặt tối giản**: `pip install vieneu` chỉ kéo các dependency torch-free (sea-g2p, onnxruntime, soundfile, soxr, tokenizers). Phù hợp môi trường server hạn chế dung lượng.
2. **Voice Cloning không cần fine-tuning**: Encode reference audio → MOSS codec → sinh giọng tương tự tức thì.
3. **Bilingual**: Nhận đầu vào tiếng Việt lẫn tiếng Anh trong cùng một chuỗi.
4. **Streaming với overlap-add**: Thuật toán linear overlap-add đảm bảo không có artifact khi ghép các chunk audio.
5. **Watermarking bảo mật**: Thư viện `perth` nhúng watermark vô hình vào audio đầu ra.

---

## 2. Tech Stack Chi tiết

| Component | Technology | Phiên bản | Mục đích | Ghi chú |
|---|---|---|---|---|
| Ngôn ngữ | Python | >=3.10 | Toàn bộ logic SDK và server | Type hints đầy đủ |
| Phonemizer | sea-g2p | >=0.7.6 | Normalize + G2P tiếng Việt/Anh | Viết bằng Rust, không cần torch |
| Inference CPU | ONNX Runtime | >=1.20.0 | Chạy model v3 Turbo và MOSS codec trên CPU | Torch-free path |
| Inference GPU | PyTorch | 2.8.0 | Chạy v3 Turbo GPU và các backbone v1/v2 | Optional, trong extra [gpu] |
| LLM Server | LMDeploy (TurbomindEngine) | 0.11.0+cu128 | GPU inference với batching tối ưu | Chỉ Linux/Windows CUDA |
| Web UI | Gradio | >=5.49.1 | Giao diện demo và production web | Entry point: `vieneu-web` |
| G2P Pipeline | sea-g2p SEAPipeline | >=0.7.6 | Chuẩn hoá văn bản + chuyển sang phoneme | Hỗ trợ `punc_norm` |
| Audio I/O | soundfile | latest | Đọc/ghi file WAV, FLAC, OGG | Dùng libsndfile |
| Resampling | soxr | latest | Resample reference audio về 16kHz | Thay thế librosa khi không có [gpu] |
| Tokenization | tokenizers (HuggingFace) | >=0.20 | Tokenizer cho phoneme stream | Torch-free |
| Model Hub | huggingface_hub | latest | Tải model weights, voices.json, ONNX files | Hỗ trợ cache offline |
| Watermarking | perth | >=0.2.0 | Nhúng watermark vô hình vào audio | Skip gracefully nếu không cài |
| Neural Codec | neucodec | >=0.0.4 | Encode/decode speech tokens (v1/v2) | Cần PyTorch |
| Neural Codec ONNX | neucodec-onnx-decoder-int8 | latest | Codec ONNX int8 (torch-free) | Dùng cho standard CPU path |
| MOSS Codec | MOSS-Audio-Tokenizer-Nano | - | Codec 16 VQ cho v3 Turbo, 48kHz | Encoder ONNX + decoder ONNX |
| GGUF Inference | llama-cpp-python | 0.3.16 | Chạy model GGUF quantized (turbo/standard) | Prebuilt wheel cho Win/Mac |
| Build tool | uv | latest | Package manager hiện đại, lock file | index-strategy: unsafe-best-match |
| Container | Docker + NVIDIA CUDA | cu128 | Deploy GPU service | Dockerfile riêng cho GPU |
| Tunnel | bore | latest | Expose local server ra public URL | `bore.pub` tunnel miễn phí |
| Triton (Linux) | triton | GPU | Compile codec kernel cho inference nhanh hơn | Không hỗ trợ Windows native |
| LoRA | PEFT | latest | Load LoRA adapter cho fine-tuning | Chỉ PyTorch backbone |
| YAML config | PyYAML | latest | Cấu hình Gradio UI (config.yaml) | Backbone và codec mapping |

---

## 3. Kiến trúc Hệ thống

### 3.1 Sơ đồ kiến trúc tổng thể (ASCII)

```
+---------------------------------------------------------------------+
|                         UI LAYER                                    |
|                                                                     |
|  +--------------------+    +--------------------------------------+ |
|  |   Gradio Web UI    |    |      External Application            | |
|  |  (vieneu-web CLI)  |    |  (Python script, Jupyter, FastAPI)   | |
|  |  config.yaml       |    |                                      | |
|  +--------+-----------+    +---------------+----------------------+ |
+-----------|-------------------------------|--------------------------|
            |                              |
            v                              v
+---------------------------------------------------------------------+
|                         SDK LAYER                                   |
|                                                                     |
|   from vieneu import Vieneu                                         |
|                                                                     |
|   +----------------------------------------------------------------+|
|   |              Vieneu() -- Factory Function                      ||
|   |  factory.py: mode= dispatcher -> concrete engine class         ||
|   +------------------------------+---------------------------------+|
|                                  |                                  |
|   +------------------------------v---------------------------------+|
|   |              BaseVieneuTTS (base.py)                          ||
|   |  - _load_voices() / _load_codec()                              ||
|   |  - _format_prompt() / _decode()                                ||
|   |  - encode_reference() / _apply_watermark()                     ||
|   |  - infer() / infer_batch() [abstract]                          ||
|   +--+----------+----------+-----------+----------+---------------+|
|      |          |           |           |          |               |
|  V3TurboVie  FastVieneu  VieNeuTTS  TurboVieneu RemoteVieneu      |
|  NeuTTS      TTS         (Standard)  TTS/GPU     TTS              |
|  (v3turbo)   (fast/gpu)  (standard)  (turbo/     (remote/api)     |
|                                      turbo_gpu)                    |
|                                                                     |
|   vieneu_utils/                                                     |
|   +-- phonemize_text.py  (SEAPipeline, G2P, PuncNormalizer)        |
|   +-- core_utils.py      (chunking, join_audio_chunks)             |
+---------------------------------------------------------------------+
            |                              |
            v                              v
+---------------------------------------------------------------------+
|                      INFERENCE LAYER                                |
|                                                                     |
|  +--------------+  +--------------+  +--------------------------+  |
|  |  ONNX Files  |  |  PyTorch     |  |  LMDeploy TurbomindEngine|  |
|  |  (v3 CPU)    |  |  Weights     |  |  (OpenAI-compat API)     |  |
|  |  + MOSS codec|  |  (v3 GPU,    |  |  POST /v1/chat/          |  |
|  |  ONNX        |  |   v1, v2)    |  |  completions             |  |
|  +--------------+  +--------------+  +--------------------------+  |
|                                                                     |
|  +--------------+  +--------------+                                 |
|  |  GGUF Model  |  |  HuggingFace |                                 |
|  |  (llama.cpp) |  |  Hub / Cache |                                 |
|  +--------------+  +--------------+                                 |
+---------------------------------------------------------------------+
```

### 3.2 Factory Pattern — Vieneu()

File `src/vieneu/factory.py` cung cấp một factory function duy nhất làm entry point cho toàn bộ SDK:

```python
from vieneu import Vieneu

# Mac dinh: v3 Turbo, tu dong chon CPU (ONNX) hoac GPU (PyTorch)
tts = Vieneu(mode="v3turbo")

# GPU LMDeploy (can pip install vieneu[gpu])
tts = Vieneu(mode="fast")

# Standard (PyTorch/GGUF)
tts = Vieneu(mode="standard")

# Remote API Client
tts = Vieneu(mode="remote", api_base="http://server:23333/v1")
```

**Bảng mapping mode → class:**

| mode string | Alias | Class được tạo | File |
|---|---|---|---|
| `"v3turbo"` | *(mặc định)* | `V3TurboVieNeuTTS` | `v3turbo.py` |
| `"remote"` | `"api"` | `RemoteVieNeuTTS` | `remote.py` |
| `"fast"` | `"gpu"` | `FastVieNeuTTS` | `fast.py` |
| `"turbo"` | - | `TurboVieNeuTTS` (GGUF CPU) | `turbo.py` |
| `"turbo_gpu"` | - | `TurboGPUVieNeuTTS` | `turbo.py` |
| `"xpu"` | - | `XPUVieNeuTTS` | `core_xpu.py` |
| `"standard"` | - | `VieNeuTTS` | `standard.py` |

### 3.3 Phân cấp Class

```
BaseVieneuTTS (ABC)                              <- base.py
+-- V3TurboVieNeuTTS                             <- v3turbo.py
|   +-- engine: OnnxV3LiteEngine (CPU path)      <- _v3_turbo_engine/onnx_runtime_lite.py
|   +-- engine: VieNeuTTSv3Turbo (GPU path)      <- _v3_turbo_engine/__init__.py
+-- FastVieNeuTTS                                <- fast.py
|   +-- backbone: LMDeploy pipeline
|   +-- codec: DistillNeuCodec (PyTorch)
+-- VieNeuTTS (Standard)                         <- standard.py
|   +-- backbone: Llama (GGUF) | AutoModelForCausalLM
|   +-- codec: NeuCodecOnnxDecoder (ONNX) | NeuCodec (PyTorch)
+-- BaseTurboVieNeuTTS                           <- turbo.py (internal base)
|   +-- TurboVieNeuTTS (GGUF CPU)
|   +-- TurboGPUVieNeuTTS (PyTorch/LMDeploy GPU)
+-- RemoteVieNeuTTS                              <- remote.py
    +-- (codec local, backbone = HTTP requests)
```

### 3.4 Dependency giữa các module

```
factory.py
    +-- (lazy import) v3turbo.py / fast.py / standard.py / turbo.py / remote.py
            +-- base.py
                    +-- vieneu_utils.phonemize_text (SEAPipeline, G2P, PuncNormalizer)
                    +-- vieneu_utils.core_utils (split_text_into_chunks, join_audio_chunks)
                    +-- huggingface_hub (tai voices.json, weights)
                    +-- perth (watermarking, optional)
```

Tất cả import nặng (torch, lmdeploy, llama_cpp, neucodec) đều là **lazy import** — chỉ được gọi khi class tương ứng được khởi tạo. Điều này giúp `import vieneu` nhanh và không gây lỗi ImportError với các package chưa cài.

---

## 4. Các Engine TTS Chi tiết

### 4.1 V3TurboVieNeuTTS — Engine mặc định

**Mục đích**: Engine mới nhất, hỗ trợ 48kHz, tag cảm xúc, và chế độ CPU hoàn toàn không cần PyTorch (ONNX Runtime).

**Khởi tạo:**
```python
from vieneu import Vieneu

# Tu dong: CPU -> ONNX, GPU -> PyTorch
tts = Vieneu(mode="v3turbo")

# Bat buoc dung ONNX (CPU torch-free)
tts = Vieneu(mode="v3turbo", backend="onnx")

# Bat buoc dung PyTorch (can GPU + pip install vieneu[gpu])
tts = Vieneu(mode="v3turbo", device="cuda", backend="pytorch")

# Voice cloning voi audio mau
wav = tts.infer("Xin chao the gioi", ref_audio="speaker.wav")

# Dung preset voice
wav = tts.infer("Xin chao", voice="Ngoc Linh")

# Tag cam xuc
wav = tts.infer("[cuoi] That thu vi qua!", voice="Ngoc Linh")
```

**Tham số quan trọng:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `backbone_repo` | `pnnbao-ump/VieNeu-TTS-v3-Turbo` | HF repo chứa model weights / ONNX |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"` |
| `backend` | `"auto"` | `"auto"`, `"onnx"`, `"pytorch"` |
| `temperature` | 0.8 | Độ ngẫu nhiên trong sinh âm thanh |
| `top_k` | 25 | Top-K sampling |
| `top_p` | 0.95 | Top-P (nucleus) sampling |
| `max_new_frames` | 300 | Số frame tối đa cho một chunk |
| `repetition_penalty` | 1.2 | Phạt token lặp |
| `max_chars` | 256 | Giới hạn ký tự mỗi chunk |
| `silence_p` | 0.15 | Thêm 0.15 giây silence giữa các chunk |
| `crossfade_p` | 0.0 | Crossfade giữa các chunk (giây) |

**Ưu điểm**: Chất lượng cao nhất (48kHz), torch-free CPU, emotion tags, voice cloning, sample rate cao nhất.

**Nhược điểm**: Phiên bản còn đang thử nghiệm (early access), streaming chưa ổn định.

**Phù hợp nhất**: Ứng dụng production mới, cần chất lượng cao, chạy trên bất kỳ máy nào có Python.

---

### 4.2 FastVieNeuTTS — Engine GPU tốc độ cao

**Mục đích**: Tối ưu cho GPU NVIDIA, dùng LMDeploy TurbomindEngine để đạt throughput cao nhất. Hỗ trợ batching song song và Triton kernel.

**Khởi tạo:**
```python
from vieneu import Vieneu

tts = Vieneu(
    mode="fast",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
    backbone_device="cuda",
    codec_repo="neuphonic/distill-neucodec",
    codec_device="cuda",
    memory_util=0.3,    # 30% VRAM cho KV cache
    tp=1,               # Tensor parallel size
    max_batch_size=4,
)

# Infer don
wav = tts.infer("Xin chao", ref_audio="speaker.wav")

# Batch processing
wavs = tts.infer_batch(["Cau mot.", "Cau hai.", "Cau ba."], ref_audio="speaker.wav")

# Streaming
for chunk in tts.infer_stream("Day la van ban dai...", ref_audio="speaker.wav"):
    play_audio(chunk)
```

**Tham số quan trọng:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `memory_util` | 0.3 | Tỷ lệ VRAM dành cho KV cache |
| `tp` | 1 | Tensor parallel (số GPU) |
| `enable_prefix_caching` | False | Cache prefix prompt |
| `quant_policy` | 0 | KV quantization: 0, 4, hoặc 8 bit |
| `enable_triton` | True | Compile codec với Triton (Linux only) |
| `max_batch_size` | 4 | Batch size tối đa |

**Ưu điểm**: Throughput cao nhất cho GPU, batching thực sự song song, streaming với overlap-add.

**Nhược điểm**: Chỉ chạy CUDA, yêu cầu LMDeploy (không hỗ trợ macOS), Triton không chạy trên Windows.

**Phù hợp nhất**: Production server GPU, xử lý nhiều request đồng thời.

---

### 4.3 VieNeuTTS (Standard) — Engine linh hoạt CPU/GPU

**Mục đích**: Backend linh hoạt nhất, hỗ trợ cả PyTorch full-precision và GGUF quantized (qua llama-cpp-python). Dùng cho CPU hoặc GPU với yêu cầu thấp hơn.

**Khởi tạo:**
```python
from vieneu import Vieneu

# GGUF quantized (mac dinh, tiet kiem RAM)
tts = Vieneu(
    mode="standard",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec-onnx-decoder-int8",
    gguf_filename="VieNeu-TTS-v2-Q4-K-M.gguf",
)

# PyTorch full precision
tts = Vieneu(
    mode="standard",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
    backbone_device="cuda",
    codec_repo="neuphonic/distill-neucodec",
    codec_device="cuda",
    gguf_filename=None,  # Dung PyTorch
)

# LoRA adapter (chi voi PyTorch backbone)
tts.load_lora_adapter("my-org/my-lora-adapter")
```

**Ưu điểm**: Hỗ trợ LoRA, GGUF giảm RAM đáng kể, batch inference với PyTorch.

**Nhược điểm**: Chậm hơn FastVieNeuTTS trên GPU, GGUF không hỗ trợ LoRA.

**Phù hợp nhất**: Thử nghiệm fine-tuning, máy CPU mạnh, môi trường không có LMDeploy.

---

### 4.4 TurboVieNeuTTS — Engine GGUF CPU/GPU (v2 Turbo)

**Mục đích**: Phiên bản Turbo của v2, dùng kiến trúc speaker embedding thay vì voice cloning bằng reference transcript. Chạy GGUF bằng llama-cpp trên CPU.

**Khởi tạo:**
```python
from vieneu import Vieneu

# GGUF tren CPU (TurboVieNeuTTS)
tts = Vieneu(
    mode="turbo",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF",
    device="cpu",
)

# PyTorch/LMDeploy tren GPU (TurboGPUVieNeuTTS)
tts = Vieneu(
    mode="turbo_gpu",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2-Turbo",
    device="cuda",
    backend="lmdeploy",  # hoac "standard"
)

wav = tts.infer("Xin chao", voice=my_voice_embedding)
```

**Điểm khác biệt so với Standard**: Turbo sử dụng `speaker embedding` (vector 128 chiều) thay vì reference codes + transcript. Decoder ONNX nhận `content_ids` + `voice_embedding` trực tiếp.

**Ưu điểm**: Nhanh hơn Standard trên CPU, decoder ONNX tích hợp sẵn voice embedding.

**Nhược điểm**: Voice cloning chất lượng thấp hơn v3 Turbo, sample rate 24kHz.

**Phù hợp nhất**: CPU inference không cần PyTorch với v2 model.

---

### 4.5 RemoteVieNeuTTS — Client API

**Mục đích**: Client kết nối đến LMDeploy server từ xa. Codec chạy local (CPU), backbone chạy remote. Hỗ trợ async và streaming qua SSE.

**Khởi tạo:**
```python
from vieneu import Vieneu

tts = Vieneu(
    mode="remote",
    api_base="http://your-server:23333/v1",
    model_name="pnnbao-ump/VieNeu-TTS",
    codec_repo="neuphonic/distill-neucodec",
    codec_device="cpu",
)

# Sync inference
wav = tts.infer("Xin chao", ref_audio="speaker.wav")

# Async inference (xu ly nhieu chunk song song)
import asyncio
wav = asyncio.run(tts.infer_async("Van ban dai...", ref_audio="speaker.wav"))

# Batch async voi concurrency limit
wavs = tts.infer_batch(["Cau mot.", "Cau hai."], ref_audio="speaker.wav")

# Streaming SSE
for chunk in tts.infer_stream("Van ban...", ref_audio="speaker.wav"):
    play_audio(chunk)
```

**Ưu điểm**: Không cần GPU trên client, xử lý multi-chunk song song bằng asyncio.gather, hỗ trợ bore tunnel cho public URL.

**Nhược điểm**: Phụ thuộc network, latency cao hơn local, cần thiết lập server riêng.

**Phù hợp nhất**: Laptop/desktop không có GPU muốn dùng server GPU từ xa.

---

### 4.6 XPUVieNeuTTS — Intel GPU (Beta)

**Mục đích**: Hỗ trợ Intel GPU thông qua `torch.xpu`. Trạng thái: Beta, chưa ổn định.

**Khởi tạo:**
```python
from vieneu import Vieneu

# Tu dong raise RuntimeError neu driver Intel XPU khong dung
tts = Vieneu(mode="xpu")
```

**Ưu điểm**: Tận dụng Intel Arc GPU và Intel Data Center GPU.

**Nhược điểm**: Beta, ít được test, yêu cầu driver Intel XPU đặc biệt.

---

### 4.7 Khởi động server từ xa (serve.py)

`serve.py` cung cấp CLI để khởi động LMDeploy server:

```bash
# Khoi dong server co ban
vieneu-serve --model pnnbao-ump/VieNeu-TTS-v2 --port 23333

# Voi tunnel public (can cai bore)
vieneu-serve --model pnnbao-ump/VieNeu-TTS-v2 --tunnel

# Multi-GPU (Tensor Parallel)
vieneu-serve --model pnnbao-ump/VieNeu-TTS-v2 --tp 2

# KV cache quantization
vieneu-serve --model pnnbao-ump/VieNeu-TTS-v2 --quant-policy 8
```

---

## 5. Pipeline Xử lý Âm thanh End-to-End

### 5.1 Text Normalization & Phonemization

Toàn bộ pipeline G2P được delegate cho thư viện `sea-g2p` (viết bằng Rust), đảm bảo tốc độ cao và không cần PyTorch.

#### Các bước xử lý trong `phonemize_text.py`:

```
Input text (raw)
    |
    v
[1] split_text_into_chunks() -- core_utils.py
    Cat text thanh chunks <= max_chars=256 ky tu
    Uu tien tach tai dau cau cuoi cau, sau do dau phay
    |
    v
[2] PuncNormalizer.normalize(chunk, punc_norm=True) -- phonemize_text.py
    sea_g2p.Normalizer: chuan hoa so, viet tat, dau cau
    punc_norm=True: them "." vao cuoi cau ngan (<5 tu) hoac thieu dau ket thuc
    |
    v
[3] Xu ly emotion tags (v3 Turbo)
    _EMOTION_SPLIT_RE.split(text) -> tach tag [cuoi]/[sigh]/...
    _EMOTION_TAG_TO_K mapping -> <|emotion_1|> / <|emotion_2|> / <|emotion_3|>
    |
    v
[4] SEAPipeline.run() / G2P.phonemize_batch() -- sea-g2p
    Chuyen doi grapheme -> phoneme cho tieng Viet va tieng Anh
    Output: chuoi IPA-like phones
    |
    v
[5] _ensure_terminal_punct() -- chot dau cau cuoi chunk
    Dam bao ket thuc bang ".", "!" hoac "?"
    |
    v
Output: phoneme string san sang cho model inference
```

#### Bảng mapping Emotion Tag:

| Tag trong text (tiếng Anh) | Tag trong text (tiếng Việt) | Token đầu ra | Ý nghĩa |
|---|---|---|---|
| `[chuckle]` | `[cười]`, `[cuoi]` | `<\|emotion_1\|>` | Tiếng cười nhẹ |
| `[sigh]` | `[thở dài]`, `[tho dai]` | `<\|emotion_2\|>` | Tiếng thở dài |
| `[clear throat]` | `[hắng giọng]`, `[hang giong]` | `<\|emotion_3\|>` | Hắng giọng |

Ví dụ sử dụng:
```python
text = "[cuoi] Ban co biet khong? [tho dai] Kho lam day."
# Sau phonemize_text_with_emotions():
# "<|emotion_1|> ban ko biet xong. <|emotion_2|>. xo lam day."
```

#### Chiến lược chunking (split-first approach):

Chiến lược quan trọng được áp dụng từ v3.0.5: **cắt text TRƯỚC khi normalize**, không phải sau. Lý do:

- Nếu normalize toàn bộ văn bản rồi mới cắt, một chunk có thể kết thúc bằng dấu "," lửng.
- Khi cắt trước, mỗi chunk là một đơn vị độc lập, và `punc_norm=True` thêm dấu kết thúc hợp lệ cho từng chunk.

```python
# split_text_into_chunks() -- uu tien tach theo thu tu:
# 1. Dau cau cuoi cau (. ! ?)
# 2. Dau cau nhe (, ; : - -)
# 3. Dau cach giua tu
# 4. Hard cut tai max_chars ky tu
```

#### Hỗ trợ song ngữ (Bilingual):

sea-g2p tự động nhận diện ngôn ngữ ở mức từng token:
- Từ tiếng Việt → phoneme tiếng Việt
- Từ tiếng Anh → phoneme tiếng Anh (CMU dict hoặc G2P)
- Số → đọc theo ngữ cảnh ngôn ngữ

---

### 5.2 Inference Flow

#### CPU ONNX Path (v3 Turbo):

```
phonemes (string)
    |
    v
[1] OnnxV3LiteEngine._tokenize(phonemes)
    -> Tokenizer ONNX (tokenizers library, torch-free)
    -> input_ids: np.ndarray [1, T]
    |
    v
[2] MOSS Encoder ONNX (ref_codes -> context)
    ref_codes: np.ndarray (T_ref, 16)  [16 VQ codebooks]
    -> ref_context: np.ndarray
    |
    v
[3] Autoregressive generation (ONNX inference session)
    - Vong lap sinh tung frame MOSS token
    - Sampling: temperature=0.8, top_k=25, top_p=0.95
    - Dung khi gap EOS hoac dat max_new_frames=300
    - Output: speech_ids list
    |
    v
[4] MOSS Decoder ONNX
    codes: np.ndarray [1, 1, T] (int32)
    -> waveform: np.ndarray [T_audio] (float32, 48kHz)
    |
    v
[5] perth watermarking (optional)
    -> final_wav: np.ndarray (float32, 48kHz)
```

#### GPU PyTorch Path (v3 Turbo / Fast):

```
phonemes (string)
    |
    v
[1] _format_prompt(ref_codes, ref_text, phonemes)
    Dinh dang:
    "user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_phones} {input_phones}
    <|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
    |
    v
[2] LMDeploy pipeline hoac AutoModelForCausalLM.generate()
    - bfloat16, CUDA
    - max_new_tokens=2048
    - repetition_penalty=1.2
    - Dung tai "<|SPEECH_GENERATION_END|>"
    - Output: chuoi token "<|speech_42|><|speech_15|>..."
    |
    v
[3] extract_speech_ids(output_str)
    Regex: r"<\|speech_(\d+)\|>" -> list int
    |
    v
[4] NeuCodec / DistillNeuCodec decode
    codes: torch.Tensor [1, 1, T]
    -> waveform: np.ndarray (float32, 24kHz)
    |
    v
[5] perth watermarking (optional)
```

#### Voice Cloning Flow:

```
ref_audio (WAV file path)
    |
    v
[1] _load_ref_mono(ref_audio, target_sr=16000)
    librosa.load() -> fallback -> soundfile + soxr.resample()
    -> wav: np.ndarray (float32, mono, 16kHz)
    |
    v
[2] encode_reference() -> MOSS Tokenizer Nano
    wav_tensor: [1, 1, T] -> codec.encode_code()
    -> ref_codes: np.ndarray (T_frames, 16)  [16 VQ codebooks]
    |
    v
[3] ref_codes duoc dua vao _format_prompt() nhu context
    -> Model "nghe" giong cua speaker qua reference codes
    |
    v
[4] Inference binh thuong voi ref_codes lam conditioning
    -> Giong noi dau ra mo phong speaker goc
```

#### Streaming: Overlap-Add Algorithm:

Thuật toán streaming dùng trong `fast.py`, `standard.py`, và `remote.py`:

```python
# Cac tham so streaming (tu BaseVieneuTTS):
streaming_frames_per_chunk = 50   # So frame decode moi lan (v3: 50, standard: 25)
streaming_lookforward = 5         # Frame nhin truoc
streaming_lookback = 50           # Frame nhin lai (context)
streaming_overlap_frames = 1      # Frame overlap
hop_length = 480                  # So sample/frame

# Moi khi tich luy du streaming_frames_per_chunk + lookforward tokens:
# 1. Decode mot cua so [tokens_start : tokens_end] bao gom lookback + lookforward
# 2. Cat phan audio [sample_start : sample_end] tu ket qua decode
# 3. Them vao audio_cache
# 4. _linear_overlap_add(audio_cache, stride=stride_samples)
#    -> Trung binh co trong so phan overlap giua chunk cu va moi
# 5. Yield phan audio moi chua duoc emit
```

Tham số `silence_p=0.15` (giây) thêm khoảng lặng ngắn giữa các chunk khi join, giúp câu nói nghe tự nhiên hơn.

---

### 5.3 Voice Preset System

#### Cấu trúc JSON của `voices_v3_turbo.json`:

```json
{
  "meta": {
    "spec": "vieneu.voice.presets",
    "spec_version": "1.1",
    "engine": "VieNeu-TTS-v3-Turbo",
    "checkpoint": "pnnbao-ump/VieNeu-TTS-v3-Turbo-Fixed-vi-emotion",
    "codec": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
    "n_vq": 16,
    "sample_rate": 48000,
    "frame_rate_hz": 12.5
  },
  "default_voice": "Ngoc Linh",
  "presets": {
    "Ngoc Lan": {
      "reserved_id": 13,
      "description": "nu, giong diu dang",
      "gender": "nu",
      "n_frames": 35,
      "codes": [
        [223, 796, 514, 58, 777, 556, 721, 957, 711, 227, 754, 838, 519, 811, 54, 708],
        ...
      ]
    }
  }
}
```

**Các trường quan trọng:**
- `reserved_id`: Token ID đặc biệt trong vocabulary của model (ids 13–42) để kích hoạt "emotion path" — giọng được train cùng model, chất lượng cao nhất.
- `codes`: Ma trận `(T, n_vq)` với `n_vq=16` codebooks của MOSS Tokenizer Nano. Đây là reference frames của speaker.
- `sample_rate`: 48000 Hz (v3 Turbo đặc trưng).
- `frame_rate_hz`: 12.5 Hz (80ms/frame).

**Danh sách giọng preset trong v3 Turbo (đọc từ `voices_v3_turbo.json`):**

| Tên giọng | reserved_id | Giới tính | Mô tả |
|---|---|---|---|
| Ngọc Lan | 13 | nữ | giọng dịu dàng |
| Gia Bảo | 16 | nam | giọng mượt mà |
| Thái Sơn | 17 | nam | giọng chắc khỏe |
| Đức Trí | 21 | nam | giọng rõ ràng |
| Mỹ Duyên | 22 | nữ | giọng mượt mà |
| Trúc Ly | 30 | nữ | giọng trẻ trung |
| Xuân Vĩnh | 32 | nam | giọng vui tươi |
| Trọng Hữu | 36 | nam | giọng uyên bác |
| Bình An | 37 | nam | giọng điềm đạm |
| Ngọc Linh *(mặc định)* | 41 | nữ | giọng tươi sáng |

**Cách thêm giọng mới:**

1. Thu âm reference audio (WAV, 16kHz+, 10–30 giây, mono).
2. Encode bằng MOSS tokenizer:
```python
tts = Vieneu(mode="v3turbo", backend="pytorch")
ref_codes = tts.encode_reference("my_speaker.wav")
# ref_codes.shape = (T, 16)
```
3. Thêm vào `voices_v3_turbo.json`:
```json
"Ten Giong Moi": {
  "reserved_id": null,
  "description": "nam/nu, mo ta",
  "gender": "nam",
  "n_frames": T,
  "codes": [[...16 values each row...], ...]
}
```
4. `reserved_id: null` → dùng emotion-tag path (không có speaker token riêng). Chất lượng vẫn tốt.

---

## 6. So sánh Các Phiên bản Model

| Model | Sample Rate | Hardware | Min VRAM | Latency | Chất lượng | Song ngữ | Voice Cloning | Streaming | Codec |
|---|---|---|---|---|---|---|---|---|---|
| v3 Turbo (ONNX CPU) | 48kHz | CPU | Không cần | Trung bình (5-10x GPU) | Cao | Có | Có (MOSS) | Không ổn định | MOSS Tokenizer Nano |
| v3 Turbo (PyTorch GPU) | 48kHz | CUDA GPU | ~4GB | Nhanh | Rất cao | Có | Có (MOSS) | Đang phát triển | MOSS Tokenizer Nano |
| v2 GPU (Fast/LMDeploy) | 24kHz | CUDA GPU | ~6GB | Rất nhanh | Cao | Có (v2) | Có (NeuCodec) | Có (linear OA) | DistillNeuCodec |
| v1 GPU (Fast/LMDeploy) | 24kHz | CUDA GPU | ~6GB | Rất nhanh | Trung bình | Không | Có (NeuCodec) | Có (linear OA) | DistillNeuCodec |
| v2 Standard (PyTorch) | 24kHz | CPU/GPU | ~4GB | Chậm | Cao | Có | Có (NeuCodec) | Có | ONNX int8 |
| v2 Standard (GGUF Q4) | 24kHz | CPU | Không cần | Trung bình | Trung bình | Có | Có | Có | ONNX int8 |
| v2 Turbo (GGUF) | 24kHz | CPU | Không cần | Trung bình | Trung bình | Không | Speaker emb | Có | VieNeu-Codec ONNX |
| v2 Turbo GPU | 24kHz | CUDA GPU | ~4GB | Nhanh | Trung bình | Không | Speaker emb | Có | VieNeu-Codec ONNX |
| Remote (client) | 24kHz | CPU (client) | Không cần | Phụ thuộc network | Server quyết định | Có | Có | Có (SSE) | DistillNeuCodec (local) |

---

## 7. Remote Server Architecture

### 7.1 LMDeploy TurbomindEngine

LMDeploy TurbomindEngine là inference engine chuyên biệt cho LLM, được tối ưu hóa cho:
- **Continuous batching**: Gộp nhiều request thành một batch tự động.
- **Paged KV cache**: Quản lý bộ nhớ KV cache hiệu quả.
- **bfloat16 precision**: Giảm VRAM, tốc độ cao trên Ampere+ GPU.
- **Tensor Parallel**: Chia model trên nhiều GPU (`--tp 2`).

### 7.2 OpenAI-Compatible Protocol

LMDeploy serve API server expose endpoint `/v1/chat/completions` tương thích với OpenAI API format. `RemoteVieNeuTTS` gửi request theo format này:

```json
POST /v1/chat/completions
{
  "model": "pnnbao-ump/VieNeu-TTS",
  "messages": [
    {
      "role": "user",
      "content": "user: Convert the text to speech:<|TEXT_PROMPT_START|><|emotion_0|>xin chao the gioi <|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|><|speech_223|><|speech_796|>..."
    }
  ],
  "max_tokens": 2048,
  "temperature": 1.0,
  "top_k": 50,
  "repetition_penalty": 1.2,
  "stop": ["<|SPEECH_GENERATION_END|>"],
  "stream": false
}
```

Response từ server:
```json
{
  "choices": [
    {
      "message": {
        "content": "<|speech_42|><|speech_15|><|speech_99|>..."
      }
    }
  ]
}
```

### 7.3 Client-Server Codec Split Architecture

Kiến trúc tách biệt giữa backbone (server) và codec (client):

```
+---------------- SERVER (GPU) ----------------+
|  LMDeploy TurbomindEngine                    |
|  Nhan phoneme prompt -> sinh speech tokens   |
|  Output: "<|speech_42|><|speech_15|>..."     |
+----------------------------------------------+
                     | HTTP (speech token strings)
                     v
+---------------- CLIENT (CPU) ----------------+
|  RemoteVieNeuTTS                             |
|  Nhan speech token strings                   |
|  extract_speech_ids() -> list[int]           |
|  NeuCodecOnnxDecoder.decode_code(codes)      |
|  -> audio waveform (float32, 24kHz)          |
+----------------------------------------------+
```

**Lợi ích của kiến trúc này**: Client không cần GPU mạnh, codec ONNX chạy nhanh trên CPU (< 100ms cho một chunk), băng thông tối thiểu (chỉ text token strings qua HTTP, không phải binary audio).

### 7.4 Bore Tunnel cho Public URL

```bash
# Cai bore (Rust tool)
cargo install bore-cli
# hoac download binary tu GitHub releases

# Khoi dong server voi tunnel
vieneu-serve --model pnnbao-ump/VieNeu-TTS-v2 --tunnel
# Output: Public URL: http://bore.pub:12345

# Client dung public URL
tts = Vieneu(mode="remote", api_base="http://bore.pub:12345/v1")
```

---

## 8. Vấn đề Đã Biết & Hạn chế

### 8.1 Platform Limitations

#### Triton không hỗ trợ Windows
`triton` (Triton-lang compiler) không có native Windows build. Khi `enable_triton=True` trên Windows, hàm `_compile_codec_with_triton()` sẽ fail và log warning, fallback về codec PyTorch không compile. `triton-windows` package tồn tại trong `pyproject.toml` nhưng chức năng bị giới hạn.

**Ảnh hưởng**: Codec decode chậm hơn ~20–30% so với Linux có Triton.

**Workaround**: Chạy trong WSL2 hoặc Docker Linux trên Windows để có Triton đầy đủ.

#### Long Path Issues trên Windows
Một số file trong HuggingFace cache có đường dẫn rất dài (>260 ký tự). Windows mặc định giới hạn MAX_PATH = 260.

**Workaround**: Bật long path trong registry:
```
HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1
```
Hoặc đặt HF_HOME tại thư mục ngắn:
```bash
set HF_HOME=C:\hf
```

#### Intel XPU: Beta Status
`XPUVieNeuTTS` phụ thuộc vào `torch.xpu` — tính năng còn trong giai đoạn thử nghiệm của PyTorch. Nếu driver Intel không đúng phiên bản, khởi tạo sẽ throw `RuntimeError`.

---

### 8.2 Performance Issues

#### Streaming First-Chunk Latency
Do mô hình cần sinh `streaming_frames_per_chunk + streaming_lookforward` tokens trước khi emit chunk đầu tiên, thời gian chờ chunk đầu vẫn đáng kể:
- Fast mode (LMDeploy): ~2–4 giây
- Standard mode (GGUF): ~3–8 giây
- v3 Turbo ONNX: ~4–10 giây

**Workaround**: Tăng `streaming_frames_per_chunk` để emit ít lần hơn nhưng chunk lớn hơn, giảm overhead decode. Hoặc dùng `infer()` thay vì `infer_stream()` khi latency quan trọng hơn throughput.

#### GPU OOM trên Batch Lớn
Khi `infer_batch()` với nhiều chunk dài, KV cache tăng mạnh. Nếu `memory_util=0.3` không đủ, LMDeploy sẽ throw OOM.

**Workaround**: Giảm `max_batch_size`, tăng `memory_util` hoặc dùng `quant_policy=8` (KV int8).

#### ONNX CPU 5–10x chậm hơn GPU
ONNX Runtime CPU không tận dụng được CUDA kernel tối ưu. Inference 5 giây audio:
- GPU (LMDeploy): ~0.5–1 giây
- CPU (ONNX v3): ~3–8 giây tuỳ CPU

**Workaround**: Dùng `RemoteVieNeuTTS` kết nối đến GPU server, hoặc chấp nhận latency cao trên CPU.

---

### 8.3 Quality Limitations

#### Giới hạn 256 ký tự mỗi chunk
Model được train với context window nhất định. Chunk dài hơn 256 ký tự có thể tạo ra:
- Giọng đọc kém tự nhiên ở cuối chunk
- Lặp từ (repetition)
- Cắt đột ngột

**Thiết kế**: Hệ thống luôn split trước 256 ký tự, đây là giới hạn hard-coded được khuyến cáo.

#### Emotion Tags còn thử nghiệm
Chỉ có 3 emotion tags (`[cười]`, `[thở dài]`, `[hắng giọng]`). Emotion checkpoint chưa phải bản ổn định:
- Đôi khi emotion không rõ ràng
- Cảm xúc phức tạp (tức giận, buồn) chưa được hỗ trợ

#### Song ngữ chỉ trong v2+
Model v1 (`pnnbao-ump/VieNeu-TTS`) chỉ được train với tiếng Việt. Văn bản tiếng Anh trong v1 sẽ được đọc theo phiên âm tiếng Việt không chính xác.

---

## 9. Risk Matrix

| Risk | Mức độ | Xác suất | Giảm thiểu |
|---|---|---|---|
| Model weights licensing thay đổi | Cao | Thấp | Apache 2.0 license rõ ràng; model repo public; document rõ ràng |
| HuggingFace Hub không khả dụng | Cao | Trung bình | `local_files_only=True` fallback sang cache local; hỗ trợ local path |
| CUDA version drift (cu128 vs cu121) | Trung bình | Cao | uv lock file với URL wheel cụ thể; test CI đa phiên bản |
| Memory leak trong long-running service | Trung bình | Trung bình | `close()` method + context manager; `gc.collect()` + `cuda.empty_cache()` |
| Deepfake/misuse giọng người khác | Cao | Trung bình | Perth watermarking bắt buộc; tài liệu ethical use; không cung cấp server public mặc định |
| Dependency breaking changes (llama-cpp 0.3.16) | Trung bình | Cao | Pin phiên bản chính xác `llama-cpp-python==0.3.16`; prebuilt wheel riêng |
| Audio watermark bị circumvented | Thấp | Thấp | Perth dùng implicit watermark khó detect; production nên thêm visible watermark bổ sung |
| sea-g2p thay đổi phoneme format | Cao | Thấp | Pin `>=0.7.6`; unit test phonemize output; test CI tự động |
| LMDeploy API thay đổi | Trung bình | Trung bình | Wrapper mỏng; nếu API thay đổi chỉ cần sửa `fast.py` / `turbo.py` |
| Windows MAX_PATH issue | Thấp | Cao (trên Windows) | Document workaround; test CI trên Windows; khuyến cáo HF_HOME ngắn |

---

## 10. Hướng Nâng cấp cho Sản xuất Chuyên nghiệp

### 10.1 Nâng cấp cho Làm phim & Dubbing

#### SSML Support Roadmap
SSML (Speech Synthesis Markup Language) cho phép kiểm soát chi tiết âm thanh:
```xml
<speak>
  <prosody rate="slow" pitch="+2st">Day la doan quan trong.</prosody>
  <break time="500ms"/>
  <emphasis level="strong">Rat quan trong!</emphasis>
</speak>
```

Kế hoạch triển khai:
1. Thêm parser SSML (lxml hoặc xml.etree)
2. Map `<prosody rate>` → điều chỉnh `temperature` và `max_new_frames`
3. Map `<break>` → thêm silence chunk vào join_audio_chunks
4. Map `<emphasis>` → emotion token tương ứng

#### SRT/VTT → Batch TTS Pipeline

```python
# Thiet ke pipeline subtitle-to-audio
def subtitle_to_audio(srt_path: str, voice: str, output_dir: str):
    segments = parse_srt(srt_path)
    # segments = [(start_ms, end_ms, text), ...]

    tts = Vieneu(mode="v3turbo")

    for i, (start_ms, end_ms, text) in enumerate(segments):
        duration_ms = end_ms - start_ms
        wav = tts.infer(text, voice=voice)

        # Time-stretch de khop voi duration
        target_samples = int(duration_ms / 1000 * tts.sample_rate)
        wav = time_stretch(wav, target_samples)

        sf.write(f"{output_dir}/seg_{i:04d}.wav", wav, tts.sample_rate)

    # Mix vao timeline FFmpeg
    ffmpeg_concat(output_dir, srt_path, "output.wav")
```

#### Multi-speaker JSON Script Format

```json
{
  "metadata": {
    "project": "Film Dub Project Alpha",
    "sample_rate": 48000
  },
  "characters": {
    "narrator": {"voice": "Binh An", "emotion": "natural"},
    "alice": {"voice": "Ngoc Linh", "emotion": "natural"},
    "bob": {"ref_audio": "bob_sample.wav"}
  },
  "script": [
    {"time": "00:00:01.500", "character": "narrator", "text": "Ngay xua..."},
    {"time": "00:00:05.200", "character": "alice", "text": "[cuoi] That sao?"},
    {"time": "00:00:07.800", "character": "bob", "text": "Toi khong biet."}
  ]
}
```

#### FFmpeg Timeline Sync Approach

```bash
# Mix nhieu track audio theo timeline
ffmpeg \
  -i video.mp4 \
  -i narrator_track.wav \
  -i alice_track.wav \
  -i bob_track.wav \
  -filter_complex \
    "[1:a]adelay=1500|1500[na]; \
     [2:a]adelay=5200|5200[aa]; \
     [3:a]adelay=7800|7800[ba]; \
     [na][aa][ba]amix=inputs=3[out]" \
  -map 0:v -map "[out]" \
  output_dubbed.mp4
```

#### Export Formats

| Format | Sample Rate | Bitrate | Dùng cho |
|---|---|---|---|
| WAV PCM 24-bit | 48kHz | ~2.3 Mbps | Master archive, post-processing |
| FLAC | 48kHz | ~800kbps | Lossless archive, cộng tác |
| MP3 320k | 44.1kHz | 320kbps | Phân phối, podcast |
| AAC-LC | 48kHz | 192kbps | Streaming, mobile |

LUFS normalization theo tiêu chuẩn phát thanh:
```python
import pyloudnorm as pyln

meter = pyln.Meter(tts.sample_rate)  # ITU-R BS.1770
loudness = meter.integrated_loudness(audio)
audio_normalized = pyln.normalize.loudness(audio, loudness, -16.0)  # -16 LUFS cho podcast
```

#### ADR (Automated Dialogue Replacement) Workflow

```
[Original Video] -> Extract dialogue transcript (Whisper ASR)
    |
    v
[Script Editor] -> Sua/dieu chinh transcript
    |
    v
[VieNeu-TTS] -> Sinh audio moi voi giong clone tu actor goc
    |
    v
[FFmpeg] -> Time-stretch audio de khop lip movement
    |
    v
[Final Video] -> Replace audio track
```

---

### 10.2 API Service

#### Tại sao core không có HTTP server (Design Decision: SDK-first)

VieNeu-TTS được thiết kế theo triết lý **SDK-first**: library Python sạch không phụ thuộc vào bất kỳ HTTP framework nào. Điều này cho phép:
- Tích hợp vào Django, FastAPI, Flask, Celery, gRPC mà không conflict.
- Test đơn giản hơn (không cần mock HTTP).
- Deploy như library (embedded) hoặc microservice tùy ý.

LMDeploy serve (`serve.py`) là exception duy nhất vì nó cần expose API để `RemoteVieNeuTTS` kết nối.

#### FastAPI Wrapper Design

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import soundfile as sf
from vieneu import Vieneu

app = FastAPI()
tts = Vieneu(mode="v3turbo")

class TTSRequest(BaseModel):
    text: str
    voice: str = "Ngoc Linh"
    emotion: str = "natural"
    temperature: float = 0.8
    format: str = "wav"  # wav, mp3, flac

# OpenAI-compatible endpoint
@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    wav = tts.infer(request.text, voice=request.voice)

    buffer = io.BytesIO()
    sf.write(buffer, wav, tts.sample_rate, format=request.format.upper())
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type=f"audio/{request.format}",
        headers={"Content-Disposition": f"attachment; filename=speech.{request.format}"}
    )

# SSE Streaming endpoint
@app.post("/v1/tts/stream")
async def tts_stream(request: TTSRequest):
    async def generate():
        for chunk in tts.infer_stream(request.text, voice=request.voice):
            yield chunk.tobytes()
    return StreamingResponse(generate(), media_type="audio/pcm")
```

#### Auth: API Key + JWT

```python
# API Key header auth (simple)
from fastapi import Header
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(401, "Invalid API key")
    return x_api_key

# JWT cho multi-tenant
import jwt
async def verify_jwt(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload["tenant_id"]
```

#### Rate Limiting: Token Bucket Algorithm

```python
import asyncio

class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate = rate      # tokens/second
        self.burst = burst    # max bucket size
        self.tokens = {}
        self.last_check = {}

    async def consume(self, key: str, n: int = 1) -> bool:
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_check.get(key, now)
        self.tokens[key] = min(
            self.burst,
            self.tokens.get(key, self.burst) + elapsed * self.rate
        )
        self.last_check[key] = now
        if self.tokens[key] >= n:
            self.tokens[key] -= n
            return True
        return False

# 10 requests/phut per API key
limiter = TokenBucket(rate=10/60, burst=10)
```

#### Queue: Celery + Redis cho Async Jobs

```python
from celery import Celery

celery_app = Celery("vieneu_tasks", broker="redis://localhost:6379/0")

@celery_app.task(bind=True)
def synthesize_async(self, text: str, voice: str, request_id: str):
    tts = get_tts_singleton()
    wav = tts.infer(text, voice=voice)

    # Save to object storage
    storage.save(f"results/{request_id}.wav", wav)
    return {"request_id": request_id, "status": "completed"}

# Client poll status
@app.get("/v1/tts/status/{request_id}")
async def get_status(request_id: str):
    result = celery_app.AsyncResult(request_id)
    if result.ready():
        return {"status": "completed", "download_url": f"/v1/tts/download/{request_id}"}
    return {"status": "processing"}
```

#### WebSocket cho Real-time

```python
from fastapi import WebSocket

@app.websocket("/v1/tts/ws")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        text = data["text"]
        voice = data.get("voice", "Ngoc Linh")

        for chunk in tts.infer_stream(text, voice=voice):
            await websocket.send_bytes(chunk.tobytes())

        await websocket.send_json({"event": "done"})
```

---

### 10.3 Scalability

#### Horizontal Scaling với Nginx + LMDeploy

```nginx
# nginx.conf
upstream vieneu_backends {
    least_conn;
    server gpu-server-1:23333 weight=3;
    server gpu-server-2:23333 weight=3;
    server gpu-server-3:23333 weight=1;
}

server {
    listen 443 ssl;
    location /v1/ {
        proxy_pass http://vieneu_backends;
        proxy_read_timeout 120s;
        proxy_set_header X-Request-ID $request_id;
    }
}
```

#### GPU Sharing Strategies

1. **MIG (Multi-Instance GPU)**: NVIDIA A100 có thể chia thành 7 instance độc lập, mỗi instance chạy một LMDeploy server.
2. **Tensor Parallel**: `--tp 2` chia model trên 2 GPU — phù hợp khi model không vừa 1 GPU.
3. **Time-sharing**: Một GPU chạy nhiều request sequential với continuous batching (LMDeploy mặc định).

#### Kubernetes Deployment Overview

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vieneu-tts-gpu
spec:
  replicas: 3
  selector:
    matchLabels: {app: vieneu-tts}
  template:
    spec:
      containers:
      - name: vieneu-server
        image: vieneu-tts:v3.0.5-cuda
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "16Gi"
          requests:
            memory: "8Gi"
        command:
        - vieneu-serve
        - --model
        - pnnbao-ump/VieNeu-TTS-v2
        - --port
        - "23333"
      nodeSelector:
        accelerator: nvidia-tesla-a100
```

#### Auto-scaling Trigger: Queue Depth

```yaml
# k8s/hpa.yaml -- scale based on Redis queue length
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vieneu-hpa
spec:
  scaleTargetRef:
    name: vieneu-tts-gpu
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: External
    external:
      metric:
        name: redis_queue_length
        selector:
          matchLabels: {queue: "vieneu_tasks"}
      target:
        type: AverageValue
        averageValue: "10"  # Scale up neu queue > 10 items/replica
```

---

### 10.4 Observability

#### Structured Logging

```python
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", None),
            "text_length": getattr(record, "text_length", None),
            "latency_ms": getattr(record, "latency_ms", None),
            "voice": getattr(record, "voice", None),
            "engine": getattr(record, "engine", None),
            "error": getattr(record, "error", None),
            "message": record.getMessage(),
        }
        return json.dumps(log_data, ensure_ascii=False)

# Su dung trong infer():
logger.info("Inference completed",
    extra={"request_id": req_id, "text_length": len(text),
           "latency_ms": int((time.time()-t0)*1000), "voice": voice})
```

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

inference_duration = Histogram(
    "vieneu_inference_duration_seconds",
    "Thoi gian inference",
    ["engine", "voice", "success"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

queue_length = Gauge("vieneu_queue_length", "So request dang cho")

error_counter = Counter(
    "vieneu_errors_total",
    "Tong so loi",
    ["engine", "error_type"]
)

# Su dung trong infer():
with inference_duration.labels(engine="v3turbo", voice=voice, success="true").time():
    wav = tts.infer(text, voice=voice)
```

#### OpenTelemetry Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer("vieneu.tts")

def infer_with_tracing(text: str, voice: str):
    with tracer.start_as_current_span("vieneu.infer") as span:
        span.set_attribute("text.length", len(text))
        span.set_attribute("voice", voice)
        span.set_attribute("engine", "v3turbo")

        with tracer.start_as_current_span("phonemize"):
            phonemes = phonemize_text(text)

        with tracer.start_as_current_span("model_inference"):
            wav = tts.infer(text, voice=voice)

        span.set_attribute("audio.duration_s", len(wav) / tts.sample_rate)
        return wav
```

#### Grafana Dashboard Key Panels

1. **P99 Inference Latency** — Histogram percentiles 50/95/99 theo engine
2. **Requests/Minute by Engine** — Counter rate, breakdown theo v3turbo/fast/standard
3. **GPU Memory Usage** — Gauge từ nvidia-smi exporter (VRAM used/total)
4. **Queue Depth** — Gauge từ Redis queue length
5. **Error Rate by Type** — Counter với label error_type (OOM, timeout, no_tokens)
6. **Audio Duration Generated** — Custom metric: tổng giây audio được tổng hợp/phút

---

### 10.5 Fine-tuning Pipeline

#### LoRA Fine-tuning Entry Point

VieNeu-TTS hỗ trợ load LoRA adapter qua PEFT library (chỉ PyTorch backbone):

```python
from vieneu import Vieneu

# Load base model (PyTorch, khong dung GGUF)
tts = Vieneu(
    mode="standard",
    backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
    backbone_device="cuda",
    gguf_filename=None
)

# Load LoRA adapter (thay doi voices.json tu dong)
tts.load_lora_adapter("my-org/my-speaker-lora")

# Unload va ve base model
tts.unload_lora_adapter()
```

Script fine-tuning tham khảo: `finetune/train.py` trong repo (nếu có).

#### Dataset Requirements

Để fine-tune speaker adaptation:

| Yêu cầu | Giá trị tối thiểu | Khuyến cáo |
|---|---|---|
| Số giờ audio | 1–2 giờ | 5–10 giờ |
| Sample rate | 22kHz+ | 48kHz (v3) hoặc 24kHz (v2) |
| Chất lượng | SNR > 30dB | Studio quality |
| Transcript | Bắt buộc | Verified bằng Whisper ASR |
| Alignment | Không bắt buộc | Forced alignment (Montreal FA) |
| Format | WAV PCM | WAV PCM 16-bit hoặc 24-bit |

#### Speaker Adaptation Workflow

```
[Raw Audio] -> Denoise (RNNoise/Demucs) -> Segment (VAD) -> Verify transcript (Whisper)
    |
    v
[Aligned Dataset] -> Extract MOSS codes -> Chon representative frames
    |
    v
[LoRA Training]
  - Base: VieNeu-TTS-v2 (PyTorch)
  - Target modules: q_proj, v_proj, k_proj (attention layers)
  - r=16, lora_alpha=32, epochs=3
    |
    v
[Validation] -> Infer test sentences -> MOS evaluation
    |
    v
[Export] -> Merge LoRA -> base -> Tao voice preset JSON
```

#### Export: Merge LoRA → Voice Preset JSON

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import json

# Merge LoRA vao base weights
base_model = AutoModelForCausalLM.from_pretrained("pnnbao-ump/VieNeu-TTS-v2")
peft_model = PeftModel.from_pretrained(base_model, "my-lora-adapter")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("my-merged-model")

# Tao voice preset tu reference audio
tts = Vieneu(mode="standard", backbone_repo="my-merged-model")
ref_codes = tts.encode_reference("speaker_sample.wav")

# Tao voices.json
voices = {
    "presets": {
        "My Speaker": {
            "codes": ref_codes.tolist(),
            "text": "Day la cau transcript cua doan reference audio.",
            "description": "Giong custom speaker"
        }
    },
    "default_voice": "My Speaker"
}
with open("my-merged-model/voices.json", "w") as f:
    json.dump(voices, f, ensure_ascii=False, indent=2)
```

---

## 11. Security Considerations

### 11.1 Perth Watermarking

`perth` library nhúng watermark vô hình (implicit watermark) vào dạng sóng âm thanh. Watermark tồn tại trong miền frequency, không nghe được bằng tai người nhưng có thể detect bằng tool tương ứng.

**Cách hoạt động:**
```python
# Trong BaseVieneuTTS._init_watermarker():
import perth
watermarker = perth.PerthImplicitWatermarker()

# Ap dung sau moi inference:
def _apply_watermark(self, wav: np.ndarray) -> np.ndarray:
    if self.watermarker:
        return self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
    return wav
```

**Enable/Disable:**
- Mặc định: Bật nếu `perth>=0.2.0` được cài.
- Disable per-call: `tts.infer(text, apply_watermark=False)`
- Disable hoàn toàn: `pip uninstall perth` (không khuyến cáo cho production)
- Production: Không nên disable — watermark là layer bảo vệ quan trọng.

**Giới hạn**: Perth watermark không chịu được heavy audio processing (re-encode MP3 <128kbps, pitch shift lớn, speed change >2x). Đối với production nghiêm ngặt, nên bổ sung thêm visible watermark hoặc metadata.

### 11.2 Input Sanitization

VieNeu-TTS hiện không có lớp sanitize text input tích hợp. Rủi ro tiềm ẩn khi expose API public:

- **Prompt injection**: Text chứa `<|speech_999999|>` có thể inject speech tokens trực tiếp vào context.
- **Unicode homoglyphs**: Ký tự giả mạo Unicode có thể vượt qua filter nhưng phonemize ra âm thanh khác.

**Khuyến cáo cho production:**
```python
import re

def sanitize_input(text: str, max_length: int = 5000) -> str:
    # Loai bo special tokens
    text = re.sub(r"<\|[^|]+\|>", "", text)
    # Gioi han do dai
    text = text[:max_length]
    # Loai bo control characters (giu newline)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()
```

### 11.3 API Key Management

- Không lưu API key trong code hoặc config file; dùng biến môi trường hoặc secret manager (HashiCorp Vault, AWS Secrets Manager).
- Rotate API key định kỳ (30–90 ngày).
- HuggingFace token (`hf_token`) cần đặt trong biến môi trường `HF_TOKEN`, không trong source code hay file config.
- Không log API key, kể cả dạng truncated.

### 11.4 Deepfake Risk và Giảm thiểu

Voice cloning có thể bị lạm dụng để clone giọng người khác mà không có sự đồng ý:

**Các biện pháp giảm thiểu:**
1. **Perth watermark**: Mọi audio output đều có watermark có thể trace về VieNeu-TTS.
2. **Terms of Service**: Cấm dùng để clone giọng người khác mà không được phép.
3. **Rate limiting**: Giới hạn số request/giờ để ngăn batch clone.
4. **SDK-first**: Không cung cấp public API mặc định — người dùng phải tự deploy và chịu trách nhiệm.
5. **Không hỗ trợ real-time phone audio**: Chỉ xử lý file static, không stream microphone input trực tiếp.

### 11.5 Model Access Control

- Models trên HuggingFace (`pnnbao-ump/`) là public — ai cũng có thể tải.
- Private models cần `hf_token` — không expose token qua API response hoặc log.
- ONNX weights không chứa training data, chỉ chứa model parameters (weights).
- Nếu cần kiểm soát access: dùng HuggingFace private repo với gated model (yêu cầu user agreement).

---

## 12. Roadmap

### Ngắn hạn (0–3 tháng)

| Hạng mục | Mô tả | Ưu tiên |
|---|---|---|
| FastAPI wrapper | Endpoint `/v1/audio/speech` OpenAI-compatible, SSE streaming | Cao |
| Windows long path fix | Tự động enable long path hoặc hướng dẫn rõ ràng | Trung bình |
| Streaming stability v3 | Fix first-chunk latency trong ONNX path | Cao |
| Emotion checkpoint stable | Release stable v3 với đầy đủ 3 emotion tags | Cao |
| CI/CD matrix | Test tự động trên Windows + Linux + macOS | Trung bình |
| Docker image chính thức | `vieneu-tts:cpu` và `vieneu-tts:gpu` trên Docker Hub | Trung bình |

### Trung hạn (3–6 tháng)

| Hạng mục | Mô tả | Ưu tiên |
|---|---|---|
| SSML basic support | `<break>`, `<prosody rate>`, `<emphasis>` | Cao |
| Subtitle pipeline | SRT/VTT → batch TTS với time-sync và FFmpeg export | Trung bình |
| Prometheus metrics | Metrics endpoint tích hợp trong FastAPI wrapper | Trung bình |
| Speaker fine-tuning docs | End-to-end guide fine-tune với LoRA | Cao |
| Audio quality v3 GPU | Tăng MOS score cho v3 Turbo GPU path | Cao |
| More emotion tags | Thêm `[sad]`, `[excited]`, `[angry]` | Trung bình |
| OpenTelemetry | Distributed tracing cho multi-service deployment | Thấp |

### Dài hạn (6–12 tháng)

| Hạng mục | Mô tả | Ưu tiên |
|---|---|---|
| Kubernetes Helm chart | Deploy VieNeu-TTS trên K8s với một lệnh | Trung bình |
| Speaker adaptation UI | Tích hợp trong Gradio: upload audio → tạo preset JSON | Cao |
| Multi-language | Hỗ trợ thêm Thái, Indonesia (SEA languages) | Thấp |
| Real-time streaming API | WebSocket API first-chunk < 200ms | Cao |
| v4 architecture | Kiến trúc mới non-autoregressive hoặc flow-matching | Cao |
| LUFS normalization | Tích hợp loudness normalization (-16 LUFS) vào SDK | Trung bình |
| ADR workflow CLI | Command-line tool hoàn chỉnh cho dubbing studio | Thấp |

---

*Tài liệu này được tạo dựa trên source code phiên bản 3.0.5. Các class, tham số, và flow được mô tả phản ánh trực tiếp code trong các file: `factory.py`, `base.py`, `v3turbo.py`, `fast.py`, `standard.py`, `turbo.py`, `remote.py`, `serve.py`, `vieneu_utils/phonemize_text.py`, `vieneu_utils/core_utils.py`, `pyproject.toml`, `config.yaml`, và `assets/voices_v3_turbo.json`. Mọi thay đổi kiến trúc cần được cập nhật tương ứng vào tài liệu này.*
