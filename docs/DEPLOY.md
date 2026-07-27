# VieNeu-TTS — Hướng dẫn Deploy Chi tiết

## Tổng quan deploy

VieNeu-TTS hỗ trợ ba hình thức triển khai chính, mỗi hình thức phù hợp với một nhóm người dùng và mục đích khác nhau.

### Bảng so sánh 3 môi trường

| Tiêu chí | Windows (native) | Linux (native) | Docker |
|---|---|---|---|
| **Phù hợp với ai** | Dev cá nhân, thử nghiệm nhanh | Server production, cloud VM | DevOps, CI/CD, scale |
| **Độ khó cài đặt** | Trung bình (driver CUDA phức tạp) | Thấp–Trung bình | Thấp (sau khi có Docker) |
| **GPU support** | Có (CUDA 12.x, cần cấu hình thêm) | Có (đầy đủ nhất) | Có (cần nvidia-container-toolkit) |
| **Isolated** | Không | Không | Có (container riêng biệt) |
| **Production-ready** | Kém (cần NSSM/Task Scheduler) | Tốt (systemd) | Tốt nhất (restart policy) |
| **Dễ scale** | Khó | Trung bình | Tốt (compose profiles) |
| **Dễ troubleshoot** | Khó (driver, path issues) | Tốt (journald, strace) | Trung bình (log qua docker logs) |
| **Triton support** | Hạn chế (triton-windows) | Đầy đủ | Đầy đủ (trong Linux container) |
| **CPU-only (ONNX)** | Có | Có | Có |
| **Live code reload** | Có | Có | Dev mode: có (volume mount) |

### Yêu cầu phần cứng

**Tối thiểu (CPU-only / ONNX mode — v3 Turbo):**
- CPU: x86-64, 4 nhân trở lên
- RAM: 8 GB
- Storage: 10 GB trống (model cache HuggingFace)
- OS: Windows 10 64-bit / Ubuntu 20.04 / Debian 11

**Khuyến nghị (GPU mode — v2/v3 GPU):**
- CPU: x86-64, 8 nhân trở lên
- RAM: 16 GB+
- GPU: NVIDIA RTX 2060 trở lên, VRAM 6 GB+ (8 GB+ cho v2 GPU)
- Storage: 30 GB+ (model + codec + HuggingFace cache)
- OS: Ubuntu 22.04 LTS / Debian 12

**Lưu ý về model size:**
- v3 Turbo ONNX: ~1.5 GB download
- v3 Turbo PyTorch: ~3 GB download
- VieNeu-TTS-v2 (GPU): ~7 GB download
- VieNeu-TTS-v1 (GPU): ~7 GB download

### Biến môi trường quan trọng

| Biến | Mô tả | Giá trị mặc định | Ví dụ |
|---|---|---|---|
| `HF_HOME` | Thư mục cache HuggingFace | `~/.cache/huggingface` | `/data/hf_cache` |
| `CUDA_VISIBLE_DEVICES` | Chọn GPU sử dụng | Tất cả GPU | `0` hoặc `0,1` |
| `GRADIO_SERVER_NAME` | Địa chỉ bind Gradio | `127.0.0.1` | `0.0.0.0` (public) |
| `GRADIO_SERVER_PORT` | Port Gradio | `8001` | `8080` |
| `GRADIO_SHARE` | Bật Gradio public share | `0` | `1` |
| `PYTHONUNBUFFERED` | Tắt Python output buffering | không set | `1` |
| `VIENEU_COMPILE` | Bật torch.compile (Linux only) | không set | `1` |
| `HF_TOKEN` | HuggingFace API token (model private) | không set | `hf_xxx...` |
| `PHONEMIZER_ESPEAK_LIBRARY` | Đường dẫn libespeak-ng | tự detect | `/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1` |

---

## PHẦN 1: Deploy trên Windows

### 1.1 Yêu cầu hệ thống

- Windows 10 64-bit build 1903+ / Windows 11 (khuyến nghị)
- RAM: tối thiểu 8 GB (CPU/ONNX mode), 16 GB+ (GPU mode)
- Storage: 10 GB+ trống trên ổ cài đặt (cho model cache)
- GPU (tùy chọn): NVIDIA RTX 2060+ với VRAM 6 GB+ và CUDA 12.x driver
- PowerShell 5.1+ (có sẵn trên Windows 10/11)
- Git for Windows

### 1.2 Cài đặt prerequisites

#### Python 3.12

Tải Python **từ python.org**, không dùng Microsoft Store:

```
https://www.python.org/downloads/release/python-3120/
```

**Tại sao không dùng Microsoft Store version?** Python từ Microsoft Store bị sandbox hóa (`WindowsApps` directory), giới hạn quyền truy cập file system, không tương thích với một số C extension (bao gồm `llama-cpp-python` và `onnxruntime` CUDA provider). Ngoài ra, `uv` cũng có thể không nhận ra Python Store path khi detect interpreter.

Khi cài, **bắt buộc tick vào**:
- "Add python.exe to PATH"
- "Install pip"

Verify sau khi cài:

```powershell
python --version
# Kết quả mong đợi: Python 3.12.x
```

#### CUDA Toolkit 12.x (chỉ cần nếu dùng GPU)

Tải CUDA Toolkit 12.8 từ NVIDIA:

```
https://developer.nvidia.com/cuda-12-8-0-download-archive
```

Chọn: Windows → x86_64 → 10/11 → exe (local)

Sau khi cài, verify:

```powershell
nvcc --version
nvidia-smi
```

`nvidia-smi` sẽ hiển thị GPU và driver version. Cột "CUDA Version" trên góc phải cần >= 12.8.

#### Visual C++ Redistributable

Nhiều C extension (bao gồm `onnxruntime`, `soundfile`) cần VC++ runtime:

```
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

Tải và chạy file trên.

#### Git for Windows

```
https://git-scm.com/download/win
```

Chọn "Git Bash" và "Git from command line" trong quá trình cài.

#### uv package manager

Mở PowerShell với quyền thường (không cần Admin) và chạy:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sau đó restart terminal và verify:

```powershell
uv --version
# Kết quả mong đợi: uv 0.x.x
```

### 1.3 Cài đặt VieNeu-TTS

Mở PowerShell hoặc Git Bash, chạy:

```powershell
# Clone repository
git clone https://github.com/pnnbao97/VieNeu-TTS
cd VieNeu-TTS
```

**Cài đặt CPU-only (torch-free, ONNX engine — không cần GPU):**

```powershell
uv sync
```

Chế độ này cài đặt tập phụ thuộc tối thiểu: `onnxruntime`, `sea-g2p`, `soundfile`, `soxr`, `tokenizers`, `huggingface_hub`, `gradio`. Không cài PyTorch. Toàn bộ inference chạy qua ONNX Runtime trên CPU. Phù hợp cho máy không có GPU hoặc muốn thử nghiệm nhẹ.

**Cài đặt với GPU support (PyTorch + LMDeploy):**

```powershell
uv sync --group gpu
```

Chế độ này cài thêm: `torch` (CUDA 12.8 wheel), `torchaudio`, `transformers`, `librosa`, `neucodec`, `lmdeploy` (CUDA 12.8, Python 3.12 pinned), `triton-windows`. Dung lượng download lớn hơn (~5 GB). Bắt buộc phải có NVIDIA GPU với CUDA 12.x driver.

**Kiểm tra cài đặt:**

```powershell
# Kiểm tra import cơ bản
uv run python -c "from vieneu import Vieneu; print('OK')"

# Kiểm tra CUDA (GPU mode)
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 1.4 Cấu hình môi trường Windows

#### Thiết lập HF_HOME (tránh download lại khi restart)

```powershell
# Thiết lập vĩnh viễn cho user hiện tại
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf_cache", "User")

# Hoặc cho toàn hệ thống (cần quyền Admin)
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf_cache", "Machine")
```

Thay `D:\hf_cache` bằng đường dẫn ổ đĩa có đủ dung lượng (khuyến nghị ổ D hoặc E nếu ổ C chật).

#### Bật Long Path (bắt buộc cho Windows)

Nhiều file trong model cache có đường dẫn dài hơn giới hạn 260 ký tự của Windows. Chạy PowerShell với quyền **Administrator**:

```powershell
# Bật long path support qua registry
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -Type DWord

# Xác nhận đã bật
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"
```

Restart lại máy sau khi thay đổi registry.

#### Windows Defender exclusion cho model cache

Windows Defender có thể scan và làm chậm việc đọc model files khi inference. Thêm exclusion:

```powershell
# Chạy với quyền Admin
Add-MpPreference -ExclusionPath "D:\hf_cache"
Add-MpPreference -ExclusionPath "$env:USERPROFILE\VieNeu-TTS"
```

#### Thiết lập CUDA_VISIBLE_DEVICES (multi-GPU)

```powershell
# Chỉ dùng GPU đầu tiên
[System.Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", "User")
```

### 1.5 Khởi chạy ứng dụng

**Web UI (Gradio — giao diện đầy đủ):**

```powershell
uv run vieneu-web
```

Sau khi khởi động, mở trình duyệt tại: `http://localhost:8001`

**Streaming Web UI (CPU GGUF mode):**

```powershell
uv run vieneu-stream
```

**Chạy script Python ví dụ:**

```powershell
# Tạo thư mục output
mkdir outputs

# Chạy example cơ bản (v3 Turbo, auto-detect CPU/GPU)
uv run python examples/main_v3turbo.py

# Chạy example standard mode
uv run python examples/main.py

# Chạy example remote mode (cần server chạy sẵn)
uv run python examples/main_remote.py
```

**Chạy Remote API Server (LMDeploy, chỉ GPU):**

```powershell
uv run python src/vieneu/serve.py --model pnnbao-ump/VieNeu-TTS-v2 --port 23333
```

### 1.6 Chạy như Windows Service (production)

Để VieNeu-TTS tự động khởi động cùng Windows, dùng **NSSM** (Non-Sucking Service Manager).

**Tải NSSM:**

```
https://nssm.cc/download
```

Giải nén và đặt `nssm.exe` vào `C:\Windows\System32\` hoặc thêm vào PATH.

**Tạo wrapper script** `C:\VieNeu-TTS\start_vieneu.ps1`:

```powershell
# start_vieneu.ps1
$env:HF_HOME = "D:\hf_cache"
$env:GRADIO_SERVER_NAME = "0.0.0.0"
$env:GRADIO_SERVER_PORT = "8001"
$env:PYTHONUNBUFFERED = "1"

Set-Location "C:\VieNeu-TTS"

# Ghi log với timestamp
$logFile = "C:\VieNeu-TTS\logs\vieneu_$(Get-Date -Format 'yyyyMMdd').log"
New-Item -ItemType Directory -Force -Path "C:\VieNeu-TTS\logs" | Out-Null

& uv run vieneu-web 2>&1 | Tee-Object -FilePath $logFile
```

**Đăng ký service với NSSM** (chạy PowerShell với quyền Admin):

```powershell
# Cài đặt service
nssm install VieNeuTTS powershell.exe

# Cấu hình các tham số
nssm set VieNeuTTS AppParameters "-ExecutionPolicy Bypass -File C:\VieNeu-TTS\start_vieneu.ps1"
nssm set VieNeuTTS AppDirectory "C:\VieNeu-TTS"
nssm set VieNeuTTS DisplayName "VieNeu TTS Service"
nssm set VieNeuTTS Description "Vietnamese Text-to-Speech Server"
nssm set VieNeuTTS Start SERVICE_AUTO_START
nssm set VieNeuTTS AppStdout "C:\VieNeu-TTS\logs\service_stdout.log"
nssm set VieNeuTTS AppStderr "C:\VieNeu-TTS\logs\service_stderr.log"
nssm set VieNeuTTS AppRotateFiles 1
nssm set VieNeuTTS AppRotateSeconds 86400

# Bật auto-restart khi crash
nssm set VieNeuTTS AppRestartDelay 5000

# Khởi động service
nssm start VieNeuTTS

# Kiểm tra trạng thái
nssm status VieNeuTTS
```

**Quản lý service:**

```powershell
# Dừng service
nssm stop VieNeuTTS

# Restart service
nssm restart VieNeuTTS

# Gỡ cài đặt service
nssm remove VieNeuTTS confirm
```

**Thay thế bằng Task Scheduler** (không cần cài thêm công cụ):

```powershell
# Tạo scheduled task chạy khi khởi động Windows
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File C:\VieNeu-TTS\start_vieneu.ps1" `
    -WorkingDirectory "C:\VieNeu-TTS"

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([System.TimeSpan]::Zero)

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest

Register-ScheduledTask -TaskName "VieNeuTTS" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal
```

### 1.7 Troubleshooting Windows

#### "Triton is not supported on Windows" hoặc lỗi liên quan đến triton

**Nguyên nhân:** `triton-windows` là port không chính thức của Triton cho Windows. Một số tính năng như custom CUDA kernels của LMDeploy có thể không hoạt động hoàn toàn.

**Tính năng bị ảnh hưởng:** Tăng tốc inference qua custom ops, một số optimization trong LMDeploy.

**Workaround:**

```powershell
# Tắt Triton và dùng fallback PyTorch
$env:LMDEPLOY_USE_TRITON = "0"
uv run vieneu-web
```

Hoặc nếu lỗi khi import:

```powershell
# Chạy không có triton
uv run python -c "
import sys
sys.modules['triton'] = None
from vieneu import Vieneu
tts = Vieneu()
"
```

Nếu không cần GPU inference tốc độ cao, dùng ONNX mode (CPU) thay thế — không cần triton.

#### Long path errors khi install hoặc runtime

```powershell
# Kiểm tra trạng thái Long Path
(Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem").LongPathsEnabled

# Nếu trả về 0, bật lên
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
# Sau đó restart máy
```

#### CUDA not found dù đã cài

```powershell
# Kiểm tra nvidia-smi
nvidia-smi

# Kiểm tra CUDA trong PATH
where nvcc

# Nếu thiếu, thêm CUDA vào PATH thủ công
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
[System.Environment]::SetEnvironmentVariable("PATH", "$env:PATH;$cudaPath", "User")

# Kiểm tra torch nhận GPU
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

#### Port 8001 đang bị chiếm

```powershell
# Tìm process đang dùng port 8001
netstat -ano | findstr :8001

# Kill process theo PID (thay 12345 bằng PID thực tế)
taskkill /F /PID 12345

# Chạy trên port khác
$env:GRADIO_SERVER_PORT = "8080"
uv run vieneu-web
```

#### PowerShell execution policy block

```powershell
# Cho phép script chạy (chỉ user hiện tại, an toàn hơn)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Hoặc bypass cho một lần chạy
powershell -ExecutionPolicy Bypass -File start_vieneu.ps1
```

#### Out of Memory khi inference

```powershell
# Giảm memory utilization của LMDeploy
$env:VIENEU_MEMORY_UTIL = "0.2"

# Hoặc dùng CPU/ONNX mode
uv run python -c "
from vieneu import Vieneu
tts = Vieneu(mode='v3turbo', backend='onnx')  # Force ONNX
"
```

#### pip/uv install thất bại do proxy hoặc SSL

```powershell
# Cấu hình proxy cho uv
$env:HTTPS_PROXY = "http://proxy.company.com:8080"
$env:HTTP_PROXY = "http://proxy.company.com:8080"

# Hoặc bypass SSL verification (không khuyến nghị cho production)
$env:UV_NATIVE_TLS = "true"
uv sync --group gpu
```

---

## PHẦN 2: Deploy trên Linux (Ubuntu/Debian)

### 2.1 Yêu cầu hệ thống

- Ubuntu 22.04 LTS hoặc Ubuntu 24.04 LTS (khuyến nghị)
- Debian 12 "Bookworm" cũng được hỗ trợ tốt
- RAM: 8 GB+ (CPU/ONNX mode), 16 GB+ (GPU mode)
- Storage: 30 GB+ trống (model cache + dependencies)
- NVIDIA Driver 535+ (cho GPU mode): kiểm tra bằng `nvidia-smi`
- Quyền sudo

### 2.2 Cài đặt system dependencies

```bash
# Cập nhật package list
sudo apt update && sudo apt upgrade -y

# Cài đặt dependencies cơ bản
sudo apt install -y \
    software-properties-common \
    build-essential \
    git curl wget ca-certificates \
    libsndfile1 libsndfile1-dev \
    espeak-ng libespeak-ng1 libespeak-ng-dev \
    ffmpeg \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip

# Đặt Python 3.12 làm mặc định (nếu hệ thống đang dùng Python cũ hơn)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Xác nhận version
python3 --version
# Kết quả mong đợi: Python 3.12.x
```

**Cài uv package manager:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Thêm uv vào PATH trong phiên hiện tại
source ~/.bashrc
# Hoặc nếu dùng zsh
source ~/.zshrc

# Xác nhận
uv --version
```

### 2.3 CUDA Setup (GPU mode)

**Cài NVIDIA driver từ apt (Ubuntu 22.04):**

```bash
# Thêm NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Cài driver mới nhất (535+)
sudo apt install -y nvidia-driver-535

# Reboot để driver có hiệu lực
sudo reboot
```

Sau khi reboot, xác nhận:

```bash
nvidia-smi
# Phải hiển thị GPU info và Driver Version >= 535
```

**Cài CUDA Toolkit 12.8:**

```bash
# Thêm NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Cài CUDA 12.8
sudo apt install -y cuda-toolkit-12-8

# Thêm CUDA vào PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
# Kết quả mong đợi: release 12.8
```

### 2.4 Cài đặt VieNeu-TTS

```bash
# Clone repository
git clone https://github.com/pnnbao97/VieNeu-TTS
cd VieNeu-TTS

# Cài đặt GPU mode (khuyến nghị cho server có GPU)
uv sync --group gpu

# Hoặc cài đặt CPU-only (ONNX, không cần GPU)
uv sync

# Xác nhận cài đặt
uv run python -c "from vieneu import Vieneu; print('VieNeu-TTS installed OK')"

# Kiểm tra GPU detect (nếu cài GPU mode)
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 2.5 Cấu hình môi trường Linux

Thêm vào `~/.bashrc` (user) hoặc `/etc/environment` (toàn hệ thống):

```bash
# ~/.bashrc — thêm vào cuối file
export HF_HOME=/data/huggingface_cache
export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=8001
export GRADIO_SHARE=0
export PYTHONUNBUFFERED=1
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1

# Áp dụng ngay
source ~/.bashrc
```

Tạo thư mục HF cache nếu dùng ổ data riêng:

```bash
sudo mkdir -p /data/huggingface_cache
sudo chown $USER:$USER /data/huggingface_cache
```

**Thiết lập cho toàn hệ thống** (tất cả user, kể cả service account):

```bash
sudo tee /etc/environment << 'EOF'
HF_HOME=/data/huggingface_cache
PYTHONUNBUFFERED=1
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=8001
PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
EOF
```

### 2.6 Systemd Service (production)

Tạo file service tại `/etc/systemd/system/vieneu-tts.service`:

```bash
sudo tee /etc/systemd/system/vieneu-tts.service << 'EOF'
[Unit]
Description=VieNeu TTS Web Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/VieNeu-TTS

# Environment variables
Environment=HF_HOME=/data/huggingface_cache
Environment=CUDA_VISIBLE_DEVICES=0
Environment=GRADIO_SERVER_NAME=0.0.0.0
Environment=GRADIO_SERVER_PORT=8001
Environment=GRADIO_SHARE=0
Environment=PYTHONUNBUFFERED=1
Environment=PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1

# Command to run
ExecStart=/home/ubuntu/.local/bin/uv run vieneu-web

# Restart policy
Restart=always
RestartSec=10
StartLimitIntervalSec=60
StartLimitBurst=3

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vieneu-tts

# Resource limits
TimeoutStartSec=300
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
```

**Thay `ubuntu` bằng username thực tế của bạn.**

**Enable và start service:**

```bash
# Reload systemd để nhận config mới
sudo systemctl daemon-reload

# Enable auto-start khi boot
sudo systemctl enable vieneu-tts

# Start service ngay
sudo systemctl start vieneu-tts

# Kiểm tra trạng thái
sudo systemctl status vieneu-tts

# Xem logs real-time
sudo journalctl -u vieneu-tts -f

# Xem 100 dòng log gần nhất
sudo journalctl -u vieneu-tts -n 100
```

### 2.7 Nginx Reverse Proxy + SSL

**Cài Nginx:**

```bash
sudo apt install -y nginx
```

**Tạo config** `/etc/nginx/sites-available/vieneu-tts`:

```nginx
# /etc/nginx/sites-available/vieneu-tts
upstream vieneu_backend {
    server 127.0.0.1:8001;
    keepalive 32;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name your-domain.com;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your-domain.com;

    # SSL certificates (sẽ được Certbot điền vào)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;
    gzip_min_length 1024;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=vieneu:10m rate=10r/m;
    limit_req zone=vieneu burst=20 nodelay;

    # Upload size (cho audio reference upload)
    client_max_body_size 50M;

    # Proxy headers
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Timeout (synthesis có thể lâu)
    proxy_read_timeout 300;
    proxy_connect_timeout 60;
    proxy_send_timeout 300;

    # WebSocket support (Gradio streaming cần thiết)
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";

    location / {
        proxy_pass http://vieneu_backend;
    }

    # Static files (nếu có)
    location /static/ {
        proxy_pass http://vieneu_backend/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

**Enable site và cài SSL với Certbot:**

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/vieneu-tts /etc/nginx/sites-enabled/
sudo nginx -t  # Test config
sudo systemctl restart nginx

# Cài Certbot
sudo apt install -y certbot python3-certbot-nginx

# Lấy SSL certificate (thay your-domain.com)
sudo certbot --nginx -d your-domain.com --email your@email.com --agree-tos --non-interactive

# Auto-renew certificate
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

### 2.8 Remote API Server (LMDeploy)

**Chạy trực tiếp:**

```bash
# Chạy với model mặc định
uv run python src/vieneu/serve.py \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --memory-util 0.3

# Với tunnel công khai (bore.pub)
uv run python src/vieneu/serve.py \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --tunnel

# Tensor parallel 2 GPUs
uv run python src/vieneu/serve.py \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --tp 2 \
    --memory-util 0.5
```

**Systemd service cho remote API server:**

```bash
sudo tee /etc/systemd/system/vieneu-serve.service << 'EOF'
[Unit]
Description=VieNeu TTS Remote API Server (LMDeploy)
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/VieNeu-TTS
Environment=HF_HOME=/data/huggingface_cache
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/ubuntu/.local/bin/uv run python src/vieneu/serve.py \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --memory-util 0.3
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vieneu-serve
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vieneu-serve
sudo systemctl start vieneu-serve
```

**Mở firewall cho API port:**

```bash
# Mở port 23333 (chỉ nội bộ network)
sudo ufw allow from 10.0.0.0/8 to any port 23333

# Hoặc mở public (không khuyến nghị, dùng tunnel thay thế)
sudo ufw allow 23333/tcp

# Xem trạng thái firewall
sudo ufw status
```

### 2.9 Multi-GPU Setup

**Chạy nhiều instance với GPU riêng:**

```bash
# Instance 1: GPU 0, port 8001
CUDA_VISIBLE_DEVICES=0 GRADIO_SERVER_PORT=8001 uv run vieneu-web &

# Instance 2: GPU 1, port 7861
CUDA_VISIBLE_DEVICES=1 GRADIO_SERVER_PORT=7861 uv run vieneu-web &
```

**Dùng tmux để quản lý nhiều session:**

```bash
# Cài tmux
sudo apt install -y tmux

# Tạo session mới
tmux new-session -d -s vieneu-gpu0 -x 220 -y 50
tmux send-keys -t vieneu-gpu0 "cd ~/VieNeu-TTS && CUDA_VISIBLE_DEVICES=0 uv run vieneu-web" Enter

tmux new-session -d -s vieneu-gpu1 -x 220 -y 50
tmux send-keys -t vieneu-gpu1 "cd ~/VieNeu-TTS && CUDA_VISIBLE_DEVICES=1 GRADIO_SERVER_PORT=7861 uv run vieneu-web" Enter

# Xem session
tmux list-sessions
tmux attach -t vieneu-gpu0
```

**Nginx upstream load balancing:**

```nginx
upstream vieneu_cluster {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:7861;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    # ... SSL config ...

    location / {
        proxy_pass http://vieneu_cluster;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300;
    }
}
```

### 2.10 Troubleshooting Linux

#### CUDA version mismatch

```bash
# Kiểm tra driver version
nvidia-smi | grep "Driver Version"

# Kiểm tra CUDA version torch đang dùng
uv run python -c "import torch; print(torch.version.cuda)"

# Kiểm tra compatibility
# torch 2.x + CUDA 12.8 → cần driver >= 525
# torch 2.x + CUDA 12.1 → cần driver >= 525
# Nếu không khớp, cài lại driver
```

#### libsndfile not found

```bash
sudo apt install -y libsndfile1 libsndfile1-dev
# Nếu vẫn lỗi, check library path
ldconfig -p | grep sndfile
```

#### Port binding permission (port dưới 1024)

```bash
# Dùng authbind để bind port 80 không cần root
sudo apt install -y authbind
sudo touch /etc/authbind/byport/80
sudo chown ubuntu /etc/authbind/byport/80
sudo chmod 755 /etc/authbind/byport/80

# Hoặc dùng iptables redirect
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8001
```

#### GPU không detect sau cài driver

```bash
# Kiểm tra kernel module
lsmod | grep nvidia

# Nếu không thấy, load module
sudo modprobe nvidia

# Nếu vẫn không được, cài lại driver
sudo apt purge nvidia-*
sudo apt autoremove
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### HuggingFace download chậm

```bash
# Dùng HF mirror (hữu ích tại Việt Nam)
export HF_ENDPOINT=https://hf-mirror.com

# Hoặc download offline mode (sau lần đầu đã cache)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Download riêng bằng huggingface-cli
uv run huggingface-cli download pnnbao-ump/VieNeu-TTS-v3-Turbo --local-dir /data/models/v3turbo
```

#### OOM Killer giết process

```bash
# Tăng swap
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Tăng ulimit cho process
ulimit -v unlimited
ulimit -m unlimited

# Điều chỉnh OOM score cho service (ít bị kill hơn)
# Thêm vào [Service] trong systemd unit:
# OOMScoreAdjust=-500
```

---

## PHẦN 3: Deploy với Docker

### 3.1 Prerequisites

**Cài Docker (Ubuntu 22.04/24.04):**

```bash
# Gỡ Docker cũ nếu có
sudo apt remove docker docker-engine docker.io containerd runc 2>/dev/null

# Cài Docker từ official repository
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Thêm user vào docker group (không cần sudo mỗi lần)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
# Kết quả mong đợi: Docker version 24.x.x hoặc cao hơn
docker compose version
# Kết quả mong đợi: Docker Compose version v2.x.x
```

**Cài NVIDIA Container Toolkit (nvidia-docker2):**

```bash
# Thêm NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Cấu hình Docker daemon dùng NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access trong container
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
# Phải hiển thị GPU info
```

### 3.2 Quick Start — Development

```bash
# Clone repository
git clone https://github.com/pnnbao97/VieNeu-TTS
cd VieNeu-TTS

# Tạo file .env từ template (nếu có)
cp .env.example .env 2>/dev/null || touch .env

# Chỉnh sửa .env theo nhu cầu
cat > .env << 'EOF'
HF_HOME=/root/.cache/huggingface
PORT=8001
GRADIO_SERVER_PORT=8001
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SHARE=0
MEMORY_UTIL=0.3
MODEL=pnnbao-ump/VieNeu-TTS-v2
MODEL_NAME=pnnbao-ump/VieNeu-TTS-v2
SERVE_PORT=23333
EOF

# Chạy Gradio UI (dev mode — volume mount, hot reload)
docker compose -f docker/docker-compose.yml --profile gpu up

# Chạy API Server (serve mode)
docker compose -f docker/docker-compose.yml --profile serve up

# Chạy detach (background)
docker compose -f docker/docker-compose.yml --profile gpu up -d

# Xem logs
docker compose -f docker/docker-compose.yml logs -f
```

**Lưu ý về profiles:** File `docker/docker-compose.yml` định nghĩa hai profiles riêng biệt:
- `gpu`: Gradio Web UI trên port 8001
- `serve`: LMDeploy API server trên port 23333

Chỉ được kích hoạt khi truyền `--profile <tên>`.

### 3.3 Build Docker Image

```bash
# Build GPU image (production Gradio UI)
docker build \
    -f docker/Dockerfile.gpu \
    --target prod \
    -t vieneu-tts:gpu \
    .

# Build GPU image (dev stage — có shell, dùng cho development)
docker build \
    -f docker/Dockerfile.gpu \
    --target dev \
    -t vieneu-tts:gpu-dev \
    .

# Build Serve image (LMDeploy API server)
docker build \
    -f docker/Dockerfile.serve \
    -t vieneu-tts:serve \
    .

# Build với build cache (nhanh hơn lần sau)
docker build \
    -f docker/Dockerfile.gpu \
    --target prod \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t vieneu-tts:gpu \
    .
```

**Giải thích multi-stage build trong Dockerfile.gpu:**

- `base` stage: Cài OS dependencies, Python 3.12, uv. Dùng chung cho cả dev và prod.
- `dev` stage: Chỉ chạy `uv sync` (cài tất cả deps kể cả dev tools). Container drops vào bash. Thư mục code được mount từ host — thay đổi code trên host phản ánh ngay trong container. Dùng cho development.
- `prod` stage: Copy toàn bộ source code vào image, chạy `uv sync --no-dev --group gpu` (chỉ production deps). Tự động chạy `vieneu-web` khi container start. Dùng để deploy lên server.

### 3.4 Cấu hình docker-compose.yml chi tiết

Giải thích từng section của `docker/docker-compose.yml`:

```yaml
# Anchor YAML để tái sử dụng config (DRY)
x-base-config: &base-config
  volumes:
    # Mount source code từ host → container (live reload trong dev)
    - ./:/workspace
    # Named volume cho HuggingFace model cache — tồn tại qua restart
    - huggingface_cache:/root/.cache/huggingface
    # Mount thư mục output audio ra host để dễ truy cập
    - ./output_audio:/workspace/output_audio
    # Cache cho uv và pip (tăng tốc rebuild)
    - uv_cache:/root/.cache/uv
    - pip_cache:/root/.cache/pip
  env_file:
    # Load biến môi trường từ .env (optional — không fail nếu không có)
    - path: .env
      required: false
  stdin_open: true  # Giữ stdin mở (cho interactive shell)
  tty: true         # Allocate pseudo-TTY

services:
  serve:
    profiles: ["serve"]   # Chỉ chạy khi --profile serve
    build:
      context: ..          # Build context là thư mục cha (project root)
      dockerfile: docker/Dockerfile.serve
    container_name: vieneu-tts-serve
    ports:
      - "${SERVE_PORT:-23333}:23333"  # Env var với fallback default
    volumes:
      # Chỉ mount HF cache, không mount source (serve dùng image đã baked)
      - huggingface_cache:/root/.cache/huggingface
    command:
      - "--model"
      - "${MODEL:-pnnbao-ump/VieNeu-TTS-v2}"   # Override qua .env
      - "--model-name"
      - "${MODEL_NAME:-pnnbao-ump/VieNeu-TTS-v2}"
      - "--port"
      - "23333"
      - "--memory-util"
      - "${MEMORY_UTIL:-0.3}"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all           # Dùng tất cả GPU có sẵn
              capabilities: [gpu]  # Yêu cầu NVIDIA GPU capability
    restart: unless-stopped  # Auto-restart trừ khi stop thủ công

  gpu:
    <<: *base-config  # Merge config từ anchor
    profiles: ["gpu"]
    build:
      context: ..
      dockerfile: docker/Dockerfile.gpu
      target: dev    # Dùng dev stage (có volume mount, hot reload)
    container_name: vieneu-tts-gpu-dev
    ports:
      - "${PORT:-8001}:${GRADIO_SERVER_PORT:-8001}"
    command: ["uv", "run", "apps/gradio_main.py"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

# Named volumes — tồn tại độc lập với container lifecycle
volumes:
  huggingface_cache:   # HuggingFace model cache (~5-30 GB)
  uv_cache:            # uv package cache (tăng tốc rebuild)
  pip_cache:           # pip cache (fallback)
```

### 3.5 Environment Variables cho Docker

| Variable | Default | Mô tả | Ví dụ |
|---|---|---|---|
| `HF_HOME` | `/root/.cache/huggingface` | Thư mục cache HuggingFace trong container | `/cache/hf` |
| `PORT` | `8001` | Port expose ra host | `8080` |
| `GRADIO_SERVER_PORT` | `8001` | Port Gradio lắng nghe trong container | `8001` |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Interface Gradio bind | `0.0.0.0` |
| `GRADIO_SHARE` | `0` | Bật Gradio public share | `0` hoặc `1` |
| `MODEL` | `pnnbao-ump/VieNeu-TTS-v2` | HF model ID cho serve | `pnnbao-ump/VieNeu-TTS` |
| `MODEL_NAME` | `pnnbao-ump/VieNeu-TTS-v2` | Model name trong LMDeploy API | bất kỳ string |
| `SERVE_PORT` | `23333` | Port expose cho API serve | `23333` |
| `MEMORY_UTIL` | `0.3` | GPU memory fraction cho LMDeploy | `0.3` đến `0.9` |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering | `1` |

**Tạo file .env mẫu:**

```bash
cat > .env << 'EOF'
# HuggingFace settings
HF_HOME=/root/.cache/huggingface

# Gradio settings
PORT=8001
GRADIO_SERVER_PORT=8001
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SHARE=0

# Serve (LMDeploy API) settings
SERVE_PORT=23333
MODEL=pnnbao-ump/VieNeu-TTS-v2
MODEL_NAME=pnnbao-ump/VieNeu-TTS-v2
MEMORY_UTIL=0.3

# Optional: HuggingFace token (cho model private)
# HF_TOKEN=hf_your_token_here
EOF
```

### 3.6 Persistent Model Cache

**Vấn đề:** Mỗi lần rebuild container hoặc pull image mới, Docker tạo container fresh. Nếu không dùng volume, toàn bộ model (~5–30 GB) sẽ phải download lại từ HuggingFace.

**Giải pháp — bind mount vào ổ lưu trữ lớn:**

```yaml
# docker-compose.override.yml — tạo file này để override mà không sửa file gốc
services:
  gpu:
    volumes:
      - /data/hf_cache:/root/.cache/huggingface  # Bind mount thư mục thực tế

volumes:
  huggingface_cache:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/hf_cache   # Thay bằng đường dẫn thực tế
```

**Tạo thư mục trên host:**

```bash
sudo mkdir -p /data/hf_cache
sudo chown 0:0 /data/hf_cache  # Container thường chạy root
sudo chmod 755 /data/hf_cache
```

**Pre-download model trước khi chạy container (tiết kiệm thời gian):**

```bash
# Download model ra ngoài container vào bind-mounted directory
HF_HOME=/data/hf_cache uv run huggingface-cli download pnnbao-ump/VieNeu-TTS-v3-Turbo
HF_HOME=/data/hf_cache uv run huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX
```

### 3.7 Remote API Server Container

**Chạy với docker run:**

```bash
# Chạy với model VieNeu-TTS-v2, expose port 23333
docker run --gpus all \
    --name vieneu-serve \
    -p 23333:23333 \
    -e HF_HOME=/cache \
    -e PYTHONUNBUFFERED=1 \
    -v /data/hf_cache:/cache \
    --restart unless-stopped \
    vieneu-tts:serve \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --port 23333 \
    --memory-util 0.4

# Hoặc pull và chạy image có sẵn từ Docker Hub
docker run --gpus all \
    --name vieneu-serve \
    -p 23333:23333 \
    -v /data/hf_cache:/root/.cache/huggingface \
    --restart unless-stopped \
    pnnbao97/vieneu-tts:serve \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --tunnel
```

**Kiểm tra API server đang hoạt động:**

```bash
# Health check
curl http://localhost:23333/v1/models

# Test request đơn giản
curl -X POST http://localhost:23333/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "pnnbao-ump/VieNeu-TTS-v2",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10
    }'
```

### 3.8 bore Tunnel (Public URL)

**Tại sao cần tunnel?**
- Server đứng sau NAT/firewall (phổ biến với VPS giá rẻ, home server)
- Không có public IP hoặc domain
- Muốn share demo nhanh mà không setup DNS

**bore.pub tunnel** (được tích hợp sẵn trong Dockerfile.serve):

```bash
# Cài bore standalone (không qua Docker)
curl -LsSf https://github.com/ekzhang/bore/releases/download/v0.5.1/bore-v0.5.1-x86_64-unknown-linux-musl.tar.gz | tar zxf - -C /usr/local/bin

# Expose port 23333 ra bore.pub
bore local 23333 --to bore.pub

# Output mẫu:
# Listening at bore.pub:12345
# → URL công khai: http://bore.pub:12345
```

**Trong container serve, tự động chạy bore khi dùng `--tunnel`:**

```bash
docker run --gpus all \
    -p 23333:23333 \
    -v /data/hf_cache:/root/.cache/huggingface \
    vieneu-tts:serve \
    --model pnnbao-ump/VieNeu-TTS-v2 \
    --tunnel  # bore tunnel tự động bật
```

**Ngrok như alternative** (nếu cần URL cố định):

```bash
# Cài ngrok
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Authenticate
ngrok authtoken YOUR_NGROK_TOKEN

# Expose port
ngrok http 23333
```

### 3.9 Health Check & Auto-restart

**Thêm healthcheck vào docker-compose:**

```yaml
# docker-compose.override.yml
services:
  gpu:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s  # Chờ model load xong trước khi check
    restart: unless-stopped

  serve:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:23333/v1/models"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 180s  # LMDeploy load model lâu hơn
```

**Xem health status:**

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
# STATUS sẽ hiển thị: Up 5 minutes (healthy) hoặc (unhealthy)

# Xem log healthcheck
docker inspect vieneu-tts-gpu-dev | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(json.dumps(data[0]['State']['Health'], indent=2))
"
```

### 3.10 CI/CD Pipeline cơ bản

**Script build và push lên Docker Hub:**

```bash
# build_and_push.sh
#!/bin/bash
set -e

VERSION=${1:-latest}
REGISTRY="pnnbao97"
IMAGE_GPU="$REGISTRY/vieneu-tts"
IMAGE_SERVE="$REGISTRY/vieneu-tts"

echo "Building GPU image..."
docker build \
    -f docker/Dockerfile.gpu \
    --target prod \
    -t "$IMAGE_GPU:gpu-$VERSION" \
    -t "$IMAGE_GPU:gpu-latest" \
    .

echo "Building Serve image..."
docker build \
    -f docker/Dockerfile.serve \
    -t "$IMAGE_SERVE:serve-$VERSION" \
    -t "$IMAGE_SERVE:serve-latest" \
    .

echo "Pushing images..."
docker push "$IMAGE_GPU:gpu-$VERSION"
docker push "$IMAGE_GPU:gpu-latest"
docker push "$IMAGE_SERVE:serve-$VERSION"
docker push "$IMAGE_SERVE:serve-latest"

echo "Done! Images pushed:"
echo "  $IMAGE_GPU:gpu-$VERSION"
echo "  $IMAGE_SERVE:serve-$VERSION"
```

```bash
chmod +x build_and_push.sh
./build_and_push.sh 3.0.5
```

**Rolling update khi có image mới:**

```bash
# Pull image mới
docker compose -f docker/docker-compose.yml pull

# Restart với zero downtime (nếu có nhiều replica)
docker compose -f docker/docker-compose.yml --profile gpu up -d --no-deps gpu

# Hoặc force recreate
docker compose -f docker/docker-compose.yml --profile gpu up -d --force-recreate
```

### 3.11 Troubleshooting Docker

#### nvidia-container-toolkit chưa cài hoặc cài sai

```bash
# Lỗi: "docker: Error response from daemon: could not select device driver"
# Kiểm tra toolkit đã cài chưa
dpkg -l | grep nvidia-container

# Nếu chưa có, cài lại theo bước 3.1
# Sau khi cài, restart docker daemon
sudo systemctl restart docker

# Test lại
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

#### CUDA version mismatch giữa host driver và container

```bash
# Lỗi: "CUDA error: no kernel image is available for execution on the device"
# Host driver phải >= CUDA version trong container image

# Kiểm tra host driver
nvidia-smi | grep "Driver Version"

# Kiểm tra CUDA version trong container
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvcc --version

# Nếu driver host quá cũ, cài driver mới hơn:
sudo apt install -y nvidia-driver-545
sudo reboot
```

#### Volume permission issues

```bash
# Lỗi: "Permission denied" khi container ghi vào volume
# Container thường chạy root (uid 0), nhưng bind mount directory có owner khác

# Fix: cho phép container ghi vào thư mục
sudo chmod 777 /data/hf_cache  # Đơn giản nhất

# Hoặc thêm user namespace mapping trong docker run
docker run --user $(id -u):$(id -g) ...
```

#### Out of disk space (model cache quá lớn)

```bash
# Xem disk usage của Docker
docker system df

# Xem size của từng volume
docker volume ls -q | xargs docker volume inspect | python3 -c "
import json, sys
data = json.load(sys.stdin)
for v in data:
    print(v['Name'], v['Mountpoint'])
"
du -sh /var/lib/docker/volumes/*/

# Xóa container, image không dùng (không xóa volumes)
docker system prune -a

# Xóa HF cache cũ (cẩn thận — sẽ phải download lại)
docker volume rm vieneutts_huggingface_cache

# Xóa model cụ thể
find /data/hf_cache -name "*.bin" -size +1G -ls
```

#### Container bị OOM Killed

```bash
# Triệu chứng: container tự dừng, docker inspect hiển thị OOMKilled: true
docker inspect vieneu-tts-gpu-dev | grep -A5 OOMKilled

# Giải pháp 1: Giảm memory_util
# Sửa trong .env: MEMORY_UTIL=0.2

# Giải pháp 2: Tăng memory limit cho container
# Thêm vào docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 16g

# Giải pháp 3: Dùng swap trong container (Linux host)
sudo sysctl vm.swappiness=10
```

#### Gradio WebSocket không hoạt động qua Nginx

```bash
# Triệu chứng: UI load nhưng synthesis không phản hồi, console lỗi WebSocket

# Kiểm tra Nginx config có các header này không:
# proxy_http_version 1.1;
# proxy_set_header Upgrade $http_upgrade;
# proxy_set_header Connection "upgrade";

# Kiểm tra timeout đủ dài chưa (synthesis có thể mất 30-60s):
# proxy_read_timeout 300;

# Test WebSocket trực tiếp (không qua Nginx)
curl http://localhost:8001/  # Nếu OK, vấn đề ở Nginx

# Reload Nginx sau khi sửa config
sudo nginx -t && sudo systemctl reload nginx
```

---

## Bảng so sánh nhanh (tổng kết)

| Tiêu chí | Windows (native) | Linux (native) | Docker |
|---|---|---|---|
| **Dễ cài đặt ban đầu** | Trung bình | Thấp | Thấp (sau khi có Docker) |
| **Ổn định production** | Kém (cần NSSM) | Tốt (systemd) | Tốt nhất |
| **GPU support** | Có (giới hạn) | Đầy đủ | Đầy đủ |
| **CPU/ONNX mode** | Có | Có | Có |
| **Isolated environment** | Không | Không | Có |
| **Dễ scale** | Khó | Trung bình | Tốt |
| **Hot reload code** | Có | Có | Dev mode: Có |
| **Auto-restart** | Cần NSSM/Task Scheduler | systemd native | restart policy |
| **SSL/HTTPS** | Cần IIS hoặc Caddy | Nginx + Certbot | Nginx + Certbot |
| **Multi-GPU** | Phức tạp | Tốt | Tốt |
| **Triton support** | Hạn chế | Đầy đủ | Đầy đủ |
| **Dễ troubleshoot** | Khó | Tốt (journald) | Trung bình |
| **Reproducible build** | Không | Một phần | Hoàn toàn |
| **Phù hợp cho** | Dev cá nhân, thử nghiệm | Server production | DevOps, CI/CD |

**Khuyến nghị theo use case:**

- **Thử nghiệm nhanh trên laptop Windows**: `uv sync` + `uv run vieneu-web` — không cần Docker, không cần GPU.
- **Server production Linux (1 GPU)**: Cài native + systemd + Nginx. Ít overhead nhất, dễ debug.
- **Nhiều server hoặc cần rollback**: Docker + docker-compose. Reproducible, dễ update bằng cách pull image mới.
- **Chỉ cần API (không cần UI)**: `serve` profile trong Docker — LMDeploy API server, kết hợp với `RemoteVieNeuTTS` SDK từ client.
- **GPU không có, chỉ CPU**: `uv sync` (không có `--group gpu`), v3 Turbo ONNX mode tự động được chọn.
