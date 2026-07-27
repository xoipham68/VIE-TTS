"""VieNeu engine pool — MỘT engine dùng chung cho cả process, nạp sẵn lúc khởi động.

Vì sao KHÔNG dùng threading.local() như bản cũ
----------------------------------------------
Các endpoint trong api/server.py là `def` (sync) → FastAPI chạy chúng trong anyio
threadpool (mặc định ~40 thread). Với `threading.local()`:

  1. Preload trong lifespan chỉ warm thread của event loop — KHÔNG phải thread mà
     endpoint thực sự chạy. Nói cách khác, "gọi get_engine() lúc khởi động" hoàn toàn
     KHÔNG giải quyết được độ trễ của request đầu tiên.
  2. Mỗi thread mới chạm engine sẽ nạp THÊM một bản model → phình RAM/VRAM âm thầm.
  3. close_engine() lúc shutdown chỉ đóng engine của đúng thread gọi nó → rò rỉ phần còn lại.

Giải pháp: một instance duy nhất cho cả process, mọi lời gọi infer đi qua một Lock.
Engine vốn KHÔNG thread-safe nên tuần tự hoá là đúng về mặt ngữ nghĩa — không mất mát gì.
Cần chạy song song thật thì tăng số process (`--workers N`), lưu ý mỗi worker nạp một
bản model riêng.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

# Import ở module level để tests có thể patch api.engine_pool.Vieneu
from vieneu import Vieneu

logger = logging.getLogger("vieneu.api.engine_pool")

# Câu ngắn để "chạy nóng" engine: lần infer đầu tiên còn phải dựng graph / cấp phát
# kernel, nên nếu không warmup thì request THẬT đầu tiên vẫn chậm dù model đã nạp.
WARMUP_TEXT = "Xin chào."

_engine: Optional[Any] = None
_engine_kwargs: dict[str, Any] = {}
_engine_mode: str = "v3turbo"

# Khoá khởi tạo (chống 2 thread cùng nạp model) và khoá suy luận (engine không thread-safe).
_init_lock = threading.Lock()
_infer_lock = threading.Lock()

# "idle" → chưa nạp | "loading" → đang nạp | "ready" → sẵn sàng | "error" → nạp lỗi
_state: str = "idle"
_error: Optional[str] = None
_load_seconds: float = 0.0


def configure(mode: str = "v3turbo", **kwargs: Any) -> None:
    """Cấu hình engine sẽ được tạo. Gọi một lần lúc khởi động, TRƯỚC preload()."""
    global _engine_mode, _engine_kwargs
    _engine_mode = mode
    _engine_kwargs = kwargs
    logger.info("Engine pool configured: mode=%s kwargs=%s", mode, list(kwargs.keys()))


def _create_engine() -> Any:
    """Nạp model. Tốn hàng chục giây — chỉ chạy đúng một lần cho mỗi process."""
    global _state, _error, _load_seconds
    t0 = time.perf_counter()
    _state, _error = "loading", None
    logger.info("⏳ Đang nạp VieNeu engine (mode=%s)...", _engine_mode)
    try:
        eng = Vieneu(mode=_engine_mode, **_engine_kwargs)
    except Exception as exc:
        _state, _error = "error", str(exc)
        logger.exception("❌ Nạp VieNeu engine thất bại: %s", exc)
        raise
    _load_seconds = time.perf_counter() - t0
    _state = "ready"
    logger.info("✅ VieNeu engine sẵn sàng sau %.1fs", _load_seconds)
    return eng


def get_engine(create: bool = True) -> Optional[Any]:
    """Trả engine dùng chung.

    create=False → trả None nếu chưa nạp, KHÔNG kích hoạt nạp model. Dùng cho /health:
    health check tuyệt đối không được block chờ nạp model (đó chính là nguyên nhân
    client probe 5s bị timeout rồi kết luận nhầm "server chưa chạy").
    """
    global _engine
    if _engine is not None:
        return _engine
    if not create:
        return None
    with _init_lock:
        if _engine is None:                      # double-checked locking
            _engine = _create_engine()
    return _engine


def warmup() -> bool:
    """Chạy một lần infer ngắn để engine hết "lạnh". Lỗi warmup không phải lỗi chí mạng."""
    eng = get_engine()
    try:
        t0 = time.perf_counter()
        with _infer_lock:
            eng.infer(text=WARMUP_TEXT)
        logger.info("🔥 Warmup xong sau %.1fs", time.perf_counter() - t0)
        return True
    except Exception as exc:
        logger.warning("Warmup thất bại (không chí mạng): %s", exc)
        return False


def preload(do_warmup: bool = True) -> bool:
    """Nạp + chạy nóng engine. Gọi ĐỒNG BỘ trong lifespan trước khi app nhận request.

    Trả False nếu nạp lỗi — server vẫn sống để /health báo lỗi rõ ràng thay vì
    chết câm lúc khởi động.
    """
    if os.getenv("VIENEU_PRELOAD", "1").lower() in ("0", "false", "no"):
        logger.warning("VIENEU_PRELOAD=0 → BỎ QUA nạp sẵn. Request đầu tiên sẽ rất chậm.")
        return False
    try:
        get_engine()
    except Exception:
        return False
    if do_warmup:
        warmup()
    return True


def infer(**kwargs: Any):
    """Gọi engine.infer() có khoá — engine KHÔNG thread-safe nên phải tuần tự hoá."""
    eng = get_engine()
    with _infer_lock:
        return eng.infer(**kwargs)


def is_ready() -> bool:
    return _state == "ready" and _engine is not None


def get_state() -> dict:
    """Trạng thái engine — đọc tức thời, KHÔNG chạm model."""
    device = "unknown"
    if _engine is not None:
        device = str(getattr(_engine, "device", "cpu"))
    return {
        "state": _state,
        "ready": is_ready(),
        "mode": _engine_mode,
        "device": device,
        "load_seconds": round(_load_seconds, 2),
        "error": _error,
    }


def close_engine() -> None:
    """Giải phóng engine dùng chung (gọi trong shutdown hook)."""
    global _engine, _state
    with _init_lock:
        if _engine is not None:
            try:
                _engine.close()
            except Exception:
                pass
            _engine = None
        _state = "idle"


def reset_pool() -> None:
    """Xoá engine — dùng trong tests để tái khởi tạo."""
    global _engine, _state, _error, _load_seconds
    with _init_lock:
        _engine = None
        _state, _error, _load_seconds = "idle", None, 0.0
