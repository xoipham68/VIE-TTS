"""Thread-safe VieNeu engine pool.

Vieneu instances are NOT thread-safe, so we store one per OS thread
via threading.local(). The pool is lazily initialised on first use.
"""

from __future__ import annotations

import threading
import logging
from typing import Any

# Import ở module level để tests có thể patch api.engine_pool.Vieneu
from vieneu import Vieneu

logger = logging.getLogger("vieneu.api.engine_pool")

_thread_local = threading.local()
_engine_kwargs: dict[str, Any] = {}
_engine_mode: str = "v3turbo"


def configure(mode: str = "v3turbo", **kwargs: Any) -> None:
    """Gọi một lần khi khởi động app để cấu hình engine sẽ được tạo."""
    global _engine_mode, _engine_kwargs
    _engine_mode = mode
    _engine_kwargs = kwargs
    logger.info("Engine pool configured: mode=%s kwargs=%s", mode, list(kwargs.keys()))


def get_engine():
    """Trả về Vieneu instance cho thread hiện tại, tạo mới nếu chưa có."""
    if not hasattr(_thread_local, "engine"):
        logger.info("Creating new Vieneu engine (thread=%s)", threading.current_thread().name)
        _thread_local.engine = Vieneu(mode=_engine_mode, **_engine_kwargs)
    return _thread_local.engine


def close_engine() -> None:
    """Giải phóng engine của thread hiện tại (gọi trong shutdown hook)."""
    engine = getattr(_thread_local, "engine", None)
    if engine is not None:
        try:
            engine.close()
        except Exception:
            pass
        del _thread_local.engine


def reset_pool() -> None:
    """Xoá engine trên thread hiện tại — dùng trong tests để tái khởi tạo."""
    if hasattr(_thread_local, "engine"):
        del _thread_local.engine
