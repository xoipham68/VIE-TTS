"""Kiểm tra engine pool: MỘT engine cho cả process, /health không chạm model.

Chạy:  python api/test_engine_pool.py
Không cần GPU / không tải model thật — Vieneu được thay bằng bản giả.
"""
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_created = []


class FakeEngine:
    """Giả lập engine: __init__ chậm (như nạp model thật)."""

    def __init__(self, mode="v3turbo", **kw):
        time.sleep(0.3)                     # giả lập nạp model
        _created.append(self)
        self.device = "cpu"
        self.sample_rate = 48000
        self.infer_calls = 0
        self._concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def infer(self, **kw):
        with self._lock:
            self._concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self._concurrent)
        time.sleep(0.05)
        with self._lock:
            self._concurrent -= 1
            self.infer_calls += 1
        return [0.0]

    def close(self):
        pass


def main() -> bool:
    ok = True
    with patch("api.engine_pool.Vieneu", FakeEngine):
        import api.engine_pool as pool
        pool.reset_pool()
        _created.clear()

        # 1. /health không được kích hoạt nạp model
        st = pool.get_state()
        if st["state"] != "idle" or _created:
            print(f"❌ get_state() đã tự nạp model (state={st['state']}, created={len(_created)})")
            ok = False
        else:
            print("✅ get_state() không chạm model (state=idle, 0 engine)")

        if pool.get_engine(create=False) is not None:
            print("❌ get_engine(create=False) lại tạo engine")
            ok = False
        else:
            print("✅ get_engine(create=False) trả None, không tạo engine")

        # 2. preload nạp đúng 1 lần + warmup
        pool.configure(mode="v3turbo")
        t0 = time.perf_counter()
        pool.preload(do_warmup=True)
        print(f"✅ preload xong sau {time.perf_counter() - t0:.2f}s | ready={pool.is_ready()}")
        if len(_created) != 1:
            print(f"❌ preload tạo {len(_created)} engine, mong đợi 1")
            ok = False

        # 3. NHIỀU THREAD (giống anyio threadpool của FastAPI) → vẫn chỉ 1 engine
        errors = []

        def worker():
            try:
                pool.infer(text="test")
            except Exception as e:      # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            print(f"❌ {len(errors)} thread lỗi: {errors[0]}")
            ok = False
        if len(_created) != 1:
            print(f"❌ 20 thread tạo {len(_created)} engine — vẫn bị nhân bản model!")
            ok = False
        else:
            print(f"✅ 20 thread dùng CHUNG 1 engine (bản cũ threading.local sẽ tạo tới 20)")

        eng = pool.get_engine()
        if eng.max_concurrent > 1:
            print(f"❌ Có {eng.max_concurrent} lời gọi infer chạy song song — engine không thread-safe!")
            ok = False
        else:
            print("✅ infer() được tuần tự hoá (max 1 lời gọi cùng lúc)")

        # 4. Trạng thái báo cáo đúng
        st = pool.get_state()
        if not (st["ready"] and st["state"] == "ready" and st["load_seconds"] > 0):
            print(f"❌ get_state() sai: {st}")
            ok = False
        else:
            print(f"✅ get_state(): ready=True, nạp {st['load_seconds']}s, device={st['device']}")

        # 5. close giải phóng engine dùng chung
        pool.close_engine()
        if pool.is_ready() or pool.get_engine(create=False) is not None:
            print("❌ close_engine() chưa giải phóng")
            ok = False
        else:
            print("✅ close_engine() giải phóng engine dùng chung")

    print()
    print("KẾT QUẢ:", "✅ PASS" if ok else "❌ FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
