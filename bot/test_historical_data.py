from __future__ import annotations

from pathlib import Path

import pandas as pd

import historical_data


def test_historical_download_incremental_and_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(historical_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(historical_data, "_utc_now_ms", lambda: 1_700_000_000_000)

    calls = []

    def fake_request(symbol, tf_min, start_ms, end_ms, limit=1000, retries=5):
        calls.append((start_ms, end_ms))
        rows = []
        for step in range(3):
            ts = start_ms + step * 3600000
            if ts >= end_ms:
                break
            rows.append([ts, 100 + step, 101 + step, 99 + step, 100.5 + step, 10 + step, ts + 3599999, 1000 + step, 1, 5 + step, 500 + step, 0])
        return rows

    monkeypatch.setattr(historical_data, "_request_klines", fake_request)

    first = historical_data.download_history("BTCUSDT", 60, 1, force=True)
    assert int(first["rows"]) > 0
    frame = historical_data.load_history("BTCUSDT", 60)
    assert not frame.empty
    assert frame["ts"].is_monotonic_increasing

    existing = frame.copy()
    extra = existing.tail(2).copy()
    extra.loc[:, "ts"] = extra["ts"] + 2 * 3600000
    merged = pd.concat([existing, extra], ignore_index=True)
    merged.to_parquet(Path(tmp_path) / "BTCUSDT_60m.parquet", index=False, compression="zstd")

    second = historical_data.download_history("BTCUSDT", 60, 2, force=False)
    frame2 = historical_data.load_history("BTCUSDT", 60)
    assert int(second["rows"]) == len(frame2)
    assert frame2["ts"].is_unique
    assert calls, "expected paginated requests"


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_historical_download_incremental_and_roundtrip(Path(tmp), __import__("pytest").MonkeyPatch())
    print("test_historical_data_ok")
