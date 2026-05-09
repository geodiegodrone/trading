from __future__ import annotations

import multi_bot


def test_live_gate_requires_mode_demo_only() -> None:
    ready, reason = multi_bot._live_gate_ready({"ready_for_live": True, "mode_demo_only": False, "reason": "demo disabled"})
    assert ready is False
    assert "demo" in reason.lower() or "mode_demo_only" in reason


if __name__ == "__main__":
    test_live_gate_requires_mode_demo_only()
    print("test_multi_bot_primary_only_ok")
