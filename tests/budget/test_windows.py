from __future__ import annotations

import time

from app.budget import day_window, minute_window


def test_minute_window_aligns_to_the_epoch_minute():
    window = minute_window(1_700_000_123.0)
    assert window.start == 1_700_000_100
    assert window.end == 1_700_000_160


def test_day_window_starts_at_utc_midnight():
    window = day_window(1_700_000_123.0)  # 2023-11-14T22:28:43Z
    assert window.start == 1_699_920_000  # 2023-11-14T00:00:00Z
    assert window.end == window.start + 86_400


def test_reset_in_never_goes_negative():
    window = minute_window(time.time() - 120)
    assert window.reset_in >= 0.0
