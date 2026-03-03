# my_parts/trend_metrics.py
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple

import numpy as np


def parse_unix_seconds(ts: Any):
    """
    Metadata shows ts like "1724128903" (string).
    """
    try:
        return datetime.fromtimestamp(int(float(ts)), tz=timezone.utc)
    except Exception:
        return None


def compute_trend_velocity(
    video_metadata_list,
    hashtag_to_niche_name: Dict[str, str],
    window_days: int = 7,
    now: datetime | None = None
) -> Dict[Tuple[str, str], float]:
    """
    Returns:
      velocity[(niche_name, hashtag)] = (recent - prev) / (prev + 1)

    recent window: [now-window_days, now]
    prev window:   [now-2*window_days, now-window_days)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    recent_start = now - timedelta(days=window_days)
    prev_start = now - timedelta(days=2 * window_days)

    recent = defaultdict(int)
    prev = defaultdict(int)

    for meta in video_metadata_list:
        ts = parse_unix_seconds(meta.get("timestamp"))
        if ts is None:
            continue

        tags = meta.get("hashtags", []) or []
        for tag in tags:
            niche = hashtag_to_niche_name.get(tag)
            if niche is None:
                continue

            key = (niche, tag)
            if ts >= recent_start:
                recent[key] += 1
            elif prev_start <= ts < recent_start:
                prev[key] += 1

    velocity = {}
    keys = set(recent.keys()) | set(prev.keys())
    for key in keys:
        r = recent.get(key, 0)
        p = prev.get(key, 0)
        velocity[key] = (r - p) / (p + 1.0)

    return velocity


def normalize_scores(d: Dict, clip: bool = True) -> Dict:
    """
    Normalize dict values to [0,1].
    """
    if not d:
        return d
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        return {k: 0.0 for k in d}
    out = {k: (float(v) - lo) / (hi - lo) for k, v in d.items()}
    if clip:
        out = {k: max(0.0, min(1.0, v)) for k, v in out.items()}
    return out
