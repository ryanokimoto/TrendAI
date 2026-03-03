# my_parts/explain.py
from __future__ import annotations

def explain_recommendation(tag: str, niche_name: str, sim: float, vel: float, sat: float) -> str:
    reasons = []
    reasons.append(f"matches your content well (similarity={sim:.2f})")

    if vel >= 0.6:
        reasons.append("is trending upward in this niche")
    elif vel <= 0.2:
        reasons.append("is stable (not spiking)")

    if sat <= 0.4:
        reasons.append("is not overly saturated")
    else:
        reasons.append("is commonly used (more saturated)")

    return f"#{tag} ({niche_name}): " + "; ".join(reasons) + "."
