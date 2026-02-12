from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class HazardModel:
    label: str
    name_zh: str
    within_affected_yield_loss: float  # 0..1
    spillover_coeff: float             # >=0
    base_uncertainty: float            # relative half-width baseline (0..1)
    note: str = ""

    def yield_loss_fraction(self, affected_ratio: float) -> float:
        """
        Convert affected area ratio r to whole-field yield loss fraction f in [0,1].
        Simple, explainable default:
          f = r * within_loss + (r^2) * spillover_coeff
        """
        r = clamp(affected_ratio, 0.0, 1.0)
        f = r * self.within_affected_yield_loss + (r * r) * self.spillover_coeff
        return clamp(f, 0.0, 1.0)

    def uncertainty_band(self, p50: float, confidence: float) -> Tuple[float, float, float]:
        """
        Produce (p10, p50, p90) with a relative band influenced by confidence.
        rel = base_uncertainty + (1 - confidence) * 0.6
        """
        c = clamp(confidence, 0.0, 1.0)
        rel = clamp(self.base_uncertainty + (1.0 - c) * 0.6, 0.05, 0.90)

        p10 = max(0.0, p50 * (1.0 - rel))
        p90 = min(1.0, p50 * (1.0 + rel))
        return (p10, p50, p90)


def combine_independent_loss(fracs: list[float]) -> float:
    """
    Combine multiple yield-loss fractions into a single total fraction,
    using 1 - Π(1 - f_i) to avoid exceeding 1.
    """
    prod = 1.0
    for f in fracs:
        prod *= (1.0 - clamp(f, 0.0, 1.0))
    return clamp(1.0 - prod, 0.0, 1.0)


def load_hazard_models(hazard_models_json: Dict[str, Any]) -> Dict[str, HazardModel]:
    models: Dict[str, HazardModel] = {}
    for label, obj in (hazard_models_json.get("hazards") or {}).items():
        models[label] = HazardModel(
            label=label,
            name_zh=str(obj.get("name_zh", label)),
            within_affected_yield_loss=float(obj.get("within_affected_yield_loss", 0.5)),
            spillover_coeff=float(obj.get("spillover_coeff", 0.0)),
            base_uncertainty=float(obj.get("base_uncertainty", 0.35)),
            note=str(obj.get("note", "")),
        )
    return models
