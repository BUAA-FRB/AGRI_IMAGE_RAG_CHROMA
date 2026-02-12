from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import read_json, find_stats_entry, parse_tile_id_from_path, pick_best_evidence_tile
from .loss_models import load_hazard_models, combine_independent_loss
from .models import EconomyItem, EconomyResult


M2_PER_HA = 10000.0
M2_PER_MU = 666.6666666667


@dataclass
class EconomyConfig:
    crop: str = "corn"
    stage: str = "generic"
    currency: str = "CNY"

    area_mode: str = "gsd"  # "gsd" | "ratio_only"
    gsd_m_per_px: Optional[float] = 0.1

    # if ratio_only, normalize money per 1 mu or 1 ha
    normalize_to: str = "mu"  # "mu" | "ha"


class EconomyAgent:
    def __init__(self, params_dir: str | Path) -> None:
        self.params_dir = Path(params_dir)
        self.crop_db = read_json(self.params_dir / "crop_economy.json")
        self.hazard_db_raw = read_json(self.params_dir / "hazard_models.json")
        self.hazard_models = load_hazard_models(self.hazard_db_raw)

    def _get_crop_params(self, crop: str) -> Dict[str, Any]:
        crops = (self.crop_db.get("crops") or {})
        if crop in crops:
            return crops[crop]
        default_crop = self.crop_db.get("default_crop")
        if default_crop and default_crop in crops:
            return crops[default_crop]
        for _, v in crops.items():
            return v
        raise ValueError("crop_economy.json has no crops defined.")

    def _per_area_value(self, crop_params: Dict[str, Any]) -> Tuple[str, float, float]:
        """
        Returns (currency, value_per_ha, value_per_mu) using:
          value = yield_kg_per_ha * price_per_kg
        """
        currency = str(self.crop_db.get("currency", "CNY"))
        y_ha = float(crop_params.get("yield_kg_per_ha", 0.0))
        p = float(crop_params.get("price_per_kg", 0.0))
        value_per_ha = y_ha * p
        value_per_mu = value_per_ha * (M2_PER_MU / M2_PER_HA)
        return currency, value_per_ha, value_per_mu

    def _convert_areas(
        self,
        affected_px: float,
        field_px: float,
        cfg: EconomyConfig,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Return:
          affected_m2, field_m2, affected_ha, field_ha, affected_mu, field_mu
        """
        if cfg.area_mode != "gsd" or cfg.gsd_m_per_px is None:
            return (None, None, None, None, None, None)

        g = float(cfg.gsd_m_per_px)
        affected_m2 = affected_px * (g * g)
        field_m2 = field_px * (g * g)
        affected_ha = affected_m2 / M2_PER_HA
        field_ha = field_m2 / M2_PER_HA
        affected_mu = affected_m2 / M2_PER_MU
        field_mu = field_m2 / M2_PER_MU
        return (affected_m2, field_m2, affected_ha, field_ha, affected_mu, field_mu)

    def estimate(
        self,
        latest_json_path: str | Path,
        field_stats_path: str | Path,
        cfg: Optional[EconomyConfig] = None,
    ) -> EconomyResult:
        cfg = cfg or EconomyConfig()
        latest_json_path = Path(latest_json_path)
        field_stats_path = Path(field_stats_path)

        latest = read_json(latest_json_path)
        field_stats = read_json(field_stats_path)

        warnings: List[str] = []
        assumptions: Dict[str, Any] = {}

        run_id = str(latest.get("id") or "unknown_run")

        # 1) predicted labels
        normalized = latest.get("normalized") or {}
        preds = normalized.get("predicted_labels") or []
        if not preds:
            parsed = latest.get("parsed_json") or {}
            preds = parsed.get("predicted_labels") or []
        if not preds:
            warnings.append("No predicted_labels found in latest.json (normalized/parsed_json).")
            preds = []

        # 2) determine tile_id
        query_path = str(latest.get("query_image_path") or "")
        tile_id = parse_tile_id_from_path(query_path)

        ev_tile_id, ev_split, ev_year, ev_source_path = pick_best_evidence_tile(latest)
        tile_split = ev_split
        tile_year = ev_year
        tile_source_path = ev_source_path

        if tile_id is None:
            tile_id = ev_tile_id

        if not tile_id:
            raise ValueError("Cannot determine tile_id from latest.json (query path + best evidence both failed).")

        assumptions["tile_id_resolution"] = {
            "query_image_path": query_path,
            "tile_id_from_query": parse_tile_id_from_path(query_path),
            "tile_id_from_best_evidence": ev_tile_id,
            "selected_tile_id": tile_id,
            "note": "If query is not a dataset tile, the nearest retrieved evidence tile is used as proxy.",
        }

        # 3) find stats entry
        stats_split, stats_key, entry = find_stats_entry(field_stats, tile_id)
        if not entry:
            raise FileNotFoundError(
                f"tile_id={tile_id} not found in field_stats.json. "
                f"Expected a key ending with '/{tile_id}.png'."
            )

        image_area_px = float(entry.get("image_area", 0.0))
        label_areas_map = entry.get("label_areas") or {}

        if image_area_px <= 0:
            raise ValueError(f"Invalid image_area in stats entry for tile_id={tile_id}: {image_area_px}")

        # 4) crop economics
        crop_params = self._get_crop_params(cfg.crop)
        currency, value_per_ha, value_per_mu = self._per_area_value(crop_params)

        assumptions["crop_params"] = {
            "crop": cfg.crop,
            "stage": cfg.stage,
            "currency": currency,
            "yield_kg_per_ha": crop_params.get("yield_kg_per_ha"),
            "price_per_kg": crop_params.get("price_per_kg"),
            "value_per_ha": value_per_ha,
            "value_per_mu": value_per_mu,
            "source_note": crop_params.get("source_note", ""),
        }

        # 5) estimate per label
        items: List[EconomyItem] = []
        loss_fracs_p50: List[float] = []
        loss_fracs_p10: List[float] = []
        loss_fracs_p90: List[float] = []

        # compute field area in real units if gsd
        # also compute base field value for totals (NOT sum of items)
        any_field_ha: Optional[float] = None
        any_field_mu: Optional[float] = None
        if cfg.area_mode == "gsd" and cfg.gsd_m_per_px is not None:
            field_m2 = image_area_px * (cfg.gsd_m_per_px ** 2)
            any_field_ha = field_m2 / M2_PER_HA
            any_field_mu = field_m2 / M2_PER_MU

        if cfg.area_mode == "gsd" and any_field_ha is not None:
            base_field_value_p50 = any_field_ha * value_per_ha
        else:
            # ratio_only: normalized base field value is per-1mu or per-1ha
            if cfg.normalize_to == "mu":
                base_field_value_p50 = value_per_mu
            else:
                base_field_value_p50 = value_per_ha

        for p in preds:
            label = str(p.get("label"))
            conf = float(p.get("confidence", 0.0))

            areas_list = label_areas_map.get(label) or []
            affected_px = float(sum(float(x) for x in areas_list)) if areas_list else 0.0
            affected_ratio = affected_px / image_area_px

            model = self.hazard_models.get(label)
            if model is None:
                from .loss_models import HazardModel
                model = HazardModel(
                    label=label,
                    name_zh=label,
                    within_affected_yield_loss=0.5,
                    spillover_coeff=0.0,
                    base_uncertainty=0.45,
                    note="Auto fallback hazard model (please define in hazard_models.json).",
                )
                warnings.append(f"Unknown hazard label '{label}' not in hazard_models.json, using fallback model.")

            yield_loss_p50 = model.yield_loss_fraction(affected_ratio)
            yl10, yl50, yl90 = model.uncertainty_band(yield_loss_p50, conf)

            affected_m2, field_m2, affected_ha, field_ha, affected_mu, field_mu = self._convert_areas(
                affected_px, image_area_px, cfg
            )

            # heuristic attribution money for this hazard (for breakdown only)
            # base on base_field_value_p50 (field total value) times per-hazard yield-loss fraction
            loss_value_p50 = base_field_value_p50 * yl50 if base_field_value_p50 is not None else None
            loss_value_p10 = base_field_value_p50 * yl10 if base_field_value_p50 is not None else None
            loss_value_p90 = base_field_value_p50 * yl90 if base_field_value_p50 is not None else None

            notes: List[str] = []
            notes.append(
                f"hazard={model.name_zh} within_loss={model.within_affected_yield_loss:.2f} spillover={model.spillover_coeff:.2f}"
            )
            if model.note:
                notes.append(model.note)

            if affected_px <= 0:
                notes.append("field_stats 中该 label_areas 为空或为 0，受灾比例为 0（可能是 tile 与预测标签不一致或确实无该灾害）。")

            loss_fracs_p50.append(yl50)
            loss_fracs_p10.append(yl10)
            loss_fracs_p90.append(yl90)

            items.append(
                EconomyItem(
                    label=label,
                    confidence=conf,
                    affected_area_px=affected_px,
                    field_area_px=image_area_px,
                    affected_ratio=affected_ratio,
                    affected_area_m2=affected_m2,
                    field_area_m2=field_m2,
                    affected_area_ha=affected_ha,
                    field_area_ha=field_ha,
                    affected_area_mu=affected_mu,
                    field_area_mu=field_mu,
                    yield_loss_frac_p50=yl50,
                    yield_loss_frac_p10=yl10,
                    yield_loss_frac_p90=yl90,
                    loss_value_p50=loss_value_p50,
                    loss_value_p10=loss_value_p10,
                    loss_value_p90=loss_value_p90,
                    currency=currency,
                    notes=notes,
                )
            )

        # 6) combine total yield-loss
        total_yield_loss_p50 = combine_independent_loss(loss_fracs_p50) if loss_fracs_p50 else 0.0
        total_yield_loss_p10 = combine_independent_loss(loss_fracs_p10) if loss_fracs_p10 else 0.0
        total_yield_loss_p90 = combine_independent_loss(loss_fracs_p90) if loss_fracs_p90 else 0.0

        # 7) total money loss uses combined yield-loss (avoid double count)
        total_loss_value_p50 = base_field_value_p50 * total_yield_loss_p50 if base_field_value_p50 is not None else None
        total_loss_value_p10 = base_field_value_p50 * total_yield_loss_p10 if base_field_value_p50 is not None else None
        total_loss_value_p90 = base_field_value_p50 * total_yield_loss_p90 if base_field_value_p50 is not None else None

        assumptions["combine_method"] = "Total yield-loss: 1 - Π(1 - f_i). Total money = base_field_value * total_yield_loss."
        assumptions["area_mode"] = cfg.area_mode
        assumptions["gsd_m_per_px"] = cfg.gsd_m_per_px
        assumptions["normalize_to"] = cfg.normalize_to
        assumptions["stats_key"] = str(stats_key)

        now_utc = datetime.now(timezone.utc).isoformat()

        res = EconomyResult(
            schema="agri_rag.economy_output.v2",
            time_utc=now_utc,
            latest_json_path=str(latest_json_path),
            field_stats_path=str(field_stats_path),
            run_id=run_id,
            tile_id=tile_id,
            tile_split=stats_split or tile_split,
            tile_year=tile_year,
            tile_source_path=tile_source_path,
            crop=cfg.crop,
            stage=cfg.stage,
            currency=currency,
            gsd_m_per_px=cfg.gsd_m_per_px if cfg.area_mode == "gsd" else None,
            area_mode=cfg.area_mode,
            normalize_to=cfg.normalize_to,
            items=items,
            base_field_value_p50=base_field_value_p50,
            total_loss_value_p50=total_loss_value_p50,
            total_loss_value_p10=total_loss_value_p10,
            total_loss_value_p90=total_loss_value_p90,
            total_yield_loss_frac_p50=total_yield_loss_p50,
            total_yield_loss_frac_p10=total_yield_loss_p10,
            total_yield_loss_frac_p90=total_yield_loss_p90,
            assumptions=assumptions,
            warnings=warnings,
        )
        return res
