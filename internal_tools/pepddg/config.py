from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise ValueError(f"Missing required config field: {ctx}.{key}")
    return d[key]


def _to_str_list(raw: Any, *, field_name: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out
    raise ValueError(f"Field `{field_name}` must be a list of strings.")


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


DEFAULT_ANCHOR = "__pepddg_default_anchor__"
DEFAULT_ANCHOR_PREFERENCE = ["__pepddg_default_anchor_preference__"]
CAL_ONLY_SCORE_COLUMNS = {"rankscore_3view_v19_strict3"}
CAL_ONLY_ANCHOR_COLUMNS = {"rankscore_3view_v17_mainboost", "rankscore_3view_v17_2_oodguard"}
ZS_ONLY_SCORE_COLUMNS = {"rankscore_pepddg_zs"}


@dataclass(frozen=True)
class V19SanitizedWeights:
    """Frozen weights for the PepDDG v19 sanitized baseline."""

    w_neg: float = -0.37744658504381534
    w_bind: float = -0.3740327505870402
    w_disp_pen: float = -0.017201058980289563
    w_m: float = 0.9446730912058925


def _weights_match(a: V19SanitizedWeights, b: V19SanitizedWeights, tol: float = 1e-12) -> bool:
    return (
        abs(float(a.w_neg) - float(b.w_neg)) <= tol
        and abs(float(a.w_bind) - float(b.w_bind)) <= tol
        and abs(float(a.w_disp_pen) - float(b.w_disp_pen)) <= tol
        and abs(float(a.w_m) - float(b.w_m)) <= tol
    )


@dataclass(frozen=True)
class PepDDGConfig:
    """Runtime configuration for the PepDDG internal tool."""

    input_csv: str
    output_csv: str
    output_dir: str | None = None
    mode: str = "zs"
    score_column: str | None = None
    anchor_column: str | None = DEFAULT_ANCHOR
    anchor_preference: list[str] | None = field(default_factory=lambda: list(DEFAULT_ANCHOR_PREFERENCE))
    strict_anchor: bool = False
    baseline_column_for_delta: str | None = "rankscore_3view_base"
    write_auxiliary_columns: bool = True
    write_run_manifest: bool = True
    write_policy_audit: bool = True
    write_gate_metrics: bool = True
    variant_name: str | None = None
    seed: int = 20260302
    banned_patterns: list[str] = field(
        default_factory=lambda: [
            "foldx",
            "rosetta",
            "cartddg",
            "stabddg",
            "esm2",
            "esmif",
            "esm3",
            "saprot",
            "diffaffinity",
        ]
    )
    allowed_output_roots: list[str] = field(default_factory=list)
    weights: V19SanitizedWeights = field(default_factory=V19SanitizedWeights)

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        object.__setattr__(self, "mode", mode)
        if self.score_column is None:
            object.__setattr__(
                self,
                "score_column",
                "rankscore_3view_v19_strict3" if mode == "cal" else "rankscore_pepddg_zs",
            )
        if self.variant_name is None:
            object.__setattr__(self, "variant_name", "pepddg_cal_v19" if mode == "cal" else "pepddg_zs")
        if self.anchor_column == DEFAULT_ANCHOR:
            object.__setattr__(
                self,
                "anchor_column",
                "rankscore_3view_v17_mainboost" if mode == "cal" else "rankscore_3view_base",
            )
        if self.anchor_preference == DEFAULT_ANCHOR_PREFERENCE:
            object.__setattr__(
                self,
                "anchor_preference",
                (
                    ["rankscore_3view_v17_mainboost", "rankscore_3view_v17_2_oodguard", "rankscore_3view_base"]
                    if mode == "cal"
                    else ["rankscore_3view_base"]
                ),
            )
        elif self.anchor_preference is None:
            object.__setattr__(self, "anchor_preference", [])

    def effective_anchor_column(self) -> str | None:
        return self.anchor_column

    def effective_anchor_preference(self) -> list[str]:
        return list(self.anchor_preference or [])

    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        return Path(self.output_csv).parent

    def validate(self) -> None:
        mode = self.mode.strip().lower()
        if mode not in {"zs", "cal"}:
            raise ValueError("`mode` must be one of: zs, cal.")
        anchor_column = self.effective_anchor_column()
        anchor_preference = self.effective_anchor_preference()
        if self.strict_anchor and not anchor_column:
            raise ValueError("`strict_anchor=true` requires `anchor_column` to be set.")
        if mode == "cal" and not anchor_preference and not anchor_column:
            raise ValueError("At least one anchor source is required.")
        if not str(self.score_column).strip():
            raise ValueError("`score_column` cannot be empty.")
        if mode == "zs":
            cal_markers: list[str] = []
            if str(self.score_column) in CAL_ONLY_SCORE_COLUMNS:
                cal_markers.append(f"score_column={self.score_column}")
            if anchor_column in CAL_ONLY_ANCHOR_COLUMNS:
                cal_markers.append(f"anchor_column={anchor_column}")
            for anchor in anchor_preference:
                if anchor in CAL_ONLY_ANCHOR_COLUMNS:
                    cal_markers.append(f"anchor_preference contains {anchor}")
            if cal_markers:
                raise ValueError(
                    "mode=zs cannot use Cal-only score or anchor markers; "
                    "set `mode: cal` for frozen v19 calibrated scoring. Found "
                    + ", ".join(cal_markers)
                )
        if mode == "cal":
            if str(self.score_column) in ZS_ONLY_SCORE_COLUMNS:
                raise ValueError(
                    "mode=cal cannot use ZS-only score labels; "
                    "use `rankscore_3view_v19_strict3` for frozen v19 Cal scoring. "
                    f"Found score_column={self.score_column}"
                )
            frozen = V19SanitizedWeights()
            if not _weights_match(self.weights, frozen):
                raise ValueError(
                    "mode=cal requires frozen default weights. "
                    "Use a different variant_name for custom weight exploration."
                )

        roots = [Path(p).resolve() for p in self.allowed_output_roots]
        if roots:
            out_csv = Path(self.output_csv).resolve()
            out_dir = self.resolved_output_dir().resolve()
            for p in [out_csv, out_dir]:
                if not any(_is_subpath(p, r) for r in roots):
                    raise ValueError(
                        f"Output path `{p}` is outside allowed roots: "
                        + ", ".join(str(r) for r in roots)
                    )


def load_config(path: str) -> PepDDGConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    mode_was_explicit = "mode" in raw
    raw_mode = str(raw.get("mode") or "zs").strip().lower()
    if not mode_was_explicit and raw_mode == "zs":
        cal_markers: list[str] = []
        score_column = str(raw.get("score_column") or "")
        if score_column in CAL_ONLY_SCORE_COLUMNS:
            cal_markers.append("score_column=rankscore_3view_v19_strict3")
        anchor_column = str(raw.get("anchor_column") or "")
        if anchor_column in CAL_ONLY_ANCHOR_COLUMNS:
            cal_markers.append(f"anchor_column={anchor_column}")
        for anchor in _to_str_list(raw.get("anchor_preference"), field_name="anchor_preference"):
            if anchor in CAL_ONLY_ANCHOR_COLUMNS:
                cal_markers.append(f"anchor_preference contains {anchor}")
        if cal_markers:
            raise ValueError(
                "Legacy Cal-style config must set `mode: cal`; found "
                + ", ".join(cal_markers)
            )

    weights_raw = raw.get("weights") or {}
    weights = V19SanitizedWeights(
        w_neg=float(weights_raw.get("w_neg", V19SanitizedWeights.w_neg)),
        w_bind=float(weights_raw.get("w_bind", V19SanitizedWeights.w_bind)),
        w_disp_pen=float(weights_raw.get("w_disp_pen", V19SanitizedWeights.w_disp_pen)),
        w_m=float(weights_raw.get("w_m", V19SanitizedWeights.w_m)),
    )

    cfg = PepDDGConfig(
        input_csv=str(_require(raw, "input_csv", "root")),
        output_csv=str(_require(raw, "output_csv", "root")),
        output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
        mode=raw_mode,
        score_column=str(raw["score_column"]) if raw.get("score_column") is not None else None,
        anchor_column=(
            str(raw["anchor_column"]).strip()
            if raw.get("anchor_column") is not None
            else (None if "anchor_column" in raw else DEFAULT_ANCHOR)
        ),
        anchor_preference=(
            _to_str_list(raw.get("anchor_preference"), field_name="anchor_preference")
            if "anchor_preference" in raw
            else list(DEFAULT_ANCHOR_PREFERENCE)
        ),
        strict_anchor=bool(raw.get("strict_anchor", False)),
        baseline_column_for_delta=(
            str(raw["baseline_column_for_delta"]).strip()
            if raw.get("baseline_column_for_delta") is not None
            else "rankscore_3view_base"
        ),
        write_auxiliary_columns=bool(raw.get("write_auxiliary_columns", True)),
        write_run_manifest=bool(raw.get("write_run_manifest", True)),
        write_policy_audit=bool(raw.get("write_policy_audit", True)),
        write_gate_metrics=bool(raw.get("write_gate_metrics", True)),
        variant_name=str(raw["variant_name"]) if raw.get("variant_name") is not None else None,
        seed=int(raw.get("seed") or 20260302),
        banned_patterns=_to_str_list(raw.get("banned_patterns"), field_name="banned_patterns")
        or PepDDGConfig.__dataclass_fields__["banned_patterns"].default_factory(),  # type: ignore[attr-defined]
        allowed_output_roots=_to_str_list(
            raw.get("allowed_output_roots"), field_name="allowed_output_roots"
        ),
        weights=weights,
    )
    cfg.validate()
    return cfg
