"""Drift monitor — Evidently data drift, current traffic vs reference (Step 9).

Run:
    python -m src.monitoring.drift --current data/val            # compare a folder of frames
    python -m src.monitoring.drift --current outputs/features.jsonl
    python -m src.monitoring.drift --url http://<ip>:8000/monitor/features
    python -m src.monitoring.drift --rebuild-reference           # recompute the baseline

Exit codes (the same contract ``src.registry.gate`` uses, on purpose):

    0  OK     — current batch is statistically consistent with the reference
    1  DRIFT  — the drifted-feature share exceeded ``monitor.drift_share``
    2  ERROR  — missing data / too few rows / broken config (an operational
                problem, NOT a verdict — distinguishing 1 from 2 is what keeps
                "the data moved" separate from "the monitor crashed")

What "drift" means here, plainly: the reference is the feature distribution
the model was *trained* on. Every column is compared with a two-sample K-S
test (p < 0.05 => that column drifted); the batch is flagged DRIFT when more
than ``drift_share`` of the monitored columns drifted. If serving inputs no
longer resemble training inputs, yesterday's gate metrics no longer describe
today's traffic — which is the signal to look hard at the data (and, in
Step 10, to gate a retrain).

The verdict is printed AND written to ``outputs/drift_report.json`` plus a
full HTML report, so a human can open the evidence behind the exit code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yaml
from PIL import Image

from src.monitoring.features import FEATURE_COLUMNS, extract_features, read_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

EXIT_OK, EXIT_DRIFT, EXIT_ERROR = 0, 1, 2

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class DriftError(Exception):
    """Operational failure (missing data, too few rows) — exit 2, not a verdict."""


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def extract_batch(paths: list[Path]) -> pd.DataFrame:
    """One row per image file."""
    rows = [extract_features(Image.open(p).convert("RGB")) for p in paths]
    return pd.DataFrame(rows)


def build_reference(cfg: dict, source_dir: Path, cache: Path | None,
                    rebuild: bool = False) -> pd.DataFrame:
    """Reference features for the training split, optionally cached as parquet.

    The cache exists because the reference is *fixed*: once the baseline is
    measured, re-extracting 240 images on every monitor run is pure waste.
    ``cache=None`` (ad-hoc ``--reference-dir``) skips caching entirely — a
    one-off baseline must never poison the real one.
    """
    if cache is not None and cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    if not source_dir.is_dir():
        raise DriftError(f"reference split not found: {source_dir}\n"
                         "Build the dataset first:  python -m src.data.make_gold")
    paths = sorted(p for p in source_dir.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise DriftError(f"no images in {source_dir}")
    df = extract_batch(paths)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df


def load_current(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Current batch from --url (the live API), a .jsonl log, or a folder."""
    if args.url:
        try:
            with urlopen(args.url, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise DriftError(f"could not fetch {args.url}: {exc}") from exc
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise DriftError(f"unexpected response shape from {args.url}")
        if not rows:
            raise DriftError(f"the API at {args.url} has captured no rows yet "
                             "(or capture is disabled)")
        return pd.DataFrame(rows), args.url

    path = Path(args.current)
    if not path.exists():
        raise DriftError(f"--current not found: {path}")
    if path.is_dir():
        paths = sorted(p for p in path.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise DriftError(f"no images in {path}")
        return extract_batch(paths), str(path)
    if path.suffix.lower() == ".jsonl":
        rows = read_rows(path)
        if not rows:
            raise DriftError(f"no readable rows in {path}")
        return pd.DataFrame(rows), str(path)
    raise DriftError(f"--current must be an image folder or a .jsonl log, got: {path}")


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def compare(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict,
            source_ref: str, source_cur: str) -> tuple[dict, object]:
    """Run Evidently, return (report_dict, evidently_run_for_html)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    wanted = cfg.get("features") or FEATURE_COLUMNS
    usable = [c for c in wanted
              if c in reference.columns and c in current.columns]
    dropped_missing = [c for c in wanted if c not in usable]

    # A column that is constant in the reference has no distribution to
    # compare against (every synthetic frame is 256x192, e.g.): K-S on it is
    # meaningless, so drop it and say so in the report instead of emitting a
    # false alarm — or a false all-clear.
    constant = [c for c in usable if reference[c].nunique(dropna=True) <= 1]
    usable = [c for c in usable if c not in constant]
    if not usable:
        raise DriftError("no comparable feature columns — every monitored "
                         "column is constant or missing on both sides")

    ref = reference[usable].dropna()
    cur = current[usable].dropna()

    min_rows = int(cfg.get("min_rows", 10))
    if len(ref) < min_rows:
        raise DriftError(f"reference has only {len(ref)} usable rows "
                         f"(min_rows={min_rows}) — rebuild the baseline")
    if len(cur) < min_rows:
        raise DriftError(f"current batch has only {len(cur)} usable rows "
                         f"(min_rows={min_rows}) — too few to judge; "
                         "gather more traffic, this is not a verdict")

    threshold = float(cfg.get("stat_test_threshold", 0.05))
    drift_share = float(cfg.get("drift_share", 0.5))

    report = Report(metrics=[DataDriftPreset(
        columns=usable,
        drift_share=drift_share,
        threshold=threshold,
    )])
    run = report.run(reference_data=ref, current_data=cur)
    payload = json.loads(run.json())

    columns, count, share = _parse_run(payload, threshold)
    verdict = "DRIFT" if share > drift_share else "OK"

    report_dict = {
        "verdict": verdict,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": {
            "source": source_ref,
            "n_rows": len(ref),
            "columns_used": usable,
            "dropped_constant": constant,
            "dropped_missing": dropped_missing,
        },
        "current": {
            "source": source_cur,
            "n_rows": len(cur),
            "rows_read": len(current),
            "rows_dropped_na": int(len(current) - len(cur)),
        },
        "drift": {
            "drifted_count": int(count),
            "drifted_share": float(share),
            "drift_share_limit": drift_share,
            "stat_test_threshold": threshold,
            "columns": columns,
        },
    }
    return report_dict, run


def _parse_run(payload: dict, default_threshold: float
               ) -> tuple[list[dict], int, float]:
    """Pull per-column p-values and the drifted count out of Evidently's JSON."""
    columns: list[dict] = []
    count: int | None = None
    share: float | None = None

    for metric in payload.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        value = metric.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            count = int(value.get("count", 0))
            share = float(value.get("share", 0.0))
        elif name.startswith("ValueDrift"):
            conf = metric.get("config", {})
            column = conf.get("column")
            if not column:
                continue
            thr = float(conf.get("threshold", default_threshold))
            p_value: float | None
            drifted: bool
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                p_value = float(value)
                if math.isnan(p_value):  # constant/undefined comparison
                    p_value = None
                    drifted = False
                else:
                    drifted = p_value < thr
            else:  # a method whose value is not a p-value: record, don't guess
                p_value = None
                drifted = False
            columns.append({"column": column, "p_value": p_value,
                            "threshold": thr, "drifted": drifted})

    columns.sort(key=lambda c: (not c["drifted"], c["p_value"] is None,
                                c["p_value"] if c["p_value"] is not None else 1.0))
    if count is None:  # older/newer Evidently without the aggregate metric
        count = sum(1 for c in columns if c["drifted"])
        share = (count / len(columns)) if columns else 0.0
    return columns, count, share or 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise DriftError(f"config not found: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    monitor = cfg.get("monitor")
    if not isinstance(monitor, dict):
        raise DriftError(f"{p} has no `monitor:` section (Step 9 config)")
    return monitor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Data-drift monitor: current traffic vs training reference")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--current", default=None,
                        help="image folder OR .jsonl feature log to compare")
    parser.add_argument("--url", default=None,
                        help="fetch current rows from a live API "
                             "(/monitor/features) instead of a local file")
    parser.add_argument("--reference-dir", default=None,
                        help="override the reference image folder "
                             "(default: <paths.raw_images>/<monitor.reference_split>)")
    parser.add_argument("--reference-cache", default=None,
                        help="override the reference feature cache path")
    parser.add_argument("--rebuild-reference", action="store_true",
                        help="ignore the cached reference features")
    parser.add_argument("--report", default=None, help="JSON report path")
    parser.add_argument("--html", default=None, help="HTML report path "
                                                     "('' to skip)")
    parser.add_argument("--json", action="store_true",
                        help="print the full report as JSON")
    args = parser.parse_args(argv)

    try:
        if not args.current and not args.url:
            print("[DRIFT] ERROR — nothing to compare: pass --current or --url")
            return EXIT_ERROR

        m = _load_config(args.config)
        paths = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["paths"]

        ref_dir = (Path(args.reference_dir) if args.reference_dir
                   else PROJECT_ROOT / paths["raw_images"] / m.get("reference_split", "train"))
        if args.reference_cache:
            cache: Path | None = Path(args.reference_cache)
        elif args.reference_dir:
            cache = None  # ad-hoc baseline: never read or poison the real cache
        else:
            cache = PROJECT_ROOT / m.get(
                "reference_features",
                "data/processed/reference_features.parquet")
        reference = build_reference(m, ref_dir, cache, rebuild=args.rebuild_reference)

        current, cur_source = load_current(args)
        report, run = compare(reference, current, m,
                              source_ref=str(ref_dir), source_cur=cur_source)

        # persist
        report_path = Path(args.report) if args.report else PROJECT_ROOT / m["report"]
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        html_arg = args.html if args.html is not None else m.get("html")
        if html_arg:
            html_path = Path(html_arg)
            if not html_path.is_absolute():
                html_path = PROJECT_ROOT / html_path
            html_path.parent.mkdir(parents=True, exist_ok=True)
            run.save_html(str(html_path))

        d = report["drift"]
        if args.json:
            print(json.dumps(report, indent=2))
        print(f"[DRIFT] {report['verdict']} — {d['drifted_count']}/"
              f"{len(d['columns'])} features drifted "
              f"(share {d['drifted_share']:.3f} vs limit {d['drift_share_limit']})")
        for col in d["columns"]:
            p = col["p_value"]
            p_txt = "n/a" if p is None else f"{p:.3g}"
            mark = "DRIFT" if col["drifted"] else "ok   "
            print(f"    {mark}  {col['column']:<14} p={p_txt}")
        print(f"    ref: {report['reference']['n_rows']} rows from "
              f"{report['reference']['source']}")
        print(f"    cur: {report['current']['n_rows']} rows from {cur_source}")
        print(f"  report: {report_path}")
        return EXIT_DRIFT if report["verdict"] == "DRIFT" else EXIT_OK

    except DriftError as exc:
        print(f"[DRIFT] ERROR — {exc}")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 — a crash must not look like a verdict
        print(f"[DRIFT] ERROR — {type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
