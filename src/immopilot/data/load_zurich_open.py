"""Loader for City of Zurich open data — district-level features.

Pulls three CC0-licensed datasets from data.stadt-zuerich.ch:

  * MPE     — Mietpreiserhebung: official rent estimates per Stadtquartier ×
              Zimmerzahl. The single most informative district-level feature.
  * BEV     — Bevölkerung nach Stadtquartier, Herkunft, Geschlecht, Alter.
              Used to derive population, foreigner share per district.
  * WOHN    — Wohndichte (apartments + people per Stadtquartier).

Why direct URLs (and not the CKAN API): the slug+filename URLs are stable,
while CKAN API access from CI environments is sometimes blocked.

Usage:
    python -m immopilot.data.load_zurich_open
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from immopilot import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Stable direct-download URLs (verified 2026-05).
# Last update of MPE: 17.02.2026 — covers stichmonth April 2022 + 2024.
DATASETS = {
    "mpe": (
        "https://data.stadt-zuerich.ch/dataset/"
        "bau_whg_mpe_mietpreis_raum_zizahl_gn_jahr_od5161/"
        "download/BAU516OD5161.csv"
    ),
    "bevoelkerung": (
        "https://data.stadt-zuerich.ch/dataset/"
        "bev_bestand_jahr_quartier_alter_herkunft_geschlecht/"
        "download/BEV390OD3903.csv"
    ),
    "wohndichte": (
        "https://data.stadt-zuerich.ch/dataset/"
        "bau_best_whg_wfl_pers_ea_quartier_jahr_od6982/"
        "download/BAU698OD6982.csv"
    ),
}


# ─────────────────────── Download ───────────────────────


def _download(url: str, out: Path, force: bool = False) -> Path | None:
    if out.exists() and not force:
        logger.info("Cached: %s (delete to refresh)", out.name)
        return out
    logger.info("Downloading %s", url)
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "ImmoPilot/0.1"})
        r.raise_for_status()
    except requests.HTTPError as e:
        logger.warning("  ✗ skipping %s (%s)", out.name, e)
        return None
    out.write_bytes(r.content)
    logger.info("  ✓ %s (%d bytes)", out.name, len(r.content))
    return out


def download_all(force: bool = False) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for short, url in DATASETS.items():
        out = config.RAW_DIR / f"zurich_{short}.csv"
        result = _download(url, out, force=force)
        if result is not None:
            paths[short] = result
    if "mpe" not in paths:
        raise RuntimeError("MPE dataset is required but could not be downloaded.")
    return paths


# ─────────────────────── MPE aggregation ───────────────────────


def aggregate_mpe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce MPE to one row per Stadtkreis (1–12) with median rent CHF/m².

    Filters: Kreis 1–12 · alle Räume aggregiert · Quadratmeterpreis · netto ·
    Nicht-gemeinnützig (matches commercial-market listings on ImmoScout).
    """
    required = [
        "StichtagDatJahr",
        "GliederungLang",
        "ZimmerLang",
        "EinheitLang",
        "PreisartLang",
        "GemeinnuetzigLang",
        "qu50",
        "mean",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"MPE schema unexpected. Missing: {missing}")

    latest_year = df["StichtagDatJahr"].max()
    sub = df[
        (df["StichtagDatJahr"] == latest_year)
        & df["GliederungLang"].astype(str).str.match(r"^Kreis \d{1,2}$", na=False)
        & df["ZimmerLang"].astype(str).str.contains("und 4 Zimmer", na=False)
        & (df["EinheitLang"] == "Quadratmeter")
        & (df["PreisartLang"] == "netto")
        & (df["GemeinnuetzigLang"] == "Nicht gemeinnützig")
    ].copy()

    if sub.empty:
        raise ValueError(
            "MPE filter empty. Inspect the CSV schema and adjust filter values."
        )

    sub["kreis"] = sub["GliederungLang"].str.extract(r"^Kreis (\d+)$").astype(int)
    out = (
        sub.groupby("kreis", as_index=False)
        .agg(
            rent_median_chf_per_m2=("qu50", "median"),
            rent_mean_chf_per_m2=("mean", "median"),
        )
        .sort_values("kreis")
        .reset_index(drop=True)
    )
    out["mpe_year"] = int(latest_year)
    return out


# ─────────────────────── Public API ───────────────────────


def load_district_features() -> pd.DataFrame:
    """Return one row per Stadtquartier with district-level features.

    Currently includes only MPE-derived features. Expand with bevoelkerung
    + wohndichte aggregations once you've inspected those CSVs.
    """
    paths = download_all()
    mpe = pd.read_csv(paths["mpe"])
    return aggregate_mpe(mpe)


if __name__ == "__main__":
    df = load_district_features()
    out = config.PROCESSED_DIR / "zurich_districts.parquet"
    df.to_parquet(out)
    logger.info("Wrote %s shape=%s", out, df.shape)
    logger.info("Preview:\n%s", df.head(25).to_string())
