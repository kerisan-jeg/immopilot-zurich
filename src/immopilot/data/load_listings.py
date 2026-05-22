"""Loader for Swiss apartment-rental listings.

Strategy: train on the full Swiss listings, but enrich Zurich rows with
Stadt-Zürich features (MPE rent reference per Kreis). Non-Zurich rows
have NaN in those Zurich-specific columns; the imputer handles that.

Source:
    Kaggle: fredeys/immoscout24-ch-switzerland-rental-property-dataset
    Place CSV at: data/raw/listings.csv

After loading, output is normalized to:

    rent_chf · area_m2 · rooms · plz · city · address · description ·
    lat · lon · year_built · year_renovated · is_new_building ·
    has_balcony · has_view · has_elevator · has_garage · has_parking ·
    has_fireplace · kreis (Zürich only) · is_zurich

Inspect a fresh CSV before training:

    python -m immopilot.data.load_listings inspect path/to/file.csv
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

from immopilot import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


CANONICAL_COLUMNS = [
    # Target + core
    "rent_chf",
    "area_m2",
    "rooms",
    # Location
    "plz",
    "city",
    "address",
    "lat",
    "lon",
    "kreis",       # only set for Stadt-Zürich rows; NaN otherwise
    # Temporal
    "year_built",
    "year_renovated",
    "is_new_building",
    # Amenities (binary)
    "has_balcony",
    "has_view",
    "has_elevator",
    "has_garage",
    "has_parking",
    "has_fireplace",
    # Text
    "description",
]


# ─────────────────────── Column auto-detection ───────────────────────
COLUMN_PATTERNS: dict[str, list[str]] = {
    "rent_chf": [r"^rentnet$", r"^rent$", r"^price$", r"^miete$", r"^bruttomiete$"],
    "area_m2": [r"^livingspace$", r"^living.?space$", r"^surface.?area$", r"^area$",
                r"^flaeche$", r"^fläche$", r"^size$"],
    "rooms": [r"^numberofrooms$", r"^numrooms$", r"^rooms$", r"^zimmer$",
              r"^anzahl.?zimmer$", r"^number.?of.?rooms$"],
    "address": [r"^address$", r"^adresse$", r"^location$"],
    "city": [r"^city$", r"^stadt$", r"^ort$"],
    "description": [r"^description$", r"^beschreibung$", r"^title$", r"^text$"],
    "plz": [r"^postcode$", r"^zipcode$", r"^plz$", r"^zip$", r"^postal.?code$"],
    "lat": [r"^lat$", r"^latitude$"],
    "lon": [r"^lon$", r"^lng$", r"^longitude$"],
    "year_built": [r"^year.?built$", r"^baujahr$"],
    "year_renovated": [r"^year.?last.?renovated$", r"^renovat"],
    "is_new_building": [r"^is.?new.?building$"],
    "has_balcony": [r"^has.?balcony$"],
    "has_view": [r"^has.?nice.?view$", r"^has.?view$"],
    "has_elevator": [r"^has.?elevator$", r"^has.?lift$"],
    "has_garage": [r"^has.?garage$"],
    "has_parking": [r"^has.?parking$"],
    "has_fireplace": [r"^has.?fireplace$"],
}


def _detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return mapping {canonical → actual_column_name} based on regex match."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canonical, patterns in COLUMN_PATTERNS.items():
        for pat in patterns:
            for low, orig in cols_lower.items():
                if re.search(pat, low):
                    mapping[canonical] = orig
                    break
            if canonical in mapping:
                break
    return mapping


# ─────────────────────── PLZ → Kreis mapping ───────────────────────
KREIS_BY_PLZ: dict[int, int] = {
    8001: 1, 8002: 2, 8003: 3, 8004: 4, 8005: 5,
    8006: 6, 8008: 8, 8032: 7, 8037: 5, 8038: 2,
    8041: 2, 8044: 7, 8045: 3, 8046: 11, 8047: 9,
    8048: 9, 8049: 10, 8050: 11, 8051: 11, 8052: 11,
    8053: 8, 8055: 2, 8057: 10, 8064: 9,
}
ZURICH_PLZ_RANGE = range(8000, 8100)


# ─────────────────────── Cleaning helpers ───────────────────────
def _coerce_numeric(s: pd.Series) -> pd.Series:
    """Strip currency markers, thousand separators, units; coerce to float."""
    if s.dtype.kind in "fi":
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype("string")
        .str.replace(r"[\u2019']", "", regex=True)
        .str.replace(r"[^\d.,\-]", "", regex=True)
        .str.replace(r"[.,]-+$", "", regex=True)
        .str.replace(r"^-+|-+$", "", regex=True)
        .str.replace(",", ".", regex=False)
        .str.replace(r"\.(?=.*\.)", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _coerce_bool(s: pd.Series) -> pd.Series:
    """Map various truthy/falsy representations to nullable Int64 (1/0/NA)."""
    if s.dtype == bool:
        return s.astype("Int64")
    s_str = s.astype("string").str.lower().str.strip()
    mapped = s_str.map(
        {"true": 1, "1": 1, "yes": 1, "ja": 1,
         "false": 0, "0": 0, "no": 0, "nein": 0}
    )
    return mapped.astype("Int64")


def _extract_plz(address: pd.Series) -> pd.Series:
    return (
        address.astype("string")
        .str.extract(r"\b(\d{4})\b", expand=False)
        .astype("Int64")
    )


# ─────────────────────── Public API ───────────────────────
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map source columns to canonical schema. Missing → NaN."""
    mapping = _detect_columns(df)
    logger.info("Detected schema mapping: %s", mapping)

    out = pd.DataFrame(index=df.index)
    for canonical in CANONICAL_COLUMNS:
        if canonical == "kreis":
            out[canonical] = pd.NA
            continue
        if canonical in mapping:
            out[canonical] = df[mapping[canonical]]
        else:
            out[canonical] = pd.NA

    # Type coercion
    for col in ["rent_chf", "area_m2", "rooms", "lat", "lon",
                "year_built", "year_renovated"]:
        out[col] = _coerce_numeric(out[col])
    out["plz"] = pd.to_numeric(out["plz"], errors="coerce").astype("Int64")
    for col in ["is_new_building", "has_balcony", "has_view",
                "has_elevator", "has_garage", "has_parking", "has_fireplace"]:
        out[col] = _coerce_bool(out[col])

    # Recover PLZ from address if mostly missing
    if out["plz"].isna().mean() > 0.5:
        out["plz"] = _extract_plz(out["address"])
    return out


def filter_zurich(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Stadt-Zürich rows. Kept as a helper for analysis / EDA."""
    plz_mask = df["plz"].isin(list(ZURICH_PLZ_RANGE))
    addr_mask = df["address"].astype("string").fillna("").str.contains(
        r"\bZ[üu]rich\b", case=False, regex=True
    )
    keep = plz_mask | addr_mask
    n_before = len(df)
    out = df[keep].copy()
    logger.info("Zurich filter: %d → %d rows (%.1f%% kept)",
                n_before, len(out), 100 * len(out) / max(n_before, 1))
    return out


def assign_kreis(df: pd.DataFrame) -> pd.DataFrame:
    """Add Kreis 1-12 for Stadt-Zürich rows; NaN for the rest.
    Also adds is_zurich boolean."""
    df = df.copy()
    df["kreis"] = df["plz"].map(KREIS_BY_PLZ).astype("Int64")
    df["is_zurich"] = df["kreis"].notna()
    n_zh = df["is_zurich"].sum()
    logger.info("Kreis assignment: %d/%d rows are Stadt Zürich (%.1f%%)",
                n_zh, len(df), 100 * n_zh / max(len(df), 1))
    return df


def basic_outlier_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with implausible rent/area/rooms."""
    n_before = len(df)
    out = df[
        df["rent_chf"].between(500, 12000)
        & df["area_m2"].between(10, 400)
        & df["rooms"].between(0.5, 10)
    ].copy()
    logger.info("Outlier filter: %d → %d rows (%d dropped)",
                n_before, len(out), n_before - len(out))
    return out


def load_listings() -> pd.DataFrame:
    """Main entry point: load + normalize + tag Zurich + filter outliers.
    Returns Switzerland-wide listings (not just Zurich)."""
    source = os.getenv("LISTINGS_SOURCE", "kaggle")
    if source == "kaggle":
        path = config.RAW_DIR / "listings.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected listings snapshot at {path}.\n"
                f"Download from https://www.kaggle.com/datasets/"
                f"fredeys/immoscout24-ch-switzerland-rental-property-dataset"
            )
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif source == "scrape":
        from scripts.scrape_homegate import scrape_listings  # type: ignore[import-not-found]
        df = scrape_listings()
    else:
        raise ValueError(f"Unknown LISTINGS_SOURCE={source}")

    df = normalize(df)
    df = assign_kreis(df)             # tag Zurich rows, leave others
    df = basic_outlier_filter(df)
    return df


# ─────────────────────── CLI ───────────────────────
def _inspect(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, nrows=5, encoding="utf-8-sig")
    with open(csv_path, encoding="utf-8-sig") as _fh:
        n_rows = sum(1 for _ in _fh) - 1
    print(f"\nFile: {csv_path}  ({n_rows} rows)")
    print(f"\nColumns ({len(df.columns)}):")
    for c in df.columns:
        sample = str(df[c].iloc[0])[:60]
        print(f"  {c:30}  e.g.  {sample}")
    mapping = _detect_columns(df)
    print("\nDetected canonical mapping:")
    for canonical in CANONICAL_COLUMNS:
        if canonical == "kreis":
            continue
        actual = mapping.get(canonical, "❌ NOT FOUND")
        print(f"  {canonical:18} ← {actual}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        if len(sys.argv) < 3:
            print("Usage: python -m immopilot.data.load_listings inspect <csv>")
            sys.exit(1)
        _inspect(Path(sys.argv[2]))
        return

    df = load_listings()
    out = config.PROCESSED_DIR / "listings.parquet"
    df.to_parquet(out)
    logger.info("Wrote %s shape=%s", out, df.shape)

    n_zh = df["is_zurich"].sum()
    logger.info(
        "Per-Kreis row count (Stadt Zürich, %d total):\n%s",
        n_zh,
        df[df["is_zurich"]]["kreis"].value_counts().sort_index().to_string(),
    )
    logger.info(
        "Top-10 cities (rest of CH):\n%s",
        df[~df["is_zurich"]]["city"].value_counts().head(10).to_string(),
    )


if __name__ == "__main__":
    main()
