"""Feature engineering for the numeric block.

Pipeline:
  1. Join listings with district-level features (MPE Mietpreise per Kreis)
  2. Engineer derived columns (area_per_room, building_age, size_bucket,
     is_luxurious / is_furnished / is_temporary from description)
  3. Provide a sklearn ColumnTransformer for fit/transform — guarantees
     train/inference parity. Persisted as one .joblib.
"""

from __future__ import annotations

import datetime
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from immopilot import config

logger = logging.getLogger(__name__)


# ─────────────────────── Feature schema ───────────────────────
# Continuous numeric features (incl. coordinates and engineered)
NUMERIC_FEATURES = [
    "area_m2",
    "rooms",
    "area_per_room",
    "lat",
    "lon",
    "year_built",
    "building_age",
    "years_since_renovation",
    # MPE district features (NaN for non-Zurich; imputed)
    "rent_median_chf_per_m2",
    "rent_mean_chf_per_m2",
    # Optional CV-derived (filled at inference time when photo provided)
    "condition_score",
    "kitchen_quality",
]

# Categorical features (one-hot encoded)
CATEGORICAL_FEATURES = [
    "location_kreis",  # Kreis 1-12 for Zürich rows, "other" otherwise
    "size_bucket",     # xs / s / m / l / xl
]

# Binary features from listings (Kaggle dataset)
BINARY_LISTINGS = [
    "is_new_building",
    "is_zurich",
    "has_balcony",
    "has_view",
    "has_elevator",
    "has_garage",
    "has_parking",
    "has_fireplace",
]

# Binary features derived from listing text
TEXT_DERIVED_BINARY = [
    "is_luxurious",
    "is_furnished",
    "is_temporary",
]


# ─────────────────────── Engineered columns ───────────────────────
CURRENT_YEAR = datetime.date.today().year


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Idempotent and tolerant of missing inputs."""
    df = df.copy()

    # area_per_room
    if {"area_m2", "rooms"}.issubset(df.columns):
        df["area_per_room"] = df["area_m2"] / df["rooms"].replace(0, np.nan)
    else:
        df["area_per_room"] = np.nan

    # size_bucket from area
    if "area_m2" in df.columns:
        df["size_bucket"] = pd.cut(
            df["area_m2"],
            bins=[0, 40, 65, 95, 130, np.inf],
            labels=["xs", "s", "m", "l", "xl"],
        ).astype("string")
    else:
        df["size_bucket"] = pd.NA

    # Building age & renovation freshness
    if "year_built" in df.columns:
        years = pd.to_numeric(df["year_built"], errors="coerce")
        df["building_age"] = (CURRENT_YEAR - years).clip(lower=0, upper=300)
    else:
        df["building_age"] = np.nan
    if "year_renovated" in df.columns:
        ren = pd.to_numeric(df["year_renovated"], errors="coerce")
        df["years_since_renovation"] = (CURRENT_YEAR - ren).clip(lower=0, upper=200)
    else:
        df["years_since_renovation"] = np.nan

    # Location grouping: Kreis 1-12 or "other"
    if "kreis" in df.columns:
        df["location_kreis"] = (
            df["kreis"].astype("string").fillna("other")
        )
    else:
        df["location_kreis"] = "other"

    # is_zurich derived if missing
    if "is_zurich" not in df.columns:
        if "kreis" in df.columns:
            df["is_zurich"] = df["kreis"].notna().astype("Int64")
        else:
            df["is_zurich"] = 0

    # Text-derived binary indicators
    desc = df.get("description", pd.Series([""] * len(df), index=df.index))
    desc = desc.fillna("").astype("string").str.lower()
    df["is_luxurious"] = desc.str.contains(
        r"luxuri|exklusiv|attika|penthouse|pent[\s-]?house", regex=True
    ).astype(int)
    df["is_furnished"] = desc.str.contains(
        r"möbliert|moebliert|furnished|meublé", regex=True
    ).astype(int)
    df["is_temporary"] = desc.str.contains(
        r"temporary|befristet|zwischen|temporaire", regex=True
    ).astype(int)

    # Default CV-derived features to neutral 0.5 if absent/empty.
    # When the CV block is wired in, real values from photos replace this.
    for col in ("condition_score", "kitchen_quality"):
        if col not in df.columns:
            df[col] = 0.5
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5)

    # Ensure all expected feature columns exist (NaN if absent)
    for col in NUMERIC_FEATURES + BINARY_LISTINGS:
        if col not in df.columns:
            df[col] = np.nan

    # Cast binary listings to float so sklearn can handle them
    for col in BINARY_LISTINGS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────── sklearn preprocessor ───────────────────────
def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary_listings = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )
    binary_text = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin_listings", binary_listings, BINARY_LISTINGS),
            ("bin_text", binary_text, TEXT_DERIVED_BINARY),
        ],
        verbose_feature_names_out=False,
    )


# ─────────────────────── Main pipeline ───────────────────────
def build_features(listings: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    """Join listings with district MPE features and engineer derived columns.

    Non-Zurich listings get NaN for district features; the imputer fills them.
    """
    if "kreis" in districts.columns:
        # Match dtype of left side to avoid silent join misses
        districts = districts.copy()
        districts["kreis"] = districts["kreis"].astype("Int64")
        df = listings.merge(districts, on="kreis", how="left")
    else:
        logger.warning("districts has no 'kreis' column; skipping join.")
        df = listings.copy()

    df = add_engineered_columns(df)
    df = df.dropna(subset=[config.NUMERIC_TARGET])
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    listings_p = config.PROCESSED_DIR / "listings.parquet"
    districts_p = config.PROCESSED_DIR / "zurich_districts.parquet"
    if not listings_p.exists() or not districts_p.exists():
        raise FileNotFoundError("Run `python -m immopilot.data.load_listings` and `load_zurich_open` first.")

    df = build_features(pd.read_parquet(listings_p), pd.read_parquet(districts_p))
    out = config.PROCESSED_DIR / "features.parquet"
    df.to_parquet(out)
    logger.info("Wrote %s shape=%s", out, df.shape)

    # Quick coverage stats
    feat_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_LISTINGS + TEXT_DERIVED_BINARY
    logger.info("Feature coverage (% non-null):")
    for c in feat_cols:
        if c in df.columns:
            cov = df[c].notna().mean() * 100
            logger.info("  %-30s %5.1f%%", c, cov)

    preprocessor = make_preprocessor()
    X_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_LISTINGS + TEXT_DERIVED_BINARY
    # Leakage fix: fit the preprocessor on the train+val pool only (everything
    # except the held-out test split), so imputation/scaling statistics never
    # see test rows. Replicates make_splits' first split (seed 42, stratified by
    # is_zurich) inline to avoid a circular import with models._common.
    from sklearn.model_selection import train_test_split
    _strat = df["is_zurich"] if "is_zurich" in df.columns else None
    df_pool, _df_test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=_strat
    )
    preprocessor.fit(df_pool[X_cols])
    joblib.dump(preprocessor, config.MODELS_DIR / "preprocessor.joblib")
    logger.info("Persisted preprocessor.joblib (%d output features)",
                preprocessor.transform(df[X_cols].head(1)).shape[1])


if __name__ == "__main__":
    main()
