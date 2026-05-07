"""Smoke tests — minimum sanity. Run on every CI push.

These do NOT require trained models or API keys. They verify that the package
imports cleanly, paths exist, and the feature engineering doesn't throw on a
toy DataFrame. Heavier integration tests live in `test_pipeline.py` and run
locally on demand.
"""

from __future__ import annotations

import pandas as pd

from immopilot import config


def test_config_paths_exist():
    for p in (config.DATA_DIR, config.MODELS_DIR, config.DOCS_DIR):
        assert p.exists(), f"Missing required path: {p}"


def test_seeding_is_deterministic():
    config.set_global_seed(42)
    import random

    a = random.random()
    config.set_global_seed(42)
    b = random.random()
    assert a == b


def test_feature_engineering_on_minimal_input():
    from immopilot.features.build_features import add_engineered_columns

    df = pd.DataFrame(
        {
            "rent_chf": [2500],
            "area_m2": [60],
            "rooms": [3.0],
            "kreis": [6],
            "description": ["Schöne möblierte Wohnung mit Seesicht"],
        }
    )
    out = add_engineered_columns(df)
    assert "area_per_room" in out.columns
    assert out.loc[0, "is_furnished"] == 1
    assert out.loc[0, "is_luxurious"] == 0
    assert out.loc[0, "is_temporary"] == 0


def test_chunking_logic():
    from immopilot.nlp.text_utils import chunk_text

    text = "word " * 1500
    chunks = chunk_text(text, size=512, overlap=64)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 512 for c in chunks)


def test_listings_schema_detection():
    """Auto-detection should map both English (ImmoScout) and German column names."""
    from immopilot.data.load_listings import _detect_columns

    df_en = pd.DataFrame(
        columns=["Id", "SurfaceArea", "NumRooms", "Rent", "Address", "Description", "Link", "ZipCode"]
    )
    m_en = _detect_columns(df_en)
    assert m_en["rent_chf"] == "Rent"
    assert m_en["area_m2"] == "SurfaceArea"
    assert m_en["rooms"] == "NumRooms"
    assert m_en["plz"] == "ZipCode"

    df_de = pd.DataFrame(columns=["Miete", "Fläche", "Zimmer", "Adresse", "PLZ"])
    m_de = _detect_columns(df_de)
    assert m_de["rent_chf"] == "Miete"
    assert m_de["area_m2"] == "Fläche"
    assert m_de["plz"] == "PLZ"


def test_listings_swiss_currency_parsing():
    """Swiss notations should parse correctly."""
    from immopilot.data.load_listings import _coerce_numeric

    s = pd.Series(["CHF 2400.-", "3'200", "2\u2019400.50", "1100 CHF"])
    out = _coerce_numeric(s)
    assert out.iloc[0] == 2400.0
    assert out.iloc[1] == 3200.0
    assert out.iloc[2] == 2400.5
    assert out.iloc[3] == 1100.0


def test_listings_zurich_filter_and_kreis_assignment():
    """End-to-end: only Zurich rows kept, Kreis correctly assigned from PLZ."""
    from immopilot.data.load_listings import (
        assign_kreis,
        basic_outlier_filter,
        filter_zurich,
        normalize,
    )

    raw = pd.DataFrame(
        {
            "Rent": [2400, 3200, 1100, 5500],
            "SurfaceArea": [60, 80, 35, 95],
            "NumRooms": [2.5, 3.5, 1.5, 4.0],
            "Address": [
                "Bahnhofstr. 1, 8001 Zürich",
                "Forchstr. 22, 8008 Zürich",
                "Rue de la Gare 3, 1003 Lausanne",
                "Limmatquai 5, 8001 Zürich",
            ],
            "Description": ["", "", "", ""],
            "Link": ["", "", "", ""],
            "ZipCode": [8001, 8008, 1003, 8001],
        }
    )
    df = basic_outlier_filter(assign_kreis(filter_zurich(normalize(raw))))
    assert len(df) == 3, f"Expected 3 Zurich rows, got {len(df)}"
    assert set(df["kreis"].dropna().tolist()) == {1, 8}
