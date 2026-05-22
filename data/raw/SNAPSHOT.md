# Data Snapshot Record

Exact source and version of every dataset used, for reproducibility.

## listings.csv (numeric / ML block)
- **Source**: Kaggle — `fredeys/immoscout24-ch-switzerland-rental-property-dataset`
  (Switzerland-wide ImmoScout24 rental listings). This is **not** the course-restricted
  "apartments canton of Zurich" dataset.
- **Snapshot date**: 2026-05 (see commit history for exact pull)
- **Rows after cleaning**: 664 (of which 27 are in the city of Zurich)
- **License**: per the Kaggle dataset page
- **Notes**: filtered to Zurich-relevant rows; outlier filtering and Kreis assignment
  applied in `src/immopilot/data/load_listings.py`.

## zurich_districts (district features / calibration)
- **Source**: Statistik Stadt Zürich — Mietpreiserhebung (MPE) 2024, via
  data.stadt-zuerich.ch (median + mean net rent CHF/m² per Kreis).
- **Snapshot date**: 2024 edition, pulled 2026-05
- **License**: CC BY (Stadt Zürich open data)
- **Notes**: aggregated to one row per Kreis in `load_zurich_open.py::aggregate_mpe`.
  BEV/Wohndichte are downloaded by the same module but **not** joined into the feature
  set (see DOCUMENTATION.md 2A.1).

## rag_corpus/ (NLP block)
- **Source**: Stadt Zürich district pages + Wikipedia (Zurich Kreis articles).
- **Snapshot date**: 2026-05
- **License**: CC BY-SA / public domain (verify per page)
- **Files**: 13 Markdown documents (one per district + topic), with YAML frontmatter
  (title, url, district) used as citation metadata.

## images/ (CV bonus block)
- **Sources**: self-collected interior photos + CC-licensed apartment images. **Not** the
  course-restricted dog-breeds dataset.
- **Count**: 89 images total; balanced 18-image validation set (6 modern / 6 standard /
  6 needs_renovation).
- **Classes**: modern, standard, needs_renovation.
- **Note**: raw images are not committed (size / licensing); the CV metrics are
  pre-computed in `docs/cv_eval/`.
