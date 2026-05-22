# Data — what goes where

This folder is `.gitignored` except for this README, `.gitkeep` files, the small
processed tables, and `raw/SNAPSHOT.md`. Raw downloads and images are reproduced
locally (see `SNAPSHOT.md` for exact sources and versions).

```
data/
├── raw/              ← downloaded originals — never edited by hand
│   ├── listings.csv             ← Kaggle ImmoScout24-CH rentals snapshot
│   ├── zurich_districts.csv     ← Stadt Zürich MPE open data
│   ├── rag_corpus/              ← 13 Markdown docs for RAG (one per district/topic)
│   └── SNAPSHOT.md              ← provenance: source + version of each dataset
├── interim/          ← intermediate joins / cleaning artifacts
├── processed/        ← final tables for modelling (.parquet, committed)
└── images/           ← labeled apartment photos (not committed)
    ├── train/{modern,standard,needs_renovation}/
    └── val/{modern,standard,needs_renovation}/
```

## Reproducing `raw/`

Exact sources and versions are recorded in [`raw/SNAPSHOT.md`](raw/SNAPSHOT.md).

- **listings.csv** — Kaggle dataset
  `fredeys/immoscout24-ch-switzerland-rental-property-dataset`; place the CSV at
  `raw/listings.csv`. Loading, Zurich filtering and Kreis assignment are handled by
  `src/immopilot/data/load_listings.py`.
- **zurich_districts** — Stadt Zürich Mietpreiserhebung (MPE) from
  https://data.stadt-zuerich.ch/ ; aggregated by `load_zurich_open.py`.
- **rag_corpus/** — one `.md` file per district/topic, with YAML frontmatter:
  ```yaml
  ---
  title: "Kreis 6 — Übersicht"
  url: "https://www.stadt-zuerich.ch/..."
  district: 6
  ---
  ```
- **images/** — self-collected and CC-licensed interior photos, ~30 per class,
  split into `train/` and `val/` by class folder.
