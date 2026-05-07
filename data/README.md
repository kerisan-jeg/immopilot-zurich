# Data — what goes where

This folder is `.gitignored` except for this README, `.gitkeep` files, and
small samples. All actual data is reproduced via `make data`.

```
data/
├── raw/              ← downloaded / scraped originals — NEVER edit by hand
│   ├── listings.csv             ← rentals snapshot (Kaggle or scrape)
│   ├── zurich_districts.csv     ← Stadt Zürich open data
│   ├── rag_corpus/              ← .md/.txt files for RAG (one doc per district)
│   └── SNAPSHOT.md              ← document the date/version of each download
├── interim/          ← intermediate joins, cleaning artifacts
├── processed/        ← final tables for modelling (.parquet)
└── images/           ← labeled apartment photos
    ├── samples/                 ← committed: 9 example images for tests
    ├── train/{modern,standard,needs_renovation}/
    └── val/{modern,standard,needs_renovation}/
```

## How to populate `raw/`

### `listings.csv`
Pick ONE option and document it in `SNAPSHOT.md`:

1. **Kaggle** — search "Switzerland apartment rentals" or "Zurich rentals",
   download the CSV, place it as `raw/listings.csv`. Pin the dataset version
   number.
2. **Scrape Homegate** — `python scripts/scrape_homegate.py` (Playwright).
   Respect `robots.txt`, throttle requests, store the snapshot date.

### `zurich_districts.csv`
Replace the placeholder URL in `src/immopilot/data/load_zurich_open.py` with
real Stadt Zürich open-data resource URLs from
https://data.stadt-zuerich.ch/

### `rag_corpus/`
Drop one `.md` file per district + topic. Optional YAML frontmatter:

```yaml
---
title: "Kreis 6 — Übersicht"
url: "https://www.stadt-zuerich.ch/..."
district: 6
---
```

### `images/`
Collect from your own visits, friends' apartments, and CC-licensed sources.
Aim for ~150 images per class to start. Annotation guideline in
`docs/cv_annotation_guide.md` (you'll write this).
