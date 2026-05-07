"""Polite scraper for Homegate Zurich rental listings.

⚠️ Always review and respect robots.txt and ToS before running.
This script is a SKELETON — implement selectors after inspecting the live site.
Recommended throttle: ≥ 2 seconds per request, randomized.
"""

from __future__ import annotations

import logging
import random
import time

import pandas as pd
from playwright.sync_api import sync_playwright

from immopilot import config

logger = logging.getLogger(__name__)


def scrape_listings(max_pages: int = 50) -> pd.DataFrame:
    rows: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for page_num in range(1, max_pages + 1):
            url = f"https://www.homegate.ch/mieten/wohnung/zurich?p={page_num}"
            logger.info("Fetching %s", url)
            page.goto(url, wait_until="networkidle")
            # TODO: inspect the live DOM and fill in selectors
            # cards = page.query_selector_all("div[data-test='result-list-item']")
            # for c in cards: rows.append(...)
            time.sleep(2 + random.random())
        browser.close()
    df = pd.DataFrame(rows)
    df.to_csv(config.RAW_DIR / "listings.csv", index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    scrape_listings()
