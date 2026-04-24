"""
Step 4b: Deduplicate near-identical components within each CC standard.

Existing LC components (source=lc_existing) are authoritative and not touched.
AI-generated components (source=ai_generated) are deduplicated within each
standard using SequenceMatcher string similarity.

This must run BEFORE step 5 so that the component pool fed into BNCC assignment
is clean at the standard level. All surviving components are then assigned 100%
to BNCC in step 5 — nothing is dropped at the BNCC level during matching.

Input/output: data/processed/cc_components_full.csv  (overwritten in place)
Backup:       data/processed/cc_components_full_pre_dedup.csv
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STR_SIM_THRESHOLD = 0.82

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV     = PROC / "cc_components_full.csv"
BACKUP_CSV = PROC / "cc_components_full_pre_dedup.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step4b_dedup_components.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def dedup_within_standard(group: pd.DataFrame) -> pd.DataFrame:
    """
    Within a single CC standard's component rows, remove near-duplicates.
    lc_existing rows are kept unconditionally; ai_generated rows are checked
    against all already-kept descriptions.
    """
    # Prioritise lc_existing rows — keep them all first
    existing = group[group["source"] == "lc_existing"]
    generated = group[group["source"] == "ai_generated"]

    kept_descs: list[str] = list(existing["description"].str.lower().str.strip())
    keep_indices: list[int] = list(existing.index)

    for idx, row in generated.iterrows():
        desc = str(row["description"]).lower().strip()
        if any(_sim(desc, kd) > STR_SIM_THRESHOLD for kd in kept_descs):
            continue
        kept_descs.append(desc)
        keep_indices.append(idx)

    return group.loc[keep_indices]


def main() -> None:
    log.info("=== Step 4b: Deduplicate components within each CC standard ===")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    log.info("Loaded %d components across %d CC standards", len(df), df["standard_id"].nunique())

    # Backup
    df.to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")
    log.info("Backup saved to %s", BACKUP_CSV.name)

    before = len(df)
    parts = []
    for _, group in df.groupby("standard_id", sort=False):
        parts.append(dedup_within_standard(group))
    cleaned = pd.concat(parts, ignore_index=True)
    removed = before - len(cleaned)

    cleaned.to_csv(IN_CSV, index=False, encoding="utf-8-sig")

    # Summary
    counts_after = cleaned.groupby("standard_id").size()
    log.info("Done: %d → %d components (removed %d near-duplicates within standards)",
             before, len(cleaned), removed)
    log.info("Per-standard after:")
    log.info("  mean=%.1f  median=%.0f  max=%d  min=%d",
             counts_after.mean(), counts_after.median(),
             counts_after.max(), counts_after.min())

    by_source = cleaned["source"].value_counts()
    log.info("  lc_existing  : %d", by_source.get("lc_existing", 0))
    log.info("  ai_generated : %d", by_source.get("ai_generated", 0))


if __name__ == "__main__":
    main()
