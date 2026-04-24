"""
Step 6b: Deduplicate near-identical components within each BNCC habilidade.

Mirrors step 4b exactly — string similarity dedup within each habilidade's
component set.  No cross-habilidade dedup (same rationale as CC standards:
shared atomic skills across habilidades are signal for matching, not noise).

Input/output: data/processed/bncc_components.csv  (overwritten in place)
Backup:       data/processed/bncc_components_pre_dedup.csv
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

STR_SIM_THRESHOLD = 0.82

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV     = PROC / "bncc_components.csv"
BACKUP_CSV = PROC / "bncc_components_pre_dedup.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step6b_dedup_bncc_components.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def dedup_within_habilidade(group: pd.DataFrame) -> pd.DataFrame:
    kept_descs: list[str] = []
    keep_indices: list[int] = []
    for idx, row in group.iterrows():
        desc = str(row["description"]).lower().strip()
        if any(_sim(desc, kd) > STR_SIM_THRESHOLD for kd in kept_descs):
            continue
        kept_descs.append(desc)
        keep_indices.append(idx)
    return group.loc[keep_indices]


def main() -> None:
    log.info("=== Step 6b: Deduplicate BNCC components within each habilidade ===")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    log.info("Loaded %d components across %d habilidades",
             len(df), df["habilidade_code"].nunique())

    df.to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")
    log.info("Backup saved to %s", BACKUP_CSV.name)

    before = len(df)
    parts = []
    for _, group in df.groupby("habilidade_code", sort=False):
        parts.append(dedup_within_habilidade(group))
    cleaned = pd.concat(parts, ignore_index=True)
    removed = before - len(cleaned)

    cleaned.to_csv(IN_CSV, index=False, encoding="utf-8-sig")

    counts = cleaned.groupby("habilidade_code").size()
    log.info("Done: %d → %d components (removed %d within-habilidade near-duplicates)",
             before, len(cleaned), removed)
    log.info("Per-habilidade: mean=%.1f  median=%.0f  min=%d  max=%d",
             counts.mean(), counts.median(), counts.min(), counts.max())


if __name__ == "__main__":
    main()
