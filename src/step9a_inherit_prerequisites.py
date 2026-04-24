"""
Step 9a: Inherit CC prerequisite edges into BNCC habilidade graph.

Algorithm:
  For each CC prerequisite edge (standard_id → prerequisite_id):
    If both CC standards have BNCC habilidade mappings with score ≥ MIN_SCORE:
      For each BNCC habilidade H mapped to standard_id (score ≥ MIN_SCORE):
        For each BNCC habilidade P mapped to prerequisite_id (score ≥ MIN_SCORE):
          If grade(P) ≤ grade(H):   # no inversions
            Emit edge (H depends on P)

Quality filters applied:
  1. MIN_SCORE = 0.08 — both CC match scores must reach this threshold
  2. Grade order — prerequisite habilidade must be at same or lower grade level
     Grade hierarchy: EI01 < EI02 < EI03 < EF01 < ... < EF09 < EM (indeterminate)
     EM-EM edges are kept regardless (no grade number to compare)

Input:
    data/raw/lc_cc_prerequisites.csv
    data/raw/lc_cc_standards.csv
    data/processed/bncc_cc_standard_matches.csv
    data/processed/bncc_translated.csv

Output:
    data/processed/bncc_prerequisites.csv
    Columns:
        habilidade_code, prerequisite_code,
        n_cc_sources, best_hab_cc_score, best_pre_cc_score,
        source_cc_standard_id, source_cc_prereq_id, source_cc_prereq_code
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

MIN_SCORE = 0.08   # minimum CC match score for both ends of a prerequisite edge

ROOT     = Path(__file__).resolve().parent.parent
RAW      = ROOT / "data" / "raw"
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IN_PREREQS  = RAW  / "lc_cc_prerequisites.csv"
IN_CC_STDS  = RAW  / "lc_cc_standards.csv"
IN_MATCHES  = PROC / "bncc_cc_standard_matches.csv"
IN_BNCC     = PROC / "bncc_translated.csv"
OUT_CSV     = PROC / "bncc_prerequisites.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step9a_inherit_prerequisites.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def _grade_level(bncc_code: str) -> int | None:
    """
    Return a sortable integer grade level for a BNCC code.
      EI01 → 1,  EI02 → 2,  EI03 → 3
      EF01 → 11, EF02 → 12, ..., EF09 → 19
      EM   → 20 (all EM habilidades treated as same level)
      Unknown → None
    """
    code = bncc_code.upper()
    m = re.match(r"^(EI|EF|EM)(\d{2})", code)
    if not m:
        return None
    prefix, digits = m.group(1), int(m.group(2))
    if prefix == "EI":
        return digits          # 1-3
    if prefix == "EF":
        return 10 + digits     # 11-19
    if prefix == "EM":
        return 20
    return None


def _grade_order_ok(hab_code: str, pre_code: str) -> bool:
    """Return True if pre_code is at a grade level ≤ hab_code (valid prereq direction)."""
    g_hab = _grade_level(hab_code)
    g_pre = _grade_level(pre_code)
    if g_hab is None or g_pre is None:
        return True   # can't determine — keep the edge
    if g_hab == 20 and g_pre == 20:
        return True   # both EM — no ordering
    return g_pre <= g_hab


def main() -> None:
    log.info("=== Step 9a: Inherit CC prerequisite edges into BNCC graph ===")
    log.info("MIN_SCORE=%.2f  grade-order filter=ON", MIN_SCORE)

    prereqs = pd.read_csv(IN_PREREQS, encoding="utf-8-sig")
    cc_stds = pd.read_csv(IN_CC_STDS,  encoding="utf-8-sig")
    matches = pd.read_csv(IN_MATCHES,  encoding="utf-8-sig")
    bncc    = pd.read_csv(IN_BNCC,     encoding="utf-8-sig")

    log.info("CC prerequisites: %d edges", len(prereqs))

    cc_code_map: dict[str, str] = dict(zip(cc_stds["identifier"], cc_stds["statement_code"]))

    # CC standard_id → list of (habilidade_code, score), score ≥ MIN_SCORE, sorted best first
    std_to_habs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for _, row in matches[matches["score"] >= MIN_SCORE].sort_values("score", ascending=False).iterrows():
        std_to_habs[row["cc_standard_id"]].append((row["habilidade_code"], float(row["score"])))

    matched_stds = set(std_to_habs.keys())
    log.info("CC standards with ≥1 BNCC match (score≥%.2f): %d", MIN_SCORE, len(matched_stds))

    edge_map: dict[tuple[str, str], list[tuple[str, str, float, float]]] = defaultdict(list)

    skipped_unmatched = skipped_grade = used = 0

    for _, row in prereqs.iterrows():
        std_id = row["standard_id"]
        pre_id = row["prerequisite_id"]

        if std_id not in matched_stds or pre_id not in matched_stds:
            skipped_unmatched += 1
            continue

        habs = std_to_habs[std_id]
        pres = std_to_habs[pre_id]

        for hab_code, hab_score in habs:
            for pre_code, pre_score in pres:
                if hab_code == pre_code:
                    continue
                if not _grade_order_ok(hab_code, pre_code):
                    skipped_grade += 1
                    continue
                edge_map[(hab_code, pre_code)].append(
                    (std_id, pre_id, hab_score, pre_score)
                )
        used += 1

    log.info("CC edges used: %d  |  skipped (score/missing): %d  |  grade-inverted pairs dropped: %d",
             used, skipped_unmatched, skipped_grade)
    log.info("Raw BNCC pairs: %d  |  unique after dedup: %d",
             sum(len(v) for v in edge_map.values()), len(edge_map))

    rows = []
    for (hab_code, pre_code), sources in edge_map.items():
        best = max(sources, key=lambda x: x[2] + x[3])
        std_id, pre_id, hab_score, pre_score = best
        rows.append({
            "habilidade_code":       hab_code,
            "prerequisite_code":     pre_code,
            "n_cc_sources":          len(sources),
            "best_hab_cc_score":     round(hab_score, 6),
            "best_pre_cc_score":     round(pre_score, 6),
            "source_cc_standard_id": std_id,
            "source_cc_prereq_id":   pre_id,
            "source_cc_prereq_code": cc_code_map.get(pre_id, ""),
        })

    df = pd.DataFrame(rows).sort_values(
        ["habilidade_code", "best_hab_cc_score"],
        ascending=[True, False],
    )
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d BNCC prerequisite edges to %s", len(df), OUT_CSV.name)

    total_habs = bncc["bncc_code"].nunique()
    habs_with_prereq = df["habilidade_code"].nunique()
    habs_as_prereq   = df["prerequisite_code"].nunique()
    log.info("Habilidades with ≥1 prerequisite: %d / %d (%.1f%%)",
             habs_with_prereq, total_habs, 100 * habs_with_prereq / total_habs)
    log.info("Habilidades that serve as a prerequisite: %d / %d (%.1f%%)",
             habs_as_prereq, total_habs, 100 * habs_as_prereq / total_habs)

    in_degree = df.groupby("habilidade_code")["prerequisite_code"].count()
    log.info("Prerequisites per habilidade: mean=%.1f  median=%.0f  max=%d",
             in_degree.mean(), in_degree.median(), in_degree.max())

    multi = df[df["n_cc_sources"] > 1]
    log.info("Pairs supported by >1 CC edge: %d (stronger signal)", len(multi))

    # Sample: show a few edges with context
    cc_desc_map = dict(zip(cc_stds["identifier"], cc_stds["description"]))
    desc_map    = dict(zip(bncc["bncc_code"], bncc["description_en"]))
    log.info("Sample edges:")
    for _, r in df.head(8).iterrows():
        log.info("  %s → prereq %s",
                 r["habilidade_code"], r["prerequisite_code"])
        log.info("    HAB: %s", desc_map.get(r["habilidade_code"], "?")[:70])
        log.info("    PRE: %s", desc_map.get(r["prerequisite_code"], "?")[:70])
        log.info("    CC:  [%s] → [%s]  scores=(%.3f, %.3f)",
                 cc_desc_map.get(r["source_cc_standard_id"], "?")[:50],
                 cc_desc_map.get(r["source_cc_prereq_id"], "?")[:50],
                 r["best_hab_cc_score"], r["best_pre_cc_score"])


if __name__ == "__main__":
    main()
