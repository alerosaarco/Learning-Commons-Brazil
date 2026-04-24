"""
Step 8: Weighted Jaccard similarity between each BNCC habilidade and every CC standard.

For BNCC habilidade H and CC standard S:

    weighted_intersection(H, S) = Σ weight(tier(b))
        for each BNCC component b ∈ H that matched a CC component c ∈ S

    score(H, S) = weighted_intersection / (|H| + |S| - weighted_intersection)

Tier weights:
    merge         → 1.0
    link_regional → 0.7
    link          → 0.5
    none          → 0.0  (skipped; contributes nothing)

Input:
    data/processed/bncc_component_matches.csv   (bncc_component_id, cc_component_id, match_tier)
    data/processed/bncc_components.csv          (habilidade_code, component_id, description)
    data/processed/cc_components_full.csv       (standard_id, component_id, ...)

Output:
    data/processed/bncc_cc_standard_matches.csv
    Columns: habilidade_code, cc_standard_id, score, rank,
             n_merge, n_link_regional, n_link, n_none,
             n_bncc_components, n_cc_components
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

TIER_WEIGHT: dict[str, float] = {
    "merge":         1.0,
    "link_regional": 0.7,
    "link":          0.5,
    "none":          0.0,
}
TOP_N = 10  # CC standards to keep per habilidade

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IN_MATCHES  = PROC / "bncc_component_matches.csv"
IN_BNCC     = PROC / "bncc_components.csv"
IN_CC       = PROC / "cc_components_full.csv"
OUT_CSV     = PROC / "bncc_cc_standard_matches.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step8_jaccard_matching.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=== Step 8: Weighted Jaccard BNCC ↔ CC standard matching ===")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    matches = pd.read_csv(IN_MATCHES, encoding="utf-8-sig")
    bncc    = pd.read_csv(IN_BNCC,    encoding="utf-8-sig")
    cc      = pd.read_csv(IN_CC,      encoding="utf-8-sig")

    log.info("Loaded: %d component matches, %d BNCC components, %d CC components",
             len(matches), len(bncc), len(cc))

    # -----------------------------------------------------------------------
    # Build lookup tables
    # -----------------------------------------------------------------------
    # bncc_component_id → (cc_component_id, tier)
    match_map: dict[str, tuple[str, str]] = {
        row["bncc_component_id"]: (row["cc_component_id"], row["match_tier"])
        for _, row in matches.iterrows()
    }

    # cc_component_id → standard_id
    cc_comp_to_std: dict[str, str] = dict(
        zip(cc["component_id"], cc["standard_id"])
    )

    # standard_id → number of CC components
    std_size: dict[str, int] = cc.groupby("standard_id").size().to_dict()

    # habilidade_code → [component_ids]
    hab_to_comps: dict[str, list[str]] = (
        bncc.groupby("habilidade_code")["component_id"].apply(list).to_dict()
    )

    log.info("%d habilidades, %d CC standards", len(hab_to_comps), len(std_size))

    # -----------------------------------------------------------------------
    # Compute weighted Jaccard for every (habilidade, CC standard) pair
    # that has at least one non-none match
    # -----------------------------------------------------------------------
    rows: list[dict] = []
    habilidades = sorted(hab_to_comps.keys())

    for hab_code in habilidades:
        comp_ids = hab_to_comps[hab_code]
        n_bncc = len(comp_ids)

        # Per CC standard, accumulate weighted intersection and tier counts
        std_w_inter:   dict[str, float]        = defaultdict(float)
        std_tier_cnt:  dict[str, dict[str,int]] = defaultdict(lambda: defaultdict(int))

        for cid in comp_ids:
            if cid not in match_map:
                continue
            cc_cid, tier = match_map[cid]
            weight = TIER_WEIGHT.get(tier, 0.0)

            # Track none-tier separately but still record the CC standard association
            std_id = cc_comp_to_std.get(cc_cid)
            if std_id is None:
                continue

            std_tier_cnt[std_id][tier] += 1
            if weight > 0:
                std_w_inter[std_id] += weight

        # Compute score for each candidate standard
        for std_id, w_inter in std_w_inter.items():
            n_cc = std_size.get(std_id, 0)
            denominator = n_bncc + n_cc - w_inter
            score = w_inter / denominator if denominator > 0 else 0.0
            tc = std_tier_cnt[std_id]
            rows.append({
                "habilidade_code":   hab_code,
                "cc_standard_id":    std_id,
                "score":             round(score, 6),
                "n_bncc_components": n_bncc,
                "n_cc_components":   n_cc,
                "n_merge":           tc.get("merge", 0),
                "n_link_regional":   tc.get("link_regional", 0),
                "n_link":            tc.get("link", 0),
                "n_none":            tc.get("none", 0),
            })

    df = pd.DataFrame(rows)
    log.info("Computed %d (habilidade, CC standard) pairs with score > 0", len(df))

    # -----------------------------------------------------------------------
    # Keep top-N per habilidade, ranked by score descending
    # -----------------------------------------------------------------------
    df = df.sort_values(["habilidade_code", "score"], ascending=[True, False])
    df["rank"] = df.groupby("habilidade_code").cumcount() + 1
    top = df[df["rank"] <= TOP_N].copy()

    # Reorder columns
    top = top[[
        "habilidade_code", "cc_standard_id", "score", "rank",
        "n_merge", "n_link_regional", "n_link", "n_none",
        "n_bncc_components", "n_cc_components",
    ]]

    top.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d rows to %s", len(top), OUT_CSV.name)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    habs_with_match = top["habilidade_code"].nunique()
    total_habs = len(hab_to_comps)
    log.info("Habilidades with ≥1 CC standard match: %d / %d (%.1f%%)",
             habs_with_match, total_habs, 100 * habs_with_match / total_habs)

    top1 = top[top["rank"] == 1]
    log.info("Score distribution (rank-1 matches):")
    log.info("  mean=%.4f  median=%.4f  min=%.4f  max=%.4f",
             top1["score"].mean(), top1["score"].median(),
             top1["score"].min(), top1["score"].max())

    # Tier breakdown across top-1 matches
    log.info("Tier contribution totals (rank-1 only):")
    log.info("  merge=%d  link_regional=%d  link=%d  none=%d",
             top1["n_merge"].sum(), top1["n_link_regional"].sum(),
             top1["n_link"].sum(), top1["n_none"].sum())

    # Habilidades with no match at all (all components were none-tier)
    no_match = set(hab_to_comps.keys()) - set(top["habilidade_code"])
    log.info("Habilidades with NO CC standard match: %d", len(no_match))
    if no_match:
        log.info("  Sample: %s", sorted(no_match)[:10])


if __name__ == "__main__":
    main()
