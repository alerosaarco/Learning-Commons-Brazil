"""
Step 7: Match BNCC components to CC components via TF-IDF cosine similarity.

For each BNCC component, find the most similar CC component.  Components are
written in a standardised action-verb style so key-term overlap is a reliable
proxy for semantic similarity (HuggingFace embeddings are unavailable in this
environment; swap SentenceTransformer back in when network access allows).

Three-tier output per match:
  merge  — similarity >= THRESHOLD_MERGE (0.55): same skill, reuse CC text+ID
  link   — similarity >= THRESHOLD_LINK  (0.35): related skill, explicit link
  (none) — similarity <  THRESHOLD_LINK:         BNCC-specific, no CC link

Note: TF-IDF thresholds are lower than embedding thresholds for equivalent
overlap.  0.55 TF-IDF ≈ 0.92 embedding; 0.35 TF-IDF ≈ 0.80 embedding.

Outputs:
  bncc_component_matches.csv  — (bncc_component_id, cc_component_id,
                                  similarity, match_tier)
  bncc_components.csv updated — description and component_id replaced with CC
                                 equivalents for 'merge' rows
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

THRESHOLD_MERGE = 0.55
THRESHOLD_LINK  = 0.35
CHUNK           = 500

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BNCC_COMPS_CSV = PROC / "bncc_components.csv"
CC_COMPS_CSV   = PROC / "cc_components_full.csv"
MATCHES_CSV    = PROC / "bncc_component_matches.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step7_match_components.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=== Step 7: Match BNCC ↔ CC components via TF-IDF cosine similarity ===")

    bncc_df = pd.read_csv(BNCC_COMPS_CSV, encoding="utf-8-sig")
    cc_df   = pd.read_csv(CC_COMPS_CSV,   encoding="utf-8-sig")

    log.info("BNCC components: %d", len(bncc_df))
    log.info("CC   components: %d", len(cc_df))

    all_texts = cc_df["description"].tolist() + bncc_df["description"].tolist()

    log.info("Fitting TF-IDF vectoriser on full corpus…")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit(all_texts)

    cc_mat   = vec.transform(cc_df["description"].tolist())    # (N_cc,   vocab)
    bncc_mat = vec.transform(bncc_df["description"].tolist())  # (N_bncc, vocab)

    log.info("Computing similarities (chunked)…")
    match_rows = []

    for start in tqdm(range(0, len(bncc_df), CHUNK), desc="matching"):
        chunk = bncc_mat[start: start + CHUNK]
        sims  = cosine_similarity(chunk, cc_mat)       # (chunk, N_cc)
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(best_idx)), best_idx]

        for k, (idx, sim) in enumerate(zip(best_idx, best_sim)):
            bncc_row = bncc_df.iloc[start + k]
            cc_row   = cc_df.iloc[idx]

            if sim >= THRESHOLD_MERGE:
                tier = "merge"
            elif sim >= THRESHOLD_LINK:
                tier = "link"
            else:
                tier = "none"

            match_rows.append({
                "bncc_component_id": bncc_row["component_id"],
                "cc_component_id":   cc_row["component_id"],
                "similarity":        round(float(sim), 4),
                "match_tier":        tier,
            })

    matches = pd.DataFrame(match_rows)
    matches.to_csv(MATCHES_CSV, index=False, encoding="utf-8-sig")

    # Summary
    tier_counts = matches["match_tier"].value_counts()
    log.info("Match results:")
    log.info("  merge (%s): %d  — BNCC component replaced with CC text+ID",
             THRESHOLD_MERGE, tier_counts.get("merge", 0))
    log.info("  link  (%.2f–%.2f): %d  — explicit link, separate IDs",
             THRESHOLD_LINK, THRESHOLD_MERGE, tier_counts.get("link", 0))
    log.info("  none  (<%s): %d  — BNCC-specific, no CC link",
             THRESHOLD_LINK, tier_counts.get("none", 0))

    # Apply merges: replace BNCC component_id and description with CC counterparts
    merge_map = (
        matches[matches["match_tier"] == "merge"]
        .set_index("bncc_component_id")[["cc_component_id"]]
    )
    cc_desc_map = cc_df.set_index("component_id")["description"].to_dict()

    bncc_updated = bncc_df.copy()
    merged_mask = bncc_updated["component_id"].isin(merge_map.index)
    bncc_updated.loc[merged_mask, "component_id"] = (
        bncc_updated.loc[merged_mask, "component_id"].map(merge_map["cc_component_id"])
    )
    bncc_updated.loc[merged_mask, "description"] = (
        bncc_updated.loc[merged_mask, "component_id"].map(cc_desc_map)
    )

    bncc_updated.to_csv(BNCC_COMPS_CSV, index=False, encoding="utf-8-sig")

    # Also update match file to use new (merged) component IDs
    matches.loc[matches["match_tier"] == "merge", "bncc_component_id"] = (
        matches.loc[matches["match_tier"] == "merge", "cc_component_id"]
    )
    matches.to_csv(MATCHES_CSV, index=False, encoding="utf-8-sig")

    log.info("bncc_components.csv updated: %d components replaced with CC equivalents",
             merged_mask.sum())
    log.info("bncc_component_matches.csv written: %d rows", len(matches))

    # Habilidade-level summary
    bncc_updated["is_cc_shared"] = bncc_updated["component_id"].isin(cc_df["component_id"])
    by_hab = bncc_updated.groupby("habilidade_code")["is_cc_shared"].agg(["sum","count"])
    by_hab["pct_shared"] = (by_hab["sum"] / by_hab["count"] * 100).round(1)
    log.info("Habilidades with ≥1 merged CC component: %d / %d",
             (by_hab["sum"] > 0).sum(), len(by_hab))
    log.info("Mean %% of components shared with CC per habilidade: %.1f%%",
             by_hab["pct_shared"].mean())


if __name__ == "__main__":
    main()
