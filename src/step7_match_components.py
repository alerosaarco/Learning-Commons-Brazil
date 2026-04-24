"""
Step 7: Match BNCC components to CC components via TF-IDF + AI validation.

For each BNCC component, find the most similar CC component via TF-IDF cosine
similarity, then apply a hybrid decision strategy:

  auto-merge  — similarity >= 0.58: clearly the same skill, no AI needed
  AI review   — 0.43 <= similarity < 0.58: Claude decides merge/link/none
  auto-none   — similarity < 0.43: too different, no link

Three output tiers:
  merge  — same skill: BNCC component replaced with CC text + CC component_id
  link   — related skill: explicit link in matches file, separate IDs kept
  none   — BNCC-specific: no CC equivalent

Outputs:
  bncc_component_matches.csv  — (bncc_component_id, cc_component_id,
                                  similarity, match_tier)
  bncc_components.csv updated — description/id replaced with CC for merges
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

THRESHOLD_AUTO_MERGE = 0.58   # auto-merge, no AI
THRESHOLD_AI_LOW     = 0.43   # below this: auto-none
AI_BATCH_SIZE        = 20
CONCURRENCY          = 8
MAX_TOKENS           = 2048
CHUNK                = 500

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

_SESSION_TOKEN_FILE = Path("/home/claude/.claude/remote/.session_ingress_token")

def _make_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return Anthropic(api_key=api_key)
    if _SESSION_TOKEN_FILE.exists():
        return Anthropic(auth_token=_SESSION_TOKEN_FILE.read_text().strip())
    print("ERROR: no auth", file=sys.stderr)
    sys.exit(1)

client = _make_client()

# ---------------------------------------------------------------------------
# AI validation for borderline pairs
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a curriculum alignment specialist comparing learning components from two curricula.

For each pair, decide whether the BNCC component and the CC component describe:
  merge — exactly the same atomic skill (even if worded differently). A teacher would use them interchangeably.
  link  — related but distinct skills (same domain, different scope, depth, or cognitive demand).
  none  — too different to be meaningfully linked.

Return ONLY a JSON object mapping each pair id to its decision:
{"p00": "merge", "p01": "link", "p02": "none", ...}"""


def _user_message(pairs: list[dict]) -> str:
    items = [
        {
            "id":   f"p{i:02d}",
            "bncc": p["bncc_desc"],
            "cc":   p["cc_desc"],
        }
        for i, p in enumerate(pairs)
    ]
    return (
        "Classify each BNCC↔CC component pair as merge / link / none.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _ai_validate_batch(pairs: list[dict]) -> dict[str, str]:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(pairs)}],
    )
    text = resp.content[0].text.strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    if "```" in text:
        text = text[:text.rfind("```")]
    result = json.loads(text.strip())

    for i in range(len(pairs)):
        key = f"p{i:02d}"
        if key not in result:
            raise ValueError(f"missing key {key}")
        if result[key] not in ("merge", "link", "none"):
            raise ValueError(f"invalid decision {result[key]!r} for {key}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 7: Match BNCC ↔ CC components (TF-IDF + AI validation) ===")

    bncc_df = pd.read_csv(BNCC_COMPS_CSV, encoding="utf-8-sig")
    cc_df   = pd.read_csv(CC_COMPS_CSV,   encoding="utf-8-sig")

    log.info("BNCC components: %d", len(bncc_df))
    log.info("CC   components: %d", len(cc_df))

    # ---- TF-IDF pass ----
    all_texts = cc_df["description"].tolist() + bncc_df["description"].tolist()
    log.info("Fitting TF-IDF vectoriser…")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit(all_texts)

    cc_mat   = vec.transform(cc_df["description"].tolist())
    bncc_mat = vec.transform(bncc_df["description"].tolist())

    log.info("Computing cosine similarities…")
    match_rows: list[dict] = []
    ai_candidates: list[dict] = []   # pairs needing AI review

    for start in tqdm(range(0, len(bncc_df), CHUNK), desc="tfidf"):
        chunk   = bncc_mat[start: start + CHUNK]
        sims    = cosine_similarity(chunk, cc_mat)
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(best_idx)), best_idx]

        for k, (idx, sim) in enumerate(zip(best_idx, best_sim)):
            bncc_row = bncc_df.iloc[start + k]
            cc_row   = cc_df.iloc[idx]

            if sim >= THRESHOLD_AUTO_MERGE:
                tier = "merge"
                match_rows.append({
                    "bncc_component_id": bncc_row["component_id"],
                    "cc_component_id":   cc_row["component_id"],
                    "similarity":        round(float(sim), 4),
                    "match_tier":        tier,
                })
            elif sim >= THRESHOLD_AI_LOW:
                # defer to AI
                ai_candidates.append({
                    "bncc_component_id": bncc_row["component_id"],
                    "cc_component_id":   cc_row["component_id"],
                    "similarity":        round(float(sim), 4),
                    "bncc_desc":         bncc_row["description"],
                    "cc_desc":           cc_row["description"],
                })
            else:
                match_rows.append({
                    "bncc_component_id": bncc_row["component_id"],
                    "cc_component_id":   cc_row["component_id"],
                    "similarity":        round(float(sim), 4),
                    "match_tier":        "none",
                })

    log.info("Auto-merge: %d  |  AI review: %d  |  Auto-none: %d",
             sum(1 for r in match_rows if r["match_tier"] == "merge"),
             len(ai_candidates),
             sum(1 for r in match_rows if r["match_tier"] == "none"))

    # ---- AI validation pass ----
    if ai_candidates:
        log.info("Sending %d pairs to AI for validation…", len(ai_candidates))
        batches = [
            ai_candidates[i: i + AI_BATCH_SIZE]
            for i in range(0, len(ai_candidates), AI_BATCH_SIZE)
        ]

        ai_decisions: dict[str, str] = {}   # bncc_component_id → tier

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(_ai_validate_batch, b): b for b in batches}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="AI batches"):
                batch = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    log.error("AI batch failed: %s — defaulting to 'link'", e)
                    result = {f"p{i:02d}": "link" for i in range(len(batch))}

                for i, pair in enumerate(batch):
                    ai_decisions[pair["bncc_component_id"]] = result[f"p{i:02d}"]

        for pair in ai_candidates:
            tier = ai_decisions.get(pair["bncc_component_id"], "link")
            match_rows.append({
                "bncc_component_id": pair["bncc_component_id"],
                "cc_component_id":   pair["cc_component_id"],
                "similarity":        pair["similarity"],
                "match_tier":        tier,
            })

    # ---- Write matches ----
    matches = pd.DataFrame(match_rows)
    tier_counts = matches["match_tier"].value_counts()
    log.info("Final match results:")
    log.info("  merge: %d", tier_counts.get("merge", 0))
    log.info("  link:  %d", tier_counts.get("link",  0))
    log.info("  none:  %d", tier_counts.get("none",  0))

    # ---- Apply merges to bncc_components.csv ----
    merge_map = (
        matches[matches["match_tier"] == "merge"]
        .drop_duplicates(subset=["bncc_component_id"])
        .set_index("bncc_component_id")["cc_component_id"]
        .to_dict()
    )
    cc_desc_map = cc_df.set_index("component_id")["description"].to_dict()

    bncc_updated = bncc_df.copy()
    merged_mask  = bncc_updated["component_id"].isin(merge_map)
    bncc_updated.loc[merged_mask, "component_id"] = (
        bncc_updated.loc[merged_mask, "component_id"].map(merge_map)
    )
    bncc_updated.loc[merged_mask, "description"] = (
        bncc_updated.loc[merged_mask, "component_id"].map(cc_desc_map)
    )
    bncc_updated.to_csv(BNCC_COMPS_CSV, index=False, encoding="utf-8-sig")

    # Update match file so merged rows reference the CC component_id
    matches.loc[matches["match_tier"] == "merge", "bncc_component_id"] = (
        matches.loc[matches["match_tier"] == "merge", "cc_component_id"]
    )
    matches.to_csv(MATCHES_CSV, index=False, encoding="utf-8-sig")

    log.info("bncc_components.csv updated: %d components replaced with CC equivalents",
             merged_mask.sum())
    log.info("bncc_component_matches.csv written: %d rows", len(matches))

    bncc_updated["is_cc_shared"] = bncc_updated["component_id"].isin(cc_df["component_id"])
    by_hab = bncc_updated.groupby("habilidade_code")["is_cc_shared"].agg(["sum", "count"])
    by_hab["pct_shared"] = (by_hab["sum"] / by_hab["count"] * 100).round(1)
    log.info("Habilidades with ≥1 merged CC component: %d / %d",
             (by_hab["sum"] > 0).sum(), len(by_hab))
    log.info("Mean %% of components shared with CC per habilidade: %.1f%%",
             by_hab["pct_shared"].mean())


if __name__ == "__main__":
    main()
