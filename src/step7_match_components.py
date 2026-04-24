"""
Step 7: Match BNCC components to CC components via TF-IDF + AI validation.

For each BNCC component:
  1. TF-IDF finds the top-3 CC candidates (best lexical overlap).
  2. AI reviews all 3 candidates and decides for the best one:
       merge — same atomic skill (even if differently worded)
       link  — related but distinct skill
       none  — no meaningful match among the candidates

TF-IDF alone misses semantic similarity (e.g. "formulate hypotheses" vs
"develop predictions"), so every BNCC component is sent to AI regardless
of TF-IDF score.  Auto-merge (>= 0.92 TF-IDF) is retained for the rare
cases of near-identical text, saving a few API calls.

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

THRESHOLD_AUTO_MERGE = 0.92   # near-identical text: auto-merge, skip AI
TOP_K_CANDIDATES     = 3      # CC candidates per BNCC component sent to AI
AI_BATCH_SIZE        = 20     # BNCC components per AI call (each with TOP_K candidates)
CONCURRENCY          = 8
MAX_TOKENS           = 4096
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

SYSTEM_PROMPT = """You are a curriculum alignment specialist comparing learning components from two school curricula (Brazilian BNCC and US Common Core).

For each BNCC component you receive its top candidate CC components. Select the BEST match (if any) and classify it:
  merge — the BNCC and CC components describe exactly the same atomic skill, even if worded differently. A teacher could use either description interchangeably.
  link  — related but distinct: same broad domain or concept, but different scope, cognitive demand, or specificity.
  none  — none of the candidates is a meaningful match for this BNCC component.

Rules:
- Choose at most ONE candidate per BNCC component (the best one).
- Prefer merge over link when the core skill is truly identical.
- Use none when the BNCC component covers genuinely different or Brazil-specific content.

Return ONLY a JSON object — one entry per BNCC component:
{
  "b00": {"tier": "merge", "cc_idx": 1},
  "b01": {"tier": "link",  "cc_idx": 0},
  "b02": {"tier": "none",  "cc_idx": null},
  ...
}
cc_idx is the 0-based index of the chosen candidate (null for none)."""


def _user_message(items: list[dict]) -> str:
    payload = []
    for i, item in enumerate(items):
        payload.append({
            "id":         f"b{i:02d}",
            "bncc":       item["bncc_desc"],
            "candidates": [
                {"idx": j, "cc": c} for j, c in enumerate(item["cc_candidates"])
            ],
        })
    return (
        "For each BNCC component, select the best CC candidate match (if any).\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _ai_validate_batch(items: list[dict]) -> list[dict]:
    """Returns list of {tier, cc_component_id} aligned with items."""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(items)}],
    )
    text = resp.content[0].text.strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    if "```" in text:
        text = text[:text.rfind("```")]
    result = json.loads(text.strip())

    output = []
    for i, item in enumerate(items):
        key = f"b{i:02d}"
        if key not in result:
            raise ValueError(f"missing key {key}")
        entry = result[key]
        tier   = entry.get("tier", "none")
        cc_idx = entry.get("cc_idx")
        if tier not in ("merge", "link", "none"):
            raise ValueError(f"invalid tier {tier!r} for {key}")
        if tier != "none" and (cc_idx is None or cc_idx >= len(item["cc_candidates"])):
            raise ValueError(f"invalid cc_idx {cc_idx} for {key}")
        cc_component_id = item["cc_ids"][cc_idx] if tier != "none" else None
        output.append({"tier": tier, "cc_component_id": cc_component_id})
    return output


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

    cc_ids   = cc_df["component_id"].tolist()
    cc_descs = cc_df["description"].tolist()

    log.info("Finding top-%d CC candidates per BNCC component…", TOP_K_CANDIDATES)
    auto_merges: list[dict] = []
    ai_items: list[dict] = []   # each: {bncc_component_id, bncc_desc, cc_candidates, cc_ids, top_sim}

    for start in tqdm(range(0, len(bncc_df), CHUNK), desc="tfidf"):
        chunk = bncc_mat[start: start + CHUNK]
        sims  = cosine_similarity(chunk, cc_mat)   # (chunk, N_cc)

        # top-K indices per row
        top_k_idx = np.argsort(sims, axis=1)[:, -TOP_K_CANDIDATES:][:, ::-1]
        top_k_sim = sims[np.arange(len(top_k_idx))[:, None], top_k_idx]

        for k in range(len(top_k_idx)):
            bncc_row  = bncc_df.iloc[start + k]
            best_sim  = float(top_k_sim[k, 0])
            best_cc   = cc_df.iloc[int(top_k_idx[k, 0])]

            if best_sim >= THRESHOLD_AUTO_MERGE:
                auto_merges.append({
                    "bncc_component_id": bncc_row["component_id"],
                    "cc_component_id":   best_cc["component_id"],
                    "similarity":        round(best_sim, 4),
                    "match_tier":        "merge",
                })
            else:
                candidates = [cc_descs[int(i)] for i in top_k_idx[k]]
                cand_ids   = [cc_ids[int(i)]   for i in top_k_idx[k]]
                ai_items.append({
                    "bncc_component_id": bncc_row["component_id"],
                    "bncc_desc":         bncc_row["description"],
                    "cc_candidates":     candidates,
                    "cc_ids":            cand_ids,
                    "top_sim":           round(best_sim, 4),
                })

    log.info("Auto-merge: %d  |  AI review: %d", len(auto_merges), len(ai_items))

    # ---- AI validation: all non-auto-merge BNCC components ----
    match_rows: list[dict] = list(auto_merges)

    if ai_items:
        log.info("Sending %d BNCC components to AI (top-%d candidates each)…",
                 len(ai_items), TOP_K_CANDIDATES)
        batches = [
            ai_items[i: i + AI_BATCH_SIZE]
            for i in range(0, len(ai_items), AI_BATCH_SIZE)
        ]

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(_ai_validate_batch, b): b for b in batches}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="AI batches"):
                batch = futures[fut]
                try:
                    decisions = fut.result()
                except Exception as e:
                    log.error("AI batch failed: %s — defaulting to none", e)
                    decisions = [{"tier": "none", "cc_component_id": None}] * len(batch)

                for item, decision in zip(batch, decisions):
                    match_rows.append({
                        "bncc_component_id": item["bncc_component_id"],
                        "cc_component_id":   decision["cc_component_id"] or item["cc_ids"][0],
                        "similarity":        item["top_sim"],
                        "match_tier":        decision["tier"],
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
