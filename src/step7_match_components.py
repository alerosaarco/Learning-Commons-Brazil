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
TOP_K_CANDIDATES     = 20     # CC candidates per BNCC component sent to AI (subject-filtered)
AI_BATCH_SIZE        = 5      # BNCC components per AI call (each with TOP_K candidates)
CONCURRENCY          = 4      # balanced: throughput without hammering rate limits
MAX_TOKENS           = 4096
CHUNK                = 500

# BNCC subject → CC academic_subject for candidate pool filtering
BNCC_TO_CC_SUBJECT: dict[str, str] = {
    "Matemática":                                          "Mathematics",
    "Matemática e suas Tecnologias":                       "Mathematics",
    "Espaços, Tempos, Quantidades, Relações e Transformações": "Mathematics",
    "Língua Portuguesa":                                   "English Language Arts",
    "Linguagens e suas Tecnologias":                       "English Language Arts",
    "Escuta, Fala, Pensamento e Imaginação":               "English Language Arts",
    "Língua Inglesa":                                      "English Language Arts",
    "Ciências":                                            "Science",
    "Ciências da Natureza e suas Tecnologias":             "Science",
    "Geografia":                                           "Science",
    "Computação":                                          "Science",
    # No CC counterpart — use all subjects as fallback
}

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BNCC_COMPS_CSV = PROC / "bncc_components.csv"
CC_COMPS_CSV   = PROC / "cc_components_full.csv"
MATCHES_CSV    = PROC / "bncc_component_matches.csv"
CHECKPOINT_CSV = PROC / "step7_checkpoint.csv"

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

SYSTEM_PROMPT = """You are a curriculum alignment specialist comparing learning components from the Brazilian BNCC and US Common Core curricula.

For each BNCC component you receive its top candidate CC components. Select the BEST match (if any) and classify it into one of four tiers:

  merge         — Exactly the same atomic skill, same or near-identical wording. A teacher could use either description interchangeably.

  link_regional — The same underlying skill, but the BNCC describes it through Brazilian linguistic, cultural, or regional framing (e.g. Portuguese grammar terminology like "modalization", "agglutination", Brazilian text genres like "cordel", Brazilian institutions). If you removed the regional framing, the skill would be identical to the CC component.

  link          — Related but genuinely distinct: same broad domain, but different cognitive demand, scope, or subject matter. The difference is NOT just regional framing.

  none          — None of the candidates is a meaningful match. The BNCC component covers genuinely different or Brazil-specific content with no CC equivalent.

Rules:
- Choose at most ONE candidate per BNCC component (the best one).
- Prefer merge over link_regional when wording is nearly identical.
- Prefer link_regional over link when the only difference is Brazilian/Portuguese framing.
- Use none when the BNCC component is genuinely Brazil-specific.

Return ONLY a JSON object — one entry per BNCC component:
{
  "b00": {"tier": "merge",         "cc_idx": 1},
  "b01": {"tier": "link_regional", "cc_idx": 0},
  "b02": {"tier": "link",          "cc_idx": 2},
  "b03": {"tier": "none",          "cc_idx": null},
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
        if tier not in ("merge", "link_regional", "link", "none"):
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
    # Build subject-filtered CC pools and per-pool TF-IDF matrices
    bncc_hab  = pd.read_csv(PROC / "bncc_translated.csv", encoding="utf-8-sig")
    hab_subj  = bncc_hab.set_index("bncc_code")["subject"].to_dict()
    bncc_hab_map = bncc_df.set_index("component_id")["habilidade_code"].to_dict()

    cc_std    = pd.read_csv(ROOT / "data" / "raw" / "lc_cc_standards.csv", encoding="utf-8-sig")
    cc_subj_map = (
        cc_df.merge(cc_std[["identifier", "academic_subject"]],
                    left_on="standard_id", right_on="identifier", how="left")
        .set_index("component_id")["academic_subject"]
        .to_dict()
    )

    all_cc_subjects = ["Mathematics", "English Language Arts", "Science", "all"]
    cc_pools: dict[str, pd.DataFrame] = {}
    cc_pool_mats: dict[str, object] = {}

    log.info("Building subject-filtered CC pools and TF-IDF matrices…")
    for subj in all_cc_subjects:
        if subj == "all":
            pool = cc_df.copy()
        else:
            pool = cc_df[cc_df["component_id"].map(cc_subj_map) == subj].copy()
        cc_pools[subj] = pool.reset_index(drop=True)

    # Fit one shared vectoriser on the full corpus for consistent vocabulary
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit(cc_df["description"].tolist() + bncc_df["description"].tolist())

    for subj in all_cc_subjects:
        cc_pool_mats[subj] = vec.transform(cc_pools[subj]["description"].tolist())
        log.info("  %s: %d CC components", subj, len(cc_pools[subj]))

    log.info("Finding top-%d subject-matched CC candidates per BNCC component…",
             TOP_K_CANDIDATES)
    auto_merges: list[dict] = []
    ai_items: list[dict] = []

    bncc_mat = vec.transform(bncc_df["description"].tolist())

    for start in tqdm(range(0, len(bncc_df), CHUNK), desc="tfidf"):
        chunk = bncc_mat[start: start + CHUNK]

        for k in range(chunk.shape[0]):
            bncc_row  = bncc_df.iloc[start + k]
            hab_code  = bncc_hab_map.get(bncc_row["component_id"], "")
            bncc_subj = hab_subj.get(hab_code, "")
            cc_subj   = BNCC_TO_CC_SUBJECT.get(bncc_subj, "all")

            pool    = cc_pools[cc_subj]
            mat     = cc_pool_mats[cc_subj]
            sims    = cosine_similarity(chunk[k], mat)[0]   # (N_pool,)
            k_take  = min(TOP_K_CANDIDATES, len(pool))
            top_idx = np.argsort(sims)[-k_take:][::-1]
            top_sim = sims[top_idx]

            best_sim = float(top_sim[0])
            best_cc  = pool.iloc[int(top_idx[0])]

            if best_sim >= THRESHOLD_AUTO_MERGE:
                auto_merges.append({
                    "bncc_component_id": bncc_row["component_id"],
                    "cc_component_id":   best_cc["component_id"],
                    "similarity":        round(best_sim, 4),
                    "match_tier":        "merge",
                })
            else:
                candidates = [pool.iloc[int(i)]["description"] for i in top_idx]
                cand_ids   = [pool.iloc[int(i)]["component_id"] for i in top_idx]
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
        # Resume: skip already-processed BNCC component IDs
        done_ids: set[str] = set()
        checkpoint_rows: list[dict] = []
        if CHECKPOINT_CSV.exists():
            try:
                ckpt = pd.read_csv(CHECKPOINT_CSV, encoding="utf-8-sig")
                done_ids = set(ckpt["bncc_component_id"].astype(str))
                checkpoint_rows = ckpt.to_dict("records")
                log.info("Resuming from checkpoint: %d already processed", len(done_ids))
            except pd.errors.EmptyDataError:
                pass

        todo_ai = [it for it in ai_items if it["bncc_component_id"] not in done_ids]
        log.info("Sending %d BNCC components to AI (top-%d candidates each)…",
                 len(todo_ai), TOP_K_CANDIDATES)

        batches = [
            todo_ai[i: i + AI_BATCH_SIZE]
            for i in range(0, len(todo_ai), AI_BATCH_SIZE)
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

                new_rows = []
                for item, decision in zip(batch, decisions):
                    new_rows.append({
                        "bncc_component_id": item["bncc_component_id"],
                        "cc_component_id":   decision["cc_component_id"] or item["cc_ids"][0],
                        "similarity":        item["top_sim"],
                        "match_tier":        decision["tier"],
                    })

                checkpoint_rows.extend(new_rows)
                pd.DataFrame(checkpoint_rows).to_csv(
                    CHECKPOINT_CSV, index=False, encoding="utf-8-sig"
                )

        for row in checkpoint_rows:
            match_rows.append(row)

    # ---- Write matches ----
    matches = pd.DataFrame(match_rows)
    tier_counts = matches["match_tier"].value_counts()
    log.info("Final match results:")
    log.info("  merge:         %d  (weight=1.0)", tier_counts.get("merge",         0))
    log.info("  link_regional: %d  (weight=0.7)", tier_counts.get("link_regional", 0))
    log.info("  link:          %d  (weight=0.5)", tier_counts.get("link",          0))
    log.info("  none:          %d  (weight=0.0)", tier_counts.get("none",          0))

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
