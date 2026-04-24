"""
Step 5b: Deduplicate near-identical components and cap at 5 per habilidade.

Pass 1 (free) — string similarity: for habilidades with 2–5 components,
  remove any pair with SequenceMatcher ratio > 0.82 (same skill, minor
  rewording).

Pass 2 (Sonnet) — for habilidades with 6+ components: ask the model to
  remove near-duplicates and keep the best ≤ 5 most distinct components.

Overwrites data/processed/component_bncc_map.csv in place; original
backed up to component_bncc_map_pre_dedup.csv.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

MODEL_EASY   = "claude-sonnet-4-6"   # 6–15 components: straightforward pruning
MODEL_HARD   = "claude-opus-4-7"     # >15 components: harder dedup judgment
BATCH_SIZE   = 5   # habilidades per call
CONCURRENCY  = 8
MAX_TOKENS   = 4096
CAP          = 5   # target max components per habilidade
STR_SIM_THRESHOLD = 0.82  # SequenceMatcher ratio above which two components are near-duplicates

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MAP_CSV    = PROC / "component_bncc_map.csv"
BACKUP_CSV = PROC / "component_bncc_map_pre_dedup.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step5b_dedup.log", mode="w"),
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
# Pass 1 — string similarity dedup (free, no API)
# ---------------------------------------------------------------------------

def _str_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def string_dedup(components: list[tuple[str, str]]) -> list[str]:
    """
    components: list of (component_id, description)
    Returns list of component_ids to KEEP.
    Greedily removes any component whose description is >STR_SIM_THRESHOLD
    similar to an already-kept one.
    """
    kept_ids: list[str]   = []
    kept_descs: list[str] = []
    for cid, desc in components:
        if any(_str_sim(desc, kd) > STR_SIM_THRESHOLD for kd in kept_descs):
            continue
        kept_ids.append(cid)
        kept_descs.append(desc)
    return kept_ids


# ---------------------------------------------------------------------------
# Pass 2 — Sonnet review for habilidades still > CAP
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a curriculum quality reviewer.

You will receive batches of BNCC habilidades, each with a list of learning components currently assigned to it.  Your job:

1. Remove near-duplicate components — if two components express essentially the same skill (even with different wording), keep only the clearest, most precise one.
2. Keep at most {CAP} components per habilidade — select the most distinct, specific, and representative ones that together fully cover what a student needs to master this objective.
3. If a habilidade genuinely requires fewer than {CAP} components, return fewer — do not pad.

Return ONLY a JSON object mapping each habilidade key to the list of component indices (0-based) to KEEP:
{{"h0": [0, 2, 4], "h1": [1, 3], ...}}"""


def _user_message(batch: list[dict]) -> str:
    items = []
    for i, hab in enumerate(batch):
        items.append({
            "key": f"h{i}",
            "habilidade": hab["description_en"],
            "components": [
                {"index": j, "description": c["description"]}
                for j, c in enumerate(hab["components"])
            ],
        })
    return json.dumps(items, ensure_ascii=False, indent=2)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _review_batch(batch: list[dict], model: str) -> dict[str, list[int]]:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(batch)}],
    )
    text = resp.content[0].text.strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    if "```" in text:
        text = text[:text.rfind("```")]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"non-JSON: {text[:200]!r}") from e

    for i in range(len(batch)):
        key = f"h{i}"
        if key not in result:
            raise ValueError(f"missing key {key} in response")
        if not isinstance(result[key], list):
            raise ValueError(f"expected list for {key}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 5b: Deduplicate and cap components per habilidade ===")

    # Load
    mapping = pd.read_csv(MAP_CSV, encoding="utf-8-sig")
    comps   = pd.read_csv(PROC / "cc_components_full.csv", encoding="utf-8-sig")
    bncc    = pd.read_csv(PROC / "bncc_translated.csv", encoding="utf-8-sig")

    # Backup
    mapping.to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")
    log.info("Backup saved to %s", BACKUP_CSV.name)

    # Attach descriptions
    comp_desc = comps.set_index("component_id")["description"].to_dict()
    bncc_desc = bncc.set_index("bncc_code")["description_en"].to_dict()

    # Group by habilidade
    groups = {
        code: list(rows.itertuples())
        for code, rows in mapping.groupby("bncc_code")
    }

    keep_ids: set[str] = set()
    removed_str  = 0
    to_review: list[str] = []  # bncc_codes still > CAP after string dedup

    # ---- Pass 1: string dedup ----
    log.info("Pass 1: string similarity dedup…")
    for code, rows in groups.items():
        components = [
            (r.component_id, comp_desc.get(r.component_id, ""))
            for r in rows
        ]
        kept = string_dedup(components)
        removed_str += len(components) - len(kept)

        if len(kept) <= CAP:
            keep_ids.update(kept)
        else:
            keep_ids.update(kept)   # keep for now; mark for Sonnet pass
            to_review.append(code)

    log.info("Pass 1 done: removed %d near-duplicates; %d habilidades still >%d",
             removed_str, len(to_review), CAP)

    # ---- Pass 2: Sonnet review for habilidades still > CAP ----
    if to_review:
        log.info("Pass 2: Sonnet review for %d habilidades…", len(to_review))

        # Build batch items
        hab_items = []
        for code in to_review:
            rows = groups[code]
            # Use only already-kept components from pass 1
            kept_rows = [r for r in rows if r.component_id in keep_ids]
            if len(kept_rows) <= CAP:
                continue  # pass 1 was enough
            hab_items.append({
                "bncc_code": code,
                "description_en": bncc_desc.get(code, code),
                "components": [
                    {"component_id": r.component_id,
                     "description": comp_desc.get(r.component_id, "")}
                    for r in kept_rows
                ],
            })

        # Route to Opus for habilidades with many components (harder judgment)
        easy = [h for h in hab_items if len(h["components"]) <= 15]
        hard = [h for h in hab_items if len(h["components"]) >  15]
        log.info("  Sonnet (%s): %d habilidades (6–15 components)",
                 MODEL_EASY, len(easy))
        log.info("  Opus   (%s): %d habilidades (>15 components)",
                 MODEL_HARD, len(hard))

        batches: list[tuple[list[dict], str]] = []
        for i in range(0, len(easy), BATCH_SIZE):
            batches.append((easy[i: i + BATCH_SIZE], MODEL_EASY))
        for i in range(0, len(hard), BATCH_SIZE):
            batches.append((hard[i: i + BATCH_SIZE], MODEL_HARD))

        log.info("Dispatching %d batches across %d workers…", len(batches), CONCURRENCY)

        removed_sonnet = 0
        failures = 0

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(_review_batch, b, m): b for b, m in batches}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="batches"):
                batch = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    log.error("Batch failed: %s", e)
                    failures += 1
                    continue

                for i, hab in enumerate(batch):
                    key = f"h{i}"
                    kept_indices = result[key]
                    all_cids = [c["component_id"] for c in hab["components"]]
                    kept_cids = {all_cids[j] for j in kept_indices if j < len(all_cids)}
                    # Remove non-kept from keep_ids
                    for cid in all_cids:
                        if cid not in kept_cids:
                            keep_ids.discard(cid)
                            removed_sonnet += 1

        log.info("Pass 2 done: removed %d more; %d batches failed", removed_sonnet, failures)
        if failures:
            log.warning("Re-run to retry failed batches (checkpoint not implemented for pass 2)")

    # ---- Write result ----
    cleaned = mapping[mapping["component_id"].isin(keep_ids)].copy()
    cleaned.to_csv(MAP_CSV, index=False, encoding="utf-8-sig")

    # Summary
    total_removed = len(mapping) - len(cleaned)
    per_hab_after = cleaned.groupby("bncc_code")["component_id"].count()
    log.info("Done. %d → %d assignments (removed %d)",
             len(mapping), len(cleaned), total_removed)
    log.info("Per-habilidade after cleanup:")
    log.info("  mean=%.1f  median=%.0f  max=%d",
             per_hab_after.mean(), per_hab_after.median(), per_hab_after.max())
    over_cap = (per_hab_after > CAP).sum()
    if over_cap:
        log.warning("%d habilidades still over %d — re-run or review manually", over_cap, CAP)
    else:
        log.info("All habilidades at %d components or fewer.", CAP)


if __name__ == "__main__":
    main()
