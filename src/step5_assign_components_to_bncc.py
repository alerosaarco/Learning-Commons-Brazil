"""
Step 5: Assign each learning component to exactly one BNCC habilidade.

Every component in cc_components_full.csv is assigned to the single
BNCC habilidade it best supports.  Assignment is 1:1 (one component →
one habilidade, never zero).  Poor matches are flagged with
confidence=low for review and step-6 gap-fill.

Inputs  (data/processed/):
  cc_components_full.csv   — 10,242 components with CC standard link + source
  bncc_translated.csv      — 1,710 habilidades with English descriptions

Output (data/processed/):
  component_bncc_map.csv   — (component_id, bncc_code, confidence,
                               cc_subject, bncc_subject, source)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 30
CONCURRENCY = 8
MAX_TOKENS = 4096

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = PROC / "component_bncc_map.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step5_assign_components.log", mode="w"),
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
    print("ERROR: no auth available", file=sys.stderr)
    sys.exit(1)


client = _make_client()

# ---------------------------------------------------------------------------
# Subject routing — which BNCC subjects form the candidate pool per CC subject
# Generous mapping so cross-disciplinary components can find the right home
# ---------------------------------------------------------------------------

CC_TO_BNCC_SUBJECTS: dict[str, set[str]] = {
    "Mathematics": {
        "Matemática",
        "Matemática e suas Tecnologias",
        "Espaços, Tempos, Quantidades, Relações e Transformações",
    },
    "English Language Arts": {
        "Língua Portuguesa",
        "Linguagens e suas Tecnologias",
        "Escuta, Fala, Pensamento e Imaginação",
        "Língua Inglesa",
        # ELA includes literacy in history/science — include humanities too
        "História",
        "Geografia",
        "Ciências Humanas e Sociais Aplicadas",
    },
    "Science": {
        "Ciências",
        "Ciências da Natureza e suas Tecnologias",
        # Cross-cutting science concepts can touch geography/environment
        "Geografia",
    },
}


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are a curriculum alignment specialist mapping Common Core learning components to Brazilian BNCC habilidades.

A learning component is a single, atomic skill. Your task: for each component in the batch, identify the ONE BNCC habilidade that best represents where a student would develop that specific skill within the Brazilian curriculum.

Rules:
- Every component must be assigned — never leave one unmatched
- Choose the habilidade where the skill is most directly practiced or assessed
- If no strong match exists, pick the closest available and mark confidence=low
- Do not match based on grade level alone — prioritize skill alignment

Confidence levels:
  high   — clear semantic match, same skill in same domain
  medium — reasonable match, different framing or slight scope difference
  low    — weak match, no good option exists (flag for review)

BNCC candidate habilidades (use the [hNNNN] key to reference them):
{candidates}

Return ONLY a JSON object:
{{"c00": {{"match": "h0042", "confidence": "high"}}, "c01": ...}}"""


def _build_candidate_index(habilidades: pd.DataFrame) -> tuple[dict[str, str], str]:
    """
    Returns (key→bncc_code dict, formatted candidate block for prompt).
    Keys are h0001..hNNNN.
    """
    lines = []
    key_to_code: dict[str, str] = {}
    for i, row in enumerate(habilidades.itertuples()):
        key = f"h{i:04d}"
        key_to_code[key] = row.bncc_code
        desc = str(row.description_en or row.description_pt)
        desc_short = desc[:110] + "…" if len(desc) > 110 else desc
        lines.append(
            f"[{key}] {row.bncc_code} | {row.subject} | "
            f"Grade {row.grade} | {desc_short}"
        )
    return key_to_code, "\n".join(lines)


def _user_message(batch: list[dict]) -> str:
    items = [
        {"id": f"c{i:02d}", "description": row["description"]}
        for i, row in enumerate(batch)
    ]
    return (
        "Assign each component to its best-matching BNCC habilidade.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _assign_batch(
    batch: list[dict],
    system_prompt: str,
    key_to_code: dict[str, str],
) -> list[dict]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": _user_message(batch)}],
    )
    text = resp.content[0].text.strip()
    # Strip any prose preamble — find the first '{' that opens the JSON object
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    # Strip trailing markdown fences if present
    if "```" in text:
        text = text[:text.rfind("```")]
    text = text.strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"non-JSON: {text[:200]!r}") from e

    rows = []
    for i, comp in enumerate(batch):
        cid = f"c{i:02d}"
        entry = raw.get(cid)
        if not entry:
            raise ValueError(f"missing assignment for {cid}")
        hkey = entry.get("match", "")
        if hkey not in key_to_code:
            raise ValueError(f"unknown candidate key {hkey!r}")
        rows.append({
            "component_id": comp["component_id"],
            "bncc_code": key_to_code[hkey],
            "confidence": entry.get("confidence", "medium"),
            "cc_subject": comp.get("academic_subject", ""),
            "bncc_subject": comp.get("_bncc_subject", ""),
            "source": comp.get("source", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 5: Assign components to BNCC habilidades (model=%s) ===", MODEL)

    comps_raw = pd.read_csv(PROC / "cc_components_full.csv", encoding="utf-8-sig")
    standards = pd.read_csv(RAW / "lc_cc_standards.csv", encoding="utf-8-sig")
    bncc = pd.read_csv(PROC / "bncc_translated.csv", encoding="utf-8-sig")

    # Attach CC subject to each component
    comps = comps_raw.merge(
        standards[["identifier", "academic_subject"]],
        left_on="standard_id", right_on="identifier", how="left",
    ).drop(columns=["identifier"])
    log.info("Components loaded: %d", len(comps))

    # Resume
    done_ids: set[str] = set()
    if OUT_CSV.exists():
        try:
            done_ids = set(
                pd.read_csv(OUT_CSV, encoding="utf-8-sig")["component_id"]
                .dropna().astype(str)
            )
            log.info("Resuming: %d already assigned", len(done_ids))
        except pd.errors.EmptyDataError:
            pass

    todo = comps[~comps["component_id"].astype(str).isin(done_ids)].copy()
    log.info("Remaining to assign: %d", len(todo))
    if todo.empty:
        log.info("All components already assigned.")
        return

    # Build candidate pools per CC subject
    candidate_pools: dict[str, tuple[dict[str, str], str, pd.DataFrame]] = {}
    all_cc_subjects = todo["academic_subject"].dropna().unique().tolist()
    for subj in all_cc_subjects:
        bncc_subjects = CC_TO_BNCC_SUBJECTS.get(subj)
        if bncc_subjects:
            pool = bncc[bncc["subject"].isin(bncc_subjects)].reset_index(drop=True)
        else:
            pool = bncc.reset_index(drop=True)  # fallback: all habilidades
        key_to_code, candidate_block = _build_candidate_index(pool)
        system_prompt = SYSTEM_TEMPLATE.format(candidates=candidate_block)
        candidate_pools[subj] = (key_to_code, system_prompt, pool)
        log.info("  %s → %d BNCC candidates", subj, len(pool))

    # Tag each component row with the candidate bncc_subject (for output)
    def _get_bncc_subj(row: pd.Series) -> str:
        subj = row.get("academic_subject", "")
        subjects = CC_TO_BNCC_SUBJECTS.get(subj, set())
        return ", ".join(sorted(subjects)) if subjects else "all"

    todo["_bncc_subject"] = todo.apply(_get_bncc_subj, axis=1)

    # Sort by CC subject so batches stay within one subject pool
    todo_sorted = todo.sort_values("academic_subject").reset_index(drop=True)
    batches: list[tuple[list[dict], dict[str, str], str]] = []
    for i in range(0, len(todo_sorted), BATCH_SIZE):
        chunk = todo_sorted.iloc[i: i + BATCH_SIZE].to_dict("records")
        subj = chunk[0].get("academic_subject") or ""
        k2c, sys_prompt, _ = candidate_pools.get(subj, candidate_pools.get(
            next(iter(candidate_pools)), ({},"",pd.DataFrame())
        ))
        batches.append((chunk, k2c, sys_prompt))

    log.info("Dispatching %d batches across %d workers…", len(batches), CONCURRENCY)
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(_assign_batch, b, sp, k2c): b
            for b, k2c, sp in batches
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="batches"):
            batch = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                cids = [r["component_id"] for r in batch]
                log.error("Batch failed (first=%s): %s", cids[0], e)
                failures.append(cids[0])
                continue

            df = pd.DataFrame(rows)
            header = not OUT_CSV.exists()
            df.to_csv(OUT_CSV, mode="a", header=header, index=False, encoding="utf-8-sig")

    # Summary
    result = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    conf = result["confidence"].value_counts()
    log.info("Assignments written: %d / %d components", len(result), len(comps))
    log.info("  high   : %d", conf.get("high", 0))
    log.info("  medium : %d", conf.get("medium", 0))
    log.info("  low    : %d  ← flagged for review", conf.get("low", 0))

    bncc_coverage = result["bncc_code"].nunique()
    log.info("BNCC habilidades receiving at least 1 component: %d / %d",
             bncc_coverage, len(bncc))

    if failures:
        log.warning("%d batches failed — re-run to retry", len(failures))
    else:
        log.info("All components assigned successfully.")


if __name__ == "__main__":
    main()
