"""
Step 4: Fill learning-component gaps for every leaf-level CC standard.

LC already has components for 509 Math standards.  This step generates
components for the remaining 2,325 (ELA, Science, Math gaps) using Sonnet,
then writes a single unified table that downstream steps use as the component pool.

Inputs  (data/raw/):
  lc_cc_standards.csv     — all CC standards
  lc_components.csv       — existing LC component descriptions
  lc_cc_components.csv    — existing standard→component edges

Outputs (data/processed/):
  cc_components_full.csv  — (standard_id, component_id, description, source)
                            source = "lc_existing" | "ai_generated"
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
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
BATCH_SIZE = 20
CONCURRENCY = 8
MAX_TOKENS = 8192

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT / "cc_components_full.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step4_fill_cc_components.log", mode="w"),
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
    print("ERROR: no ANTHROPIC_API_KEY and no Claude Code session token", file=sys.stderr)
    sys.exit(1)

client = _make_client()

# ---------------------------------------------------------------------------
# Style anchors — 10 examples per subject
# Math: real Learning Commons components (source of truth)
# ELA:  representative samples from the AI-generated pass (same style)
# Science: hand-crafted at matching granularity (LC has no Science components)
# ---------------------------------------------------------------------------

ANCHORS: dict[str, list[str]] = {
    "Mathematics": [
        "Represent a multiple of a/b as a multiple of 1/b",
        "Use an understanding of a/b as a multiple of 1/b to multiply a fraction by a whole number",
        "Describe a sequence of transformations that exhibits congruence between two given congruent figures",
        "Find the conjugate of a complex number",
        "Use the conjugate of a complex number to find its modulus",
        "Prove the elimination method of solving systems of equations",
        "Design a simulation to generate frequencies for compound events",
        "Convert between forms of writing decimals to the thousandths",
        "Use place value understanding to round whole numbers to the nearest 10 or 100",
        "Fluently add and subtract within 1000 using strategies based on place value and properties of operations",
    ],
    "English Language Arts": [
        "Identify the main topic of an informational text with prompting and support",
        "Retell key details from an informational text with prompting and support",
        "Select relevant evidence from informational texts to support a specific analytical claim",
        "Integrate quoted or paraphrased textual evidence into research writing accurately",
        "Evaluate the strength of evidence drawn from informational texts for a given argument",
        "Identify the organizational structure a history or social studies text uses to present information",
        "Apply evidence from multiple informational sources to support reflection or research conclusions",
        "Distinguish between literal and non-literal language in context",
        "Determine the central idea of a passage and summarize it without personal opinion",
        "Use context clues to determine the meaning of unknown vocabulary in a text",
    ],
    "Science": [
        "Identify patterns in observational data to support a scientific explanation",
        "Construct a model to represent the relationship between variables in a system",
        "Analyze data to determine whether evidence supports a proposed causal mechanism",
        "Use mathematical representations to describe a natural phenomenon quantitatively",
        "Compare inherited traits of offspring with those of parents to distinguish genetic from environmental variation",
        "Evaluate competing design solutions against specified scientific criteria and constraints",
        "Plan and conduct an investigation to test a proposed cause-and-effect relationship",
        "Obtain and evaluate information from multiple sources to communicate scientific ideas",
        "Apply conservation of energy principles to predict outcomes in physical systems",
        "Distinguish between a scientific explanation and a claim not supported by empirical evidence",
    ],
}

_SYSTEM_TEMPLATE = """You are an expert curriculum specialist generating learning components for Common Core State Standards (CCSS).

A learning component is a single, atomic skill a student must master as part of a larger standard. The Learning Commons Knowledge Graph uses these components as the fundamental unit for comparing standards across curricula.

Rules for every component you write:
- Begin with a precise action verb (Identify, Apply, Explain, Construct, Analyze, Compare, Use, Solve, Interpret, etc.)
- Describe exactly ONE specific, testable skill — no compound skills joined by "and"
- Keep it to 5–20 words
- Use subject-appropriate technical vocabulary
- Match the granularity of these {subject} examples:

{examples}

Generate 3–5 components per standard — enough to fully cover mastery, no redundancy.

Return ONLY a JSON object mapping each statement_code to its list of component strings:
{{"CODE1": ["component a", "component b", ...], "CODE2": [...]}}"""


def _system_prompt(subject: str) -> str:
    examples = ANCHORS.get(subject, ANCHORS["Mathematics"])
    return _SYSTEM_TEMPLATE.format(
        subject=subject,
        examples="\n".join(f"  • {e}" for e in examples),
    )


def _user_message(batch: list[dict]) -> str:
    items = [
        {"statement_code": r["statement_code"], "description": r["description"]}
        for r in batch
    ]
    return (
        "Generate 3–5 learning components for each of the following CC standards.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _generate_batch(batch: list[dict]) -> dict[str, list[str]]:
    subject = batch[0].get("academic_subject", "Mathematics")
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(subject),
        messages=[{"role": "user", "content": _user_message(batch)}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"non-JSON response: {text[:300]!r}") from e

    missing = [r["statement_code"] for r in batch if r["statement_code"] not in result]
    if missing:
        raise ValueError(f"missing codes in response: {missing[:5]}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 4: Fill CC component gaps (model=%s) ===", MODEL)

    standards = pd.read_csv(RAW / "lc_cc_standards.csv", encoding="utf-8-sig")
    lc_components = pd.read_csv(RAW / "lc_components.csv", encoding="utf-8-sig")
    lc_edges = pd.read_csv(RAW / "lc_cc_components.csv", encoding="utf-8-sig")

    # Leaf standards only
    leaf = standards[standards["statement_type"] == "Standard"].copy()
    log.info("Leaf CC standards: %d", len(leaf))

    # --- Write existing LC components (source = lc_existing) ---
    existing_rows = (
        lc_edges
        .merge(lc_components.rename(columns={"identifier": "component_id"}),
               on="component_id", how="left")
        [["standard_id", "component_id", "description"]]
        .assign(source="lc_existing")
    )
    log.info("Existing LC components: %d", len(existing_rows))

    # --- Identify standards that need generation ---
    covered_ids = set(lc_edges["standard_id"].unique())
    need_gen = leaf[~leaf["identifier"].isin(covered_ids)].copy()
    log.info("Standards needing AI generation: %d", len(need_gen))

    # --- Resume: skip standards already in OUT_CSV ---
    done_standard_ids: set[str] = set()
    if OUT_CSV.exists():
        try:
            existing_out = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
            done_standard_ids = set(
                existing_out[existing_out["source"] == "ai_generated"]["standard_id"]
                .dropna().astype(str).tolist()
            )
            log.info("Resuming: %d standards already AI-generated", len(done_standard_ids))
        except pd.errors.EmptyDataError:
            pass

    todo = need_gen[~need_gen["identifier"].isin(done_standard_ids)].to_dict("records")
    log.info("Remaining to generate: %d standards", len(todo))

    # --- Write existing components on first run ---
    if not OUT_CSV.exists():
        existing_rows.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        log.info("Wrote %d existing LC components to %s", len(existing_rows), OUT_CSV.name)

    # --- Generate missing components in subject-ordered batches ---
    # Sort by subject so batches stay within one subject (better prompt coherence)
    todo_df = pd.DataFrame(todo).sort_values("academic_subject")
    batches = [
        todo_df.iloc[i : i + BATCH_SIZE].to_dict("records")
        for i in range(0, len(todo_df), BATCH_SIZE)
    ]
    log.info("Dispatching %d batches across %d workers…", len(batches), CONCURRENCY)

    failures: list[tuple[list[str], str]] = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(_generate_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="batches"):
            batch = futures[fut]
            try:
                mapping = fut.result()
            except Exception as e:
                codes = [r["statement_code"] for r in batch]
                log.error("Batch failed (%s…): %s", codes[0], e)
                failures.append((codes, str(e)))
                continue

            rows = []
            for std in batch:
                components = mapping.get(std["statement_code"], [])
                for desc in components:
                    rows.append({
                        "standard_id": std["identifier"],
                        "component_id": str(uuid.uuid4()),
                        "description": desc.strip(),
                        "source": "ai_generated",
                    })

            pd.DataFrame(rows).to_csv(
                OUT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
            )

    # --- Summary ---
    result = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    by_source = result["source"].value_counts()
    log.info("Final cc_components_full.csv: %d rows", len(result))
    log.info("  lc_existing  : %d", by_source.get("lc_existing", 0))
    log.info("  ai_generated : %d", by_source.get("ai_generated", 0))
    log.info("Standards covered: %d / %d leaf standards",
             result["standard_id"].nunique(), len(leaf))

    if failures:
        log.warning("%d batches failed — re-run to retry", len(failures))
    else:
        log.info("All standards processed successfully.")


if __name__ == "__main__":
    main()
