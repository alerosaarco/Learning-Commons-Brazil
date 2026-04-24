"""
Step 6: Generate 3-5 learning components for every BNCC habilidade.

Mirrors step 4 exactly — each habilidade is decomposed into atomic sub-skills
using LC-style language (action verb + specific skill, 5-20 words).  CC
components from the same subject are used as style anchors so both pools share
the same grammatical register, enabling embedding-based matching in step 7.

Input:  data/processed/bncc_translated.csv   (1,710 habilidades)
        data/processed/cc_components_full.csv (style anchors)
Output: data/processed/bncc_components.csv
        columns: habilidade_code, component_id, description
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

load_dotenv()

MODEL       = "claude-sonnet-4-6"
BATCH_SIZE  = 20
CONCURRENCY = 8
MAX_TOKENS  = 8192

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = PROC / "bncc_components.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step6_generate_bncc_components.log", mode="w"),
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
# BNCC subject → CC subject anchor pool
# ---------------------------------------------------------------------------

BNCC_TO_CC_SUBJECT: dict[str, str] = {
    "Matemática":                              "Mathematics",
    "Matemática e suas Tecnologias":           "Mathematics",
    "Espaços, Tempos, Quantidades, Relações e Transformações": "Mathematics",
    "Língua Portuguesa":                       "English Language Arts",
    "Linguagens e suas Tecnologias":           "English Language Arts",
    "Escuta, Fala, Pensamento e Imaginação":   "English Language Arts",
    "Língua Inglesa":                          "English Language Arts",
    "Ciências":                                "Science",
    "Ciências da Natureza e suas Tecnologias": "Science",
    # Subjects with no direct CC counterpart → use ELA anchors (closest register)
    "História":                                "English Language Arts",
    "Geografia":                               "Science",
    "Ciências Humanas e Sociais Aplicadas":    "English Language Arts",
    "Arte":                                    "English Language Arts",
    "Educação Física":                         "Science",
    "Ensino Religioso":                        "English Language Arts",
    "Computação":                              "Science",
}

_SYSTEM_TEMPLATE = """You are an expert curriculum specialist decomposing Brazilian BNCC habilidades into atomic learning components.

A learning component is a single, atomic skill a student must master as part of a larger standard. Use the same style as Learning Commons components for US Common Core standards.

Rules for every component you write:
- Begin with a precise action verb (Identify, Apply, Explain, Construct, Analyze, Compare, Use, Solve, Interpret, Evaluate, Generate, Describe, etc.)
- Describe exactly ONE specific, testable skill — no compound skills joined by "and"
- Keep it to 5–20 words
- Use subject-appropriate technical vocabulary
- Stay faithful to the specific skill described in the habilidade — do not add scope beyond what the habilidade requires
- Match this style exactly:

{anchors}

Generate 3–5 components per habilidade — enough to fully cover its scope, with no redundancy.

Return ONLY a JSON object mapping each habilidade code to its list of component strings:
{{"EF01MA01": ["component a", "component b", ...], "EF01MA02": [...]}}"""


def _build_system_prompt(anchor_descs: list[str]) -> str:
    bullets = "\n".join(f"  • {d}" for d in anchor_descs[:12])
    return _SYSTEM_TEMPLATE.format(anchors=bullets)


def _user_message(batch: list[dict]) -> str:
    items = [
        {"code": r["bncc_code"], "description": r["description_en"]}
        for r in batch
    ]
    return (
        "Decompose each BNCC habilidade into 3–5 atomic learning components.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _generate_batch(batch: list[dict], system_prompt: str) -> dict[str, list[str]]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
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

    missing = [r["bncc_code"] for r in batch if r["bncc_code"] not in result]
    if missing:
        raise ValueError(f"missing keys: {missing[:5]}")

    return result


def main() -> None:
    log.info("=== Step 6: Generate BNCC components (model=%s) ===", MODEL)

    bncc  = pd.read_csv(PROC / "bncc_translated.csv",   encoding="utf-8-sig")
    comps = pd.read_csv(PROC / "cc_components_full.csv", encoding="utf-8-sig")

    # Build per-CC-subject anchor pools (sample 12 diverse components)
    cc_anchors: dict[str, list[str]] = {}
    for cc_subj in ("Mathematics", "English Language Arts", "Science"):
        pool = comps[comps["source"] == "lc_existing"]["description"].tolist()
        if not pool:
            pool = comps["description"].tolist()
        cc_anchors[cc_subj] = pool[:12]

    # Resume
    done_codes: set[str] = set()
    if OUT_CSV.exists():
        try:
            done_codes = set(
                pd.read_csv(OUT_CSV, encoding="utf-8-sig")["habilidade_code"]
                .dropna().astype(str)
            )
            log.info("Resuming: %d habilidades already done", len(done_codes))
        except pd.errors.EmptyDataError:
            pass

    todo = bncc[~bncc["bncc_code"].astype(str).isin(done_codes)].to_dict("records")
    log.info("Remaining: %d habilidades", len(todo))
    if not todo:
        log.info("All done.")
        return

    # Build per-subject system prompts
    system_prompts: dict[str, str] = {
        cc_subj: _build_system_prompt(anchors)
        for cc_subj, anchors in cc_anchors.items()
    }

    # Sort by BNCC subject for prompt coherence; build batches
    todo_df = pd.DataFrame(todo)
    todo_df["_cc_subject"] = todo_df["subject"].map(BNCC_TO_CC_SUBJECT).fillna("English Language Arts")
    todo_df = todo_df.sort_values("_cc_subject").reset_index(drop=True)

    batches: list[tuple[list[dict], str]] = []
    for i in range(0, len(todo_df), BATCH_SIZE):
        chunk = todo_df.iloc[i: i + BATCH_SIZE].to_dict("records")
        cc_subj = chunk[0]["_cc_subject"]
        batches.append((chunk, system_prompts[cc_subj]))

    log.info("Dispatching %d batches across %d workers…", len(batches), CONCURRENCY)
    failures = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(_generate_batch, b, sp): b for b, sp in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="batches"):
            batch = futures[fut]
            try:
                mapping = fut.result()
            except Exception as e:
                log.error("Batch failed (first=%s): %s", batch[0]["bncc_code"], e)
                failures += 1
                continue

            rows = []
            for hab in batch:
                code = hab["bncc_code"]
                for desc in mapping.get(code, []):
                    rows.append({
                        "habilidade_code": code,
                        "component_id":    str(uuid.uuid4()),
                        "description":     desc.strip(),
                    })

            df = pd.DataFrame(rows)
            header = not OUT_CSV.exists()
            df.to_csv(OUT_CSV, mode="a", header=header, index=False, encoding="utf-8-sig")

    result = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    log.info("Done: %d components across %d habilidades", len(result), result["habilidade_code"].nunique())
    counts = result.groupby("habilidade_code").size()
    log.info("Per-habilidade: mean=%.1f  median=%.0f  min=%d  max=%d",
             counts.mean(), counts.median(), counts.min(), counts.max())
    if failures:
        log.warning("%d batches failed — re-run to retry", failures)
    else:
        log.info("All habilidades processed successfully.")


if __name__ == "__main__":
    main()
