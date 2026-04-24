"""
Step 3: Translate BNCC habilidade descriptions from Portuguese to English.

Uses Claude Sonnet 4.6 — plenty capable for curriculum-standard translation,
and much faster/cheaper than Opus.  Opus 4.7 is reserved for steps 4+ where
pedagogical judgement matters more.

Input:  data/processed/bncc_standards.csv  (from step 1; 1,710 rows)
Output: data/processed/bncc_translated.csv (adds `description_en` column)

The script is restart-safe: on each batch it appends to the output CSV, so a
crash or Ctrl-C loses at most one batch.  Re-running resumes from the last
completed row.
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
BATCH_SIZE = 20
CONCURRENCY = 8
MAX_TOKENS = 8192

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "processed" / "bncc_standards.csv"
OUT_CSV = ROOT / "data" / "processed" / "bncc_translated.csv"
LOGS_DIR = ROOT / "data" / "logs"

# Auth: prefer API key; fall back to Claude Code subscription OAuth token
_SESSION_TOKEN_FILE = Path("/home/claude/.claude/remote/.session_ingress_token")

def _make_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return Anthropic(api_key=api_key)
    if _SESSION_TOKEN_FILE.exists():
        token = _SESSION_TOKEN_FILE.read_text().strip()
        return Anthropic(auth_token=token)
    print("ERROR: no ANTHROPIC_API_KEY and no Claude Code session token found", file=sys.stderr)
    sys.exit(1)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step3_translate.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

client = _make_client()

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional translator specializing in K-12 curriculum standards.

You are translating Brazilian BNCC habilidades (learning objectives from Brazil's national curriculum) from Portuguese to English.

Your target register is the style used in US K-12 curriculum standards documents (Common Core State Standards, Next Generation Science Standards). That means:

- Start each translation with a precise, grade-appropriate action verb that matches how CCSS/NGSS phrase objectives (e.g., "Identify", "Analyze", "Compare", "Interpret", "Solve", "Explain", "Describe", "Determine", "Evaluate", "Construct").
- Preserve the pedagogical intent and the specificity of the original — do not summarize, generalize, or simplify.
- Keep technical domain vocabulary accurate (mathematics, linguistics, arts, sciences, history, etc.). Use the conventional English term for each concept.
- Where a Portuguese sentence names a concrete Brazilian cultural, geographic, or historical referent (e.g., "Quilombolas", "Lei Áurea", "Cerrado"), keep the proper noun and, if helpful for an English reader, add a brief parenthetical gloss once per translation.
- Match the length and structure of the original. Do NOT add commentary, rationale, or notes.
- Return natural, idiomatic English — not a word-for-word calque.

You will receive items in JSON format with a code, subject, and stage for context. Return a JSON object mapping each code to its English translation. Return ONLY the JSON object — no prose, no markdown fences."""


def _build_user_message(batch: list[dict]) -> str:
    items = [
        {
            "code": r["bncc_code"],
            "stage": r["stage"],
            "grade": r["grade"],
            "subject": r["subject"],
            "description_pt": r["description_pt"],
        }
        for r in batch
    ]
    return (
        "Translate the `description_pt` of each of the following BNCC habilidades "
        "into English, following the style guide. Return a JSON object mapping "
        "each `code` to its English translation.\n\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _translate_batch(batch: list[dict]) -> dict[str, str]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(batch)}],
    )
    text = resp.content[0].text.strip()

    # Strip accidental markdown fences if the model adds them despite instructions
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        mapping = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"model returned non-JSON: {text[:300]!r}") from e

    if not isinstance(mapping, dict):
        raise ValueError(f"expected JSON object, got {type(mapping).__name__}")

    missing = [r["bncc_code"] for r in batch if r["bncc_code"] not in mapping]
    if missing:
        raise ValueError(f"missing translations for: {missing[:5]}…")

    return {k: str(v).strip() for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_already_done() -> set[str]:
    if not OUT_CSV.exists():
        return set()
    try:
        df = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return set()
    done = set(df["bncc_code"].dropna().astype(str).tolist())
    log.info("Resuming: %d translations already present in %s", len(done), OUT_CSV.name)
    return done


def _chunks(rows: list[dict], n: int) -> list[list[dict]]:
    return [rows[i : i + n] for i in range(0, len(rows), n)]


def _append_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    header = not OUT_CSV.exists()
    df.to_csv(OUT_CSV, mode="a", header=header, index=False, encoding="utf-8-sig")


def main() -> None:
    log.info("=== Step 3: Translate BNCC habilidades PT → EN (model=%s) ===", MODEL)

    df_in = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    log.info("Loaded %d rows from %s", len(df_in), IN_CSV.relative_to(ROOT))

    done = _load_already_done()
    todo = df_in[~df_in["bncc_code"].astype(str).isin(done)].to_dict("records")
    log.info("Remaining to translate: %d", len(todo))

    if not todo:
        log.info("Nothing to do — all rows translated.")
        return

    batches = _chunks(todo, BATCH_SIZE)
    log.info("Dispatching %d batches of up to %d across %d workers…",
             len(batches), BATCH_SIZE, CONCURRENCY)

    failures: list[tuple[list[str], str]] = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(_translate_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="batches"):
            batch = futures[fut]
            try:
                mapping = fut.result()
            except Exception as e:
                codes = [r["bncc_code"] for r in batch]
                log.error("Batch failed (%d codes, first=%s): %s", len(codes), codes[0], e)
                failures.append((codes, str(e)))
                continue

            rows = []
            for src in batch:
                rows.append({
                    "bncc_code": src["bncc_code"],
                    "stage": src["stage"],
                    "grade": src["grade"],
                    "subject": src["subject"],
                    "description_pt": src["description_pt"],
                    "description_en": mapping[src["bncc_code"]],
                })
            _append_rows(rows)

    # Summary
    df_out = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    log.info("Translated total: %d / %d", len(df_out), len(df_in))
    if failures:
        log.warning("Failed batches: %d  (see log for details)", len(failures))
        log.warning("Re-run the script to retry the missing codes.")
    else:
        log.info("All habilidades translated successfully.")


if __name__ == "__main__":
    main()
