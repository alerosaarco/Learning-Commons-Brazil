"""
Step 3: Translate BNCC habilidade descriptions from Portuguese to English.

Input:  data/processed/bncc_standards.csv   (from Step 1)
Output: data/processed/bncc_translated.csv  (same + description_en column)

Uses Claude claude-haiku-4-5-20251001 for cost efficiency (translation doesn't need Sonnet).
Processes in batches of 20. Resumes from partial output if interrupted.

Requires ANTHROPIC_API_KEY in .env or environment.
"""

import json
import logging
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_CSV = PROCESSED_DIR / "bncc_standards.csv"
OUTPUT_CSV = PROCESSED_DIR / "bncc_translated.csv"

BATCH_SIZE = 20
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are translating Brazilian educational standards (BNCC) to English for "
    "the purpose of aligning them with US Common Core standards. "
    "Translate accurately, preserving educational terminology. "
    "Use standard US math education vocabulary where applicable "
    "(e.g., 'grandezas e medidas' → 'measurement and data', "
    "'números naturais' → 'natural numbers', "
    "'figuras geométricas' → 'geometric figures'). "
    "Return ONLY a JSON array of translated strings, same order as input. "
    "No explanations, no markdown, just the JSON array."
)


def translate_batch(client: anthropic.Anthropic, descriptions: list[str]) -> list[str]:
    """Translate a batch of Portuguese descriptions to English."""
    user_msg = json.dumps(descriptions, ensure_ascii=False)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            translations = json.loads(raw)
            if len(translations) != len(descriptions):
                raise ValueError(
                    f"Got {len(translations)} translations for {len(descriptions)} inputs"
                )
            return translations

        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)

    # If all attempts fail, return empty strings (will be flagged for review)
    log.error("All translation attempts failed for batch. Returning empty strings.")
    return [""] * len(descriptions)


def main():
    log.info("=== Step 3: Translate BNCC Descriptions PT→EN ===")

    if not INPUT_CSV.exists():
        log.error("Input not found: %s — run Step 1 first.", INPUT_CSV)
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set. Add it to .env.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(INPUT_CSV)
    log.info("Loaded %d BNCC standards.", len(df))

    # Resume support: if output already exists, skip already-translated rows
    if OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV)
        done_codes = set(done_df["bncc_code"].tolist())
        log.info("Resuming: %d already translated.", len(done_codes))
    else:
        done_df = pd.DataFrame()
        done_codes = set()

    todo = df[~df["bncc_code"].isin(done_codes)].copy()
    log.info("%d remaining to translate.", len(todo))

    if todo.empty:
        log.info("All done. Output: %s", OUTPUT_CSV)
        return

    results = []
    descriptions = todo["description_pt"].tolist()

    for i in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Translating"):
        batch_desc = descriptions[i : i + BATCH_SIZE]
        batch_en = translate_batch(client, batch_desc)
        results.extend(batch_en)
        # Small pause to respect rate limits
        time.sleep(0.5)

    todo = todo.copy()
    todo["description_en"] = results

    # Merge with any already-translated rows
    combined = pd.concat([done_df, todo], ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d translated standards → %s", len(combined), OUTPUT_CSV)

    # Flag any empty translations
    missing = combined[combined["description_en"].fillna("") == ""]
    if not missing.empty:
        log.warning(
            "%d translations are empty and need review:\n%s",
            len(missing),
            missing["bncc_code"].tolist(),
        )


if __name__ == "__main__":
    main()
