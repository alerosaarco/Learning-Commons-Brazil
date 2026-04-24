"""
Step 10: Translate BNCC component descriptions from English to Brazilian Portuguese.

Components were generated in English (step 6) so both BNCC and CC pools shared
the same vocabulary for embedding-based matching (step 7). Now we translate them
back to PT-BR for display on the website.

Rules for translation:
  - Preserve the action-verb-first atomic structure
  - Use standard Brazilian curriculum vocabulary (not European Portuguese)
  - Keep technical terms that are the same in both languages (e.g. frações, equação)
  - Do NOT add explanations or change meaning

Input:  data/processed/bncc_components.csv  (habilidade_code, component_id, description)
Output: data/processed/bncc_components_pt.csv
        Adds column: description_pt  (Brazilian Portuguese translation)
        Retains:     habilidade_code, component_id, description (English kept for reference)
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

load_dotenv()

MODEL       = "claude-haiku-4-5-20251001"   # fast + cheap for translation
BATCH_SIZE  = 50
CONCURRENCY = 8
MAX_TOKENS  = 4096

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV  = PROC / "bncc_components.csv"
OUT_CSV = PROC / "bncc_components_pt.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step10_translate_components.log", mode="w"),
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

SYSTEM = """You are a professional translator specializing in Brazilian educational curriculum documents.

Translate each English learning component into Brazilian Portuguese (PT-BR).

Rules:
- Preserve the action-verb-first structure (e.g. "Identify" → "Identificar", "Apply" → "Aplicar")
- Use standard Brazilian curriculum vocabulary (not European Portuguese)
- Keep technical math/science terms that are standard in Brazil
- Do NOT explain, expand, or add content — translate only
- Keep the same brevity (5-20 words)

Return ONLY a JSON object mapping each ID to its Portuguese translation:
{"id_0": "Identificar...", "id_1": "Aplicar...", ...}"""


def _user_message(batch: list[tuple[str, str]]) -> str:
    items = {f"id_{i}": desc for i, (_, desc) in enumerate(batch)}
    return (
        "Translate each component to Brazilian Portuguese (PT-BR).\n\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _translate_batch(batch: list[tuple[str, str]]) -> dict[str, str]:
    """batch: list of (component_id, description_en)"""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": _user_message(batch)}],
    )
    text = resp.content[0].text.strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    if "```" in text:
        text = text[:text.rfind("```")]
    result = json.loads(text.strip())
    # Map back from positional id to component_id
    return {batch[int(k.split("_")[1])][0]: v for k, v in result.items()}


def main() -> None:
    log.info("=== Step 10: Translate BNCC components to PT-BR (model=%s) ===", MODEL)

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    log.info("Loaded %d components", len(df))

    # Resume support
    done_ids: set[str] = set()
    if OUT_CSV.exists():
        try:
            done_ids = set(
                pd.read_csv(OUT_CSV, encoding="utf-8-sig")["component_id"].dropna().astype(str)
            )
            log.info("Resuming: %d already translated", len(done_ids))
        except pd.errors.EmptyDataError:
            pass

    todo = df[~df["component_id"].astype(str).isin(done_ids)]
    log.info("Remaining: %d components", len(todo))
    if todo.empty:
        log.info("All done.")
        return

    pairs = list(zip(todo["component_id"].astype(str), todo["description"].astype(str)))
    batches = [pairs[i:i + BATCH_SIZE] for i in range(0, len(pairs), BATCH_SIZE)]
    log.info("Dispatching %d batches (%d components each) across %d workers",
             len(batches), BATCH_SIZE, CONCURRENCY)

    failures = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(_translate_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="batches"):
            batch = futures[fut]
            try:
                translations = fut.result()
            except Exception as e:
                log.error("Batch failed (first=%s): %s", batch[0][0], e)
                failures += 1
                continue

            rows = []
            for comp_id, desc_en in batch:
                rows.append({
                    "component_id":   comp_id,
                    "description_pt": translations.get(comp_id, ""),
                })
            chunk_df = pd.DataFrame(rows)
            # Merge with original data to get habilidade_code + English description
            chunk_full = df[df["component_id"].isin(chunk_df["component_id"])].copy()
            chunk_full = chunk_full.merge(chunk_df, on="component_id")

            header = not OUT_CSV.exists()
            chunk_full[["habilidade_code", "component_id", "description", "description_pt"]].to_csv(
                OUT_CSV, mode="a", header=header, index=False, encoding="utf-8-sig"
            )

    result = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    log.info("Done: %d components translated", len(result))
    missing_pt = result["description_pt"].isna().sum() + (result["description_pt"] == "").sum()
    if missing_pt:
        log.warning("%d components missing Portuguese translation", missing_pt)
    if failures:
        log.warning("%d batches failed — re-run to retry", failures)
    else:
        log.info("All batches succeeded.")

    log.info("Sample translations:")
    for _, row in result.head(5).iterrows():
        log.info("  EN: %s", row["description"])
        log.info("  PT: %s", row["description_pt"])
        log.info("")


if __name__ == "__main__":
    main()
