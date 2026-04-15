"""
Step 1: Download and parse BNCC EF Math habilidades.

Strategy:
  1. Download the official MEC PDF (Ensino Fundamental)
  2. Parse with pdfplumber, using regex to locate habilidade blocks
  3. Output: data/processed/bncc_standards.csv

BNCC code structure:  EF 05 MA 01
  EF  = Ensino Fundamental
  05  = 5th grade (01-09)
  MA  = Matemática
  01  = sequential skill number
"""

import re
import sys
import csv
import json
import logging
from pathlib import Path

import requests
import pdfplumber
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PDF_URL = (
    "https://basenacionalcomum.mec.gov.br"
    "/images/BNCC_EI_EF_110518_versaofinal_site.pdf"
)
PDF_PATH = RAW_DIR / "BNCC_EI_EF.pdf"
OUTPUT_CSV = PROCESSED_DIR / "bncc_standards.csv"

# Only extract Ensino Fundamental Math
TARGET_STAGE = "EF"
TARGET_SUBJECT = "MA"

# ---------------------------------------------------------------------------
# BNCC code regex — matches EF01MA01 through EF09MA99
# ---------------------------------------------------------------------------
BNCC_CODE_RE = re.compile(
    r"\b(EF\d{2}MA\d{2,3})\b"
)

# Thematic units for EF Math (used to detect section boundaries)
THEMATIC_UNITS = [
    "Números",
    "Álgebra",
    "Geometria",
    "Grandezas e Medidas",
    "Probabilidade e Estatística",
]

# Grade labels used in the BNCC document
GRADE_LABELS = {
    "01": "1º ano",
    "02": "2º ano",
    "03": "3º ano",
    "04": "4º ano",
    "05": "5º ano",
    "06": "6º ano",
    "07": "7º ano",
    "08": "8º ano",
    "09": "9º ano",
}


# ---------------------------------------------------------------------------
# Step 1a: Download PDF
# ---------------------------------------------------------------------------
def download_pdf() -> Path:
    if PDF_PATH.exists():
        log.info("PDF already downloaded: %s", PDF_PATH)
        return PDF_PATH

    log.info("Downloading BNCC PDF from MEC...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
        "Referer": "https://basenacionalcomum.mec.gov.br/",
    }

    with requests.get(PDF_URL, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(PDF_PATH, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="PDF download"
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    log.info("Saved: %s (%.1f MB)", PDF_PATH, PDF_PATH.stat().st_size / 1e6)
    return PDF_PATH


# ---------------------------------------------------------------------------
# Step 1b: Extract raw text blocks from PDF pages
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> list[str]:
    """Return list of page texts from the entire PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Extracting PDF pages"):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages.append(text)
    return pages


# ---------------------------------------------------------------------------
# Step 1c: Parse habilidades from extracted text
# ---------------------------------------------------------------------------
def parse_habilidades(pages: list[str]) -> list[dict]:
    """
    Strategy:
      - Concatenate all page text into one big string (preserving newlines).
      - Find all occurrences of BNCC Math codes (EF\d{2}MA\d{2,3}).
      - For each occurrence, extract:
          * The code itself
          * The description: text following the code until the next code or
            a double-newline block, cleaned of page artefacts
          * Thematic unit: last seen THEMATIC_UNIT header before the code
          * Knowledge object: paragraph header between thematic unit and code
          * Grade: derived from the code digits
    """
    full_text = "\n".join(pages)

    # Normalise: collapse runs of whitespace (but keep newlines)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)

    # --- Find all code positions ---
    matches = list(BNCC_CODE_RE.finditer(full_text))
    if not matches:
        log.warning("No BNCC Math codes found in PDF text.")
        return []

    log.info("Found %d BNCC Math code occurrences in PDF.", len(matches))

    # Deduplicate: some codes appear in tables AND in body text.
    # Keep the last occurrence (body text is usually after summary tables).
    code_to_last_match: dict[str, re.Match] = {}
    for m in matches:
        code_to_last_match[m.group(1)] = m

    habilidades: list[dict] = []

    # Sort by position in text
    sorted_matches = sorted(code_to_last_match.values(), key=lambda m: m.start())

    for i, m in enumerate(sorted_matches):
        code = m.group(1)
        grade_num = code[2:4]  # e.g. "05"
        grade = GRADE_LABELS.get(grade_num, grade_num)

        # --- Description: text after the code until next code or blank line ---
        start = m.end()
        if i + 1 < len(sorted_matches):
            end = sorted_matches[i + 1].start()
        else:
            end = start + 2000  # last entry — take generous window

        raw_desc = full_text[start:end]
        description = _clean_description(raw_desc)

        # --- Thematic unit: last THEMATIC_UNIT string before this code ---
        thematic_unit = _find_last_before(full_text, THEMATIC_UNITS, m.start())

        # --- Knowledge object: line(s) between thematic unit and code ---
        knowledge_object = _extract_knowledge_object(full_text, m.start(), thematic_unit)

        habilidades.append(
            {
                "bncc_code": code,
                "stage": "EF",
                "grade": grade,
                "subject": "Matemática",
                "subject_code": "MA",
                "description_pt": description,
                "thematic_unit": thematic_unit,
                "knowledge_object": knowledge_object,
            }
        )

    return habilidades


def _clean_description(raw: str) -> str:
    """Clean raw text following a BNCC code into a readable description."""
    # Remove leading punctuation / bullets the code was part of
    raw = raw.lstrip("() \n\t–-")

    # Take up to 3 lines (descriptions are typically 1-2 lines)
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    # Stop at lines that look like a new section header or next entry
    result_lines = []
    for ln in lines[:8]:
        # Stop if line looks like a page number, header, or thematic section
        if re.match(r"^\d{1,3}$", ln):  # lone page number
            break
        if any(ln.startswith(t) for t in THEMATIC_UNITS):
            break
        if re.match(r"^EF\d{2}(MA|LP|CI|GE|HI)", ln):  # next code
            break
        result_lines.append(ln)
        # Description sentences usually end with period or semicolon
        if ln.endswith((".", ";", ")", "!")):
            break

    description = " ".join(result_lines)
    # Normalise internal whitespace
    description = re.sub(r"\s+", " ", description).strip()
    return description


def _find_last_before(text: str, candidates: list[str], pos: int) -> str:
    """Return whichever candidate string appears closest before `pos`."""
    best_pos = -1
    best_candidate = ""
    for c in candidates:
        idx = text.rfind(c, 0, pos)
        if idx > best_pos:
            best_pos = idx
            best_candidate = c
    return best_candidate


def _extract_knowledge_object(text: str, code_pos: int, thematic_unit: str) -> str:
    """
    Extract the knowledge object paragraph:
    the block of text between the thematic unit header and the BNCC code.
    """
    if not thematic_unit:
        return ""

    # Find the last occurrence of the thematic unit before the code
    tu_pos = text.rfind(thematic_unit, 0, code_pos)
    if tu_pos == -1:
        return ""

    segment = text[tu_pos + len(thematic_unit) : code_pos]
    lines = [ln.strip() for ln in segment.split("\n") if ln.strip()]

    # Filter out lines that are noise (numbers, short fragments, grade headers)
    candidates = []
    for ln in lines:
        if re.match(r"^\d{1,3}$", ln):
            continue
        if len(ln) < 5:
            continue
        if re.match(r"^EF\d{2}MA", ln):
            continue
        candidates.append(ln)

    if not candidates:
        return ""

    # The knowledge object is usually the last substantive line before the code
    return candidates[-1]


# ---------------------------------------------------------------------------
# Step 1d: Validate and save
# ---------------------------------------------------------------------------
def save_results(habilidades: list[dict]) -> Path:
    if not habilidades:
        log.error("No habilidades parsed — check PDF extraction above.")
        sys.exit(1)

    df = pd.DataFrame(habilidades)
    # Sort by grade then sequential number
    df["_sort_key"] = df["bncc_code"].apply(
        lambda c: (int(c[2:4]), int(re.sub(r"\D", "", c[6:])) if len(c) > 6 else 0)
    )
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d habilidades → %s", len(df), OUTPUT_CSV)

    # Quick stats
    by_grade = df["grade"].value_counts().sort_index()
    log.info("Habilidades per grade:\n%s", by_grade.to_string())

    return OUTPUT_CSV


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Step 1: Parse BNCC EF Math Habilidades ===")

    pdf_path = download_pdf()
    log.info("Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)
    log.info("Extracted text from %d pages.", len(pages))

    log.info("Parsing habilidades...")
    habilidades = parse_habilidades(pages)
    log.info("Parsed %d unique habilidades.", len(habilidades))

    out = save_results(habilidades)
    log.info("Step 1 complete. Output: %s", out)

    # Preview first 5
    df = pd.read_csv(out)
    print("\nSample output:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
