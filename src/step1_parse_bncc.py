"""
Step 1: Parse ALL BNCC standards (EI + EF + EM, all subjects).

Input PDFs (drop in data/raw/):
  - BNCC_EI_EF.pdf   (Educação Infantil + Ensino Fundamental)
       https://basenacionalcomum.mec.gov.br/images/BNCC_EI_EF_110518_versaofinal_site.pdf
  - BNCC_EM.pdf       (Ensino Médio)
       https://basenacionalcomum.mec.gov.br/images/BNCC_20dez_site.pdf

Output:
  data/processed/bncc_standards.csv
  data/logs/step1_parsing.log
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pdfplumber
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
for d in (RAW_DIR, PROCESSED_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS_DIR / "step1_parsing.log"
OUTPUT_CSV = PROCESSED_DIR / "bncc_standards.csv"
PDF_EI_EF = RAW_DIR / "BNCC_EI_EF.pdf"
PDF_EM = RAW_DIR / "BNCC_EM.pdf"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code patterns — capture stage / grade / subject / sequential
# ---------------------------------------------------------------------------

CODE_RE = re.compile(
    r"\b(EI|EF|EM)(\d{2})([A-Z]{2,3})(\d{2,3})\b"
)

# Valid subject codes per stage
EI_SUBJECTS = {"EO", "CG", "TS", "EF", "ET"}
EF_SUBJECTS = {"MA", "LP", "CI", "GE", "HI", "AR", "EF", "ER"}
EM_SUBJECTS = {"LGG", "MAT", "CNT", "CHS", "LP", "LI", "ART", "EDF"}

# Subject code → full name.  EF is context-dependent, handled separately.
SUBJECT_NAMES = {
    # EF / fundamental
    "MA": "Matemática",
    "LP": "Língua Portuguesa",
    "CI": "Ciências",
    "GE": "Geografia",
    "HI": "História",
    "AR": "Arte",
    "ER": "Ensino Religioso",
    # EI fields of experience
    "EO": "O eu, o outro e o nós",
    "CG": "Corpo, Gestos e Movimentos",
    "TS": "Traços, Sons, Cores e Formas",
    "ET": "Espaços, Tempos, Quantidades, Relações e Transformações",
    # EM areas
    "LGG": "Linguagens e suas Tecnologias",
    "MAT": "Matemática e suas Tecnologias",
    "CNT": "Ciências da Natureza e suas Tecnologias",
    "CHS": "Ciências Humanas e Sociais Aplicadas",
    "LI": "Língua Inglesa",
    "ART": "Arte",
    "EDF": "Educação Física",
}


def subject_full_name(stage: str, subj_code: str) -> str:
    if subj_code == "EF":
        return (
            "Escuta, Fala, Pensamento e Imaginação"
            if stage == "EI"
            else "Educação Física"
        )
    return SUBJECT_NAMES.get(subj_code, subj_code)


# Grade labels
EI_GRADES = {
    "01": "Bebês (0–1a6m)",
    "02": "Crianças bem pequenas (1a7m–3a11m)",
    "03": "Crianças pequenas (4a–5a11m)",
}
EF_GRADES = {
    "01": "1º ano", "02": "2º ano", "03": "3º ano", "04": "4º ano",
    "05": "5º ano", "06": "6º ano", "07": "7º ano", "08": "8º ano",
    "09": "9º ano",
    "15": "1º–5º ano", "35": "3º–5º ano", "69": "6º–9º ano",
}
EM_GRADES = {"13": "1º–3º ano (Ensino Médio)"}


def grade_label(stage: str, grade_digits: str) -> str:
    if stage == "EI":
        return EI_GRADES.get(grade_digits, grade_digits)
    if stage == "EF":
        return EF_GRADES.get(grade_digits, grade_digits)
    if stage == "EM":
        return EM_GRADES.get(grade_digits, grade_digits)
    return grade_digits


# Context-header patterns (thematic unit / knowledge object / etc.)
THEMATIC_HEADERS = [
    "UNIDADES TEMÁTICAS",
    "UNIDADE TEMÁTICA",
    "CAMPOS DE EXPERIÊNCIAS",
    "CAMPO DE EXPERIÊNCIAS",
    "COMPETÊNCIAS ESPECÍFICAS",
    "COMPETÊNCIA ESPECÍFICA",
]
KNOWLEDGE_HEADERS = [
    "OBJETOS DE CONHECIMENTO",
    "OBJETO DE CONHECIMENTO",
    "OBJETIVOS DE APRENDIZAGEM E DESENVOLVIMENTO",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Habilidade:
    bncc_code: str
    stage: str
    grade: str
    grade_code: str
    subject: str
    subject_code: str
    description_pt: str
    thematic_unit: str
    knowledge_object: str
    source_pdf: str


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF preserving page order."""
    log.info("Reading %s ...", pdf_path.name)
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc=f"Pages ({pdf_path.name})"):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_text_to_habilidades(full_text: str, source_pdf: str) -> list[Habilidade]:
    # Normalise repeated whitespace inside lines
    text = re.sub(r"[ \t]{2,}", " ", full_text)

    matches = list(CODE_RE.finditer(text))
    log.info("Found %d code occurrences in %s", len(matches), source_pdf)

    # Keep only the last occurrence of each code (later ones usually have fuller descriptions)
    last_by_code: dict[str, re.Match] = {}
    for m in matches:
        stage, grade_digits, subj, _ = m.group(1), m.group(2), m.group(3), m.group(4)
        valid = (
            (stage == "EI" and subj in EI_SUBJECTS and grade_digits in EI_GRADES)
            or (stage == "EF" and subj in EF_SUBJECTS and grade_digits in EF_GRADES)
            or (stage == "EM" and subj in EM_SUBJECTS and grade_digits in EM_GRADES)
        )
        if not valid:
            continue
        full_code = m.group(0)
        last_by_code[full_code] = m

    ordered = sorted(last_by_code.values(), key=lambda m: m.start())
    log.info("Kept %d unique valid codes after validation", len(ordered))

    habilidades: list[Habilidade] = []
    for i, m in enumerate(ordered):
        code = m.group(0)
        stage, grade_digits, subj = m.group(1), m.group(2), m.group(3)

        # Description: everything after this code up to next code or ~1500 chars
        start = m.end()
        end = ordered[i + 1].start() if i + 1 < len(ordered) else min(start + 1500, len(text))
        raw = text[start:end]
        description = _clean_description(raw)

        # Thematic unit / knowledge object: search back from code position
        thematic = _search_back(text, m.start(), THEMATIC_HEADERS)
        knowledge = _search_back(text, m.start(), KNOWLEDGE_HEADERS)

        habilidades.append(
            Habilidade(
                bncc_code=code,
                stage=stage,
                grade=grade_label(stage, grade_digits),
                grade_code=grade_digits,
                subject=subject_full_name(stage, subj),
                subject_code=subj,
                description_pt=description,
                thematic_unit=thematic,
                knowledge_object=knowledge,
                source_pdf=source_pdf,
            )
        )

    return habilidades


def _clean_description(raw: str) -> str:
    """Clean raw text immediately following a BNCC code into a clean description."""
    # Strip leading parenthesis/punctuation remnants
    raw = raw.lstrip(") \n\t:–-.")
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    out: list[str] = []
    for ln in lines[:15]:
        # Stop at obvious section starters
        if re.match(r"^\d{1,3}$", ln):  # page number
            break
        if any(ln.upper().startswith(h) for h in THEMATIC_HEADERS + KNOWLEDGE_HEADERS):
            break
        if CODE_RE.match(ln):  # next code already on this line
            break
        out.append(ln)
        if ln.endswith((".", ";", "!", "?")):
            break

    desc = " ".join(out)
    desc = re.sub(r"\s+", " ", desc).strip()
    return desc


def _search_back(text: str, pos: int, headers: list[str]) -> str:
    """
    Find the nearest header that appears before `pos` and return what follows it
    on the same or next line (up to newline or 200 chars).
    """
    best_pos = -1
    best_header = ""
    for h in headers:
        idx = text.rfind(h, max(0, pos - 4000), pos)
        if idx > best_pos:
            best_pos = idx
            best_header = h
    if best_pos < 0:
        return ""

    tail = text[best_pos + len(best_header) : best_pos + len(best_header) + 300]
    tail = tail.lstrip(":\n\t ")
    # First non-empty line after header
    for ln in tail.split("\n"):
        ln = ln.strip()
        if ln and not CODE_RE.match(ln):
            return ln[:250]
    return ""


# ---------------------------------------------------------------------------
# Save + summary
# ---------------------------------------------------------------------------

def save_and_summarise(habilidades: list[Habilidade]) -> None:
    if not habilidades:
        log.error("No habilidades parsed — check input PDFs.")
        sys.exit(1)

    df = pd.DataFrame([asdict(h) for h in habilidades])
    df = df.drop_duplicates(subset=["bncc_code"]).reset_index(drop=True)
    df = df.sort_values(["stage", "subject_code", "grade_code", "bncc_code"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d habilidades → %s", len(df), OUTPUT_CSV)

    # Summary counts by stage × subject
    pivot = df.groupby(["stage", "subject"]).size().unstack(fill_value=0)
    log.info("Counts by stage × subject:\n%s", pivot.to_string())

    by_stage = df["stage"].value_counts().to_dict()
    log.info(
        "Summary: total=%d | EI=%d EF=%d EM=%d",
        len(df),
        by_stage.get("EI", 0),
        by_stage.get("EF", 0),
        by_stage.get("EM", 0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 1: Parse ALL BNCC standards ===")

    pdfs_found = [p for p in (PDF_EI_EF, PDF_EM) if p.exists()]
    if not pdfs_found:
        log.error(
            "No input PDFs found.  Drop them into data/raw/:\n"
            "  - BNCC_EI_EF.pdf  (EI + EF)\n"
            "  - BNCC_EM.pdf     (EM)"
        )
        sys.exit(1)

    all_habs: list[Habilidade] = []
    for pdf in pdfs_found:
        text = extract_pdf_text(pdf)
        habs = parse_text_to_habilidades(text, pdf.name)
        log.info("  → parsed %d habilidades from %s", len(habs), pdf.name)
        all_habs.extend(habs)

    save_and_summarise(all_habs)
    log.info("Step 1 complete.")


if __name__ == "__main__":
    main()
