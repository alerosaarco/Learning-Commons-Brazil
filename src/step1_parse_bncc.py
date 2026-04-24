"""
Step 1: Parse ALL BNCC standards.

Primary source:  the official MEC PDF (complete, but messy layout)
Enrichment:      the API tabular PDFs (partial coverage, clean metadata)

Inputs (all live in the repo root or data/raw/):
  - BNCC_EI_EF_110518_versaofinal_site (1).pdf  — official MEC, all 3 stages
  - API BNCC EI.pdf                             — enrichment
  - API BNCC EF.pdf                             — enrichment
  - API BNCC EM.pdf                             — enrichment

Output:
  data/processed/bncc_standards.csv
  data/logs/step1_parsing.log
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import pandas as pd
import pdfplumber
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
for d in (RAW_DIR, PROCESSED_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS_DIR / "step1_parsing.log"
OUTPUT_CSV = PROCESSED_DIR / "bncc_standards.csv"

CANDIDATE_DIRS = [ROOT, RAW_DIR]

MEC_PDF_CANDIDATES = [
    "BNCC_EI_EF_110518_versaofinal_site (1).pdf",
    "BNCC_EI_EF_110518_versaofinal_site.pdf",
    "BNCC_EI_EF.pdf",
]
API_PDFS = {"EI": "API BNCC EI.pdf", "EF": "API BNCC EF.pdf", "EM": "API BNCC EM.pdf"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_RE = re.compile(r"\b(EI|EF|EM)(\d{2})([A-Z]{2,3})(\d{2,3})\b")
CODE_LEAD_RE = re.compile(r"^\s*\((EI|EF|EM)(\d{2})([A-Z]{2,3})(\d{2,3})\)\s*")
PAGE_CODE_RE = re.compile(r"^\(?(EI|EF|EM)\d{2}[A-Z]{2,3}\d{2,3}\)?$")

EI_SUBJECTS = {"EO", "CG", "TS", "EF", "ET"}
EF_SUBJECTS = {"MA", "LP", "CI", "GE", "HI", "AR", "EF", "ER", "CO", "LI"}
EM_SUBJECTS = {"LGG", "MAT", "CNT", "CHS", "LP", "LI", "ART", "EDF", "CO"}

SUBJECT_NAMES = {
    "MA": "Matemática", "LP": "Língua Portuguesa", "CI": "Ciências",
    "GE": "Geografia",  "HI": "História",           "AR": "Arte",
    "ER": "Ensino Religioso", "CO": "Computação", "LI": "Língua Inglesa",
    "EO": "O eu, o outro e o nós",
    "CG": "Corpo, Gestos e Movimentos",
    "TS": "Traços, Sons, Cores e Formas",
    "ET": "Espaços, Tempos, Quantidades, Relações e Transformações",
    "LGG": "Linguagens e suas Tecnologias",
    "MAT": "Matemática e suas Tecnologias",
    "CNT": "Ciências da Natureza e suas Tecnologias",
    "CHS": "Ciências Humanas e Sociais Aplicadas",
    "ART": "Arte",
    "EDF": "Educação Física",
}


def subject_full_name(stage: str, subj_code: str) -> str:
    if subj_code == "EF":
        return "Escuta, Fala, Pensamento e Imaginação" if stage == "EI" else "Educação Física"
    return SUBJECT_NAMES.get(subj_code, subj_code)


EI_GRADES = {
    "01": "Bebês (0–1a6m)",
    "02": "Crianças bem pequenas (1a7m–3a11m)",
    "03": "Crianças pequenas (4a–5a11m)",
}
EF_GRADES = {
    "01": "1º ano", "02": "2º ano", "03": "3º ano", "04": "4º ano",
    "05": "5º ano", "06": "6º ano", "07": "7º ano", "08": "8º ano",
    "09": "9º ano",
    "12": "1º–2º ano", "15": "1º–5º ano", "35": "3º–5º ano",
    "67": "6º–7º ano", "69": "6º–9º ano", "89": "8º–9º ano",
}
EM_GRADES = {"13": "1º–3º ano (Ensino Médio)"}

THEMATIC_HEADERS = [
    "UNIDADES TEMÁTICAS", "UNIDADE TEMÁTICA",
    "CAMPOS DE EXPERIÊNCIAS", "CAMPO DE EXPERIÊNCIAS",
    "COMPETÊNCIAS ESPECÍFICAS", "COMPETÊNCIA ESPECÍFICA",
]
KNOWLEDGE_HEADERS = [
    "OBJETOS DE CONHECIMENTO", "OBJETO DE CONHECIMENTO",
    "OBJETIVOS DE APRENDIZAGEM E DESENVOLVIMENTO",
]


def grade_label(stage: str, grade_digits: str) -> str:
    table = {"EI": EI_GRADES, "EF": EF_GRADES, "EM": EM_GRADES}.get(stage, {})
    return table.get(grade_digits, grade_digits)


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
# File discovery
# ---------------------------------------------------------------------------

def _find_file(names: list[str] | str) -> Path | None:
    if isinstance(names, str):
        names = [names]
    for d in CANDIDATE_DIRS:
        for n in names:
            p = d / n
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# MEC PDF parser — column-aware text extraction + regex on full text
# ---------------------------------------------------------------------------

def _extract_mec_page_column_aware(page) -> str:
    default = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
    words = page.extract_words(x_tolerance=2, y_tolerance=2)
    if not words:
        return default

    code_tokens = [w for w in words if PAGE_CODE_RE.match(w["text"])]
    if len(code_tokens) < 3:
        return default

    # Multi-column layout only if ≥2 codes share a horizontal line
    by_line: dict[int, list[dict]] = {}
    for w in code_tokens:
        by_line.setdefault(round(w["top"] / 4), []).append(w)
    if not any(len(v) >= 2 for v in by_line.values()):
        return default

    # Build columns based on x-clusters of code positions
    code_tokens.sort(key=lambda w: w["x0"])
    clusters: list[list[dict]] = [[code_tokens[0]]]
    for w in code_tokens[1:]:
        if w["x0"] - clusters[-1][-1]["x0"] < 80:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    if len(clusters) < 2:
        return default

    boundaries: list[float] = []
    for i in range(len(clusters) - 1):
        left_max = max(w["x1"] for w in clusters[i])
        right_min = min(w["x0"] for w in clusters[i + 1])
        boundaries.append((left_max + right_min) / 2)

    def col_of(x_center: float) -> int:
        for i, b in enumerate(boundaries):
            if x_center < b:
                return i
        return len(boundaries)

    cols: dict[int, list[dict]] = {i: [] for i in range(len(clusters))}
    for w in words:
        cols[col_of((w["x0"] + w["x1"]) / 2)].append(w)

    parts = []
    for i in sorted(cols):
        col_words = sorted(cols[i], key=lambda w: (round(w["top"] / 4), w["x0"]))
        lines: list[list[str]] = []
        current_y = None
        for w in col_words:
            y = round(w["top"] / 4)
            if current_y is None or y != current_y:
                lines.append([w["text"]])
                current_y = y
            else:
                lines[-1].append(w["text"])
        parts.append("\n".join(" ".join(ln) for ln in lines))
    return "\n\n".join(parts)


def _parse_mec_text(full_text: str, source_pdf: str) -> list[Habilidade]:
    text = re.sub(r"[ \t]{2,}", " ", full_text)
    matches = list(CODE_RE.finditer(text))

    last_by_code: dict[str, re.Match] = {}
    for m in matches:
        stage, gd, subj, _ = m.group(1), m.group(2), m.group(3), m.group(4)
        valid = (
            (stage == "EI" and subj in EI_SUBJECTS and gd in EI_GRADES)
            or (stage == "EF" and subj in EF_SUBJECTS and gd in EF_GRADES)
            or (stage == "EM" and subj in EM_SUBJECTS and gd in EM_GRADES)
        )
        if valid:
            last_by_code[m.group(0)] = m

    ordered = sorted(last_by_code.values(), key=lambda m: m.start())
    out: list[Habilidade] = []
    for i, m in enumerate(ordered):
        code = m.group(0)
        stage, gd, subj = m.group(1), m.group(2), m.group(3)
        start = m.end()
        end = ordered[i + 1].start() if i + 1 < len(ordered) else min(start + 1500, len(text))
        desc = _clean_desc(text[start:end])
        thematic = _search_back(text, m.start(), THEMATIC_HEADERS)
        knowledge = _search_back(text, m.start(), KNOWLEDGE_HEADERS)
        out.append(
            Habilidade(
                bncc_code=code,
                stage=stage,
                grade=grade_label(stage, gd),
                grade_code=gd,
                subject=subject_full_name(stage, subj),
                subject_code=subj,
                description_pt=desc,
                thematic_unit=thematic,
                knowledge_object=knowledge,
                source_pdf=source_pdf,
            )
        )
    return out


def _clean_desc(raw: str) -> str:
    raw = raw.lstrip(") \n\t:–-.")
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    out: list[str] = []
    for ln in lines[:15]:
        if re.match(r"^\d{1,3}$", ln):
            break
        if any(ln.upper().startswith(h) for h in THEMATIC_HEADERS + KNOWLEDGE_HEADERS):
            break
        if CODE_RE.match(ln):
            break
        out.append(ln)
        if ln.endswith((".", ";", "!", "?")):
            break
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def _search_back(text: str, pos: int, headers: list[str]) -> str:
    best_pos, best_header = -1, ""
    for h in headers:
        idx = text.rfind(h, max(0, pos - 4000), pos)
        if idx > best_pos:
            best_pos, best_header = idx, h
    if best_pos < 0:
        return ""
    tail = text[best_pos + len(best_header) : best_pos + len(best_header) + 300].lstrip(":\n\t ")
    for ln in tail.split("\n"):
        ln = ln.strip()
        if ln and not CODE_RE.match(ln):
            return ln[:250]
    return ""


def parse_mec_pdf(pdf_path: Path) -> list[Habilidade]:
    log.info("[MEC] reading %s", pdf_path.name)
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="MEC PDF pages"):
            pages.append(_extract_mec_page_column_aware(page))
    return _parse_mec_text("\n".join(pages), pdf_path.name)


# ---------------------------------------------------------------------------
# API-PDF parser (tabular) — used for enrichment
# ---------------------------------------------------------------------------

# -------- API PDF parsing via word coordinates --------
#
# These PDFs have a consistent tabular layout:
#   EI: Campo | Ano | Habilidade
#   EF: Disciplina | Ano | Unidade Temática | Objeto do Conhecimento | Habilidade
#   EM: Disciplina | Ano | Habilidade
# pdfplumber.extract_tables misses rows, and raw text extraction loses column
# structure (content wraps across lines and adjacent columns blur together).
# So we use word-level extraction: identify column x-boundaries from the header
# row, then for each habilidade code we find, reconstruct its row by collecting
# words within each column's y-band.

_CODE_TOKEN_RE = re.compile(r"^\((EI|EF|EM)(\d{2})([A-Z]{2,3})(\d{2,3})\)$")

_API_HEADERS = {
    "EI": ["Campo", "Ano", "Habilidade"],
    "EF": ["Disciplina", "Ano", "Unidade", "Objeto", "Habilidade"],
    "EM": ["Disciplina", "Ano", "Habilidade"],
}


def _locate_columns(words: list[dict], headers: list[str]) -> list[tuple[float, float]]:
    """
    Locate the x-range of each column.

    Header words are often centred in their cells, so we can't use their x0 as
    the column-left boundary.  Instead we:
      1. Detect the *body* content (below the header row) and cluster word x0
         positions to discover actual column boundaries.
      2. Return (x_left, x_right) tuples aligned with the input `headers` list.
    Returns an empty list if column detection fails.
    """
    # Step A — find the y of the header row using exact matches
    top_words = sorted(words, key=lambda w: w["top"])[:40]
    header_x: dict[str, float] = {}
    header_y = 0.0
    for w in top_words:
        for h in headers:
            if h not in header_x and w["text"].startswith(h):
                header_x[h] = w["x0"]
                header_y = max(header_y, w["top"])
    if len(header_x) != len(headers):
        return []

    # Step B — cluster x0 of body words to find actual column boundaries
    body = [w for w in words if w["top"] > header_y + 2]
    if not body:
        return []
    xs = sorted(w["x0"] for w in body)

    # Greedy clustering: new cluster when gap > 6 px
    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] > 6:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    cluster_mins = sorted({min(c) for c in clusters})

    # Step C — assign each header to the nearest cluster-min ≤ its x0
    col_lefts: list[float] = []
    for h in headers:
        hx = header_x[h]
        # pick the largest cluster-min that is ≤ hx+2 (header is usually right of its left edge)
        candidates = [m for m in cluster_mins if m <= hx + 2]
        col_lefts.append(max(candidates) if candidates else hx)

    # Step D — sort by x and build ranges
    ranges: list[tuple[float, float]] = []
    sorted_lefts = sorted(col_lefts)
    left_to_idx = {left: i for i, left in enumerate(col_lefts)}
    for i, left in enumerate(sorted_lefts):
        right = sorted_lefts[i + 1] - 1 if i + 1 < len(sorted_lefts) else float("inf")
        ranges.append((left, right))
    # Return in the order the caller requested (same order as `headers`)
    by_original = sorted(
        enumerate(col_lefts), key=lambda t: t[1]
    )  # [(orig_idx, left), ...] sorted by left
    ordered = [ranges[i] for i, _ in sorted(
        enumerate(by_original), key=lambda t: t[1][0]
    )]
    return ordered


def _column_of(x: float, ranges: list[tuple[float, float]]) -> int:
    for i, (left, right) in enumerate(ranges):
        if left - 3 <= x < right:
            return i
    return -1


def parse_api_pdf(pdf_path: Path, stage: str) -> list[Habilidade]:
    headers = _API_HEADERS[stage]
    out: list[Habilidade] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=2) or []
            if not words:
                continue
            col_ranges = _locate_columns(words, headers)
            if not col_ranges:
                continue

            habilidade_col = len(col_ranges) - 1
            header_y = max(
                (w["top"] for w in words[:40] if w["text"] in headers), default=0
            )
            body_words = [w for w in words if w["top"] > header_y + 2]

            # Locate every code token (it sits in the Habilidade column)
            code_tokens = [
                w for w in body_words
                if _column_of(w["x0"], col_ranges) == habilidade_col
                and _CODE_TOKEN_RE.match(w["text"])
            ]
            if not code_tokens:
                continue
            code_tokens.sort(key=lambda w: w["top"])

            for i, ct in enumerate(code_tokens):
                y_top = ct["top"] - 2
                y_bot = code_tokens[i + 1]["top"] - 2 if i + 1 < len(code_tokens) else float("inf")

                m = _CODE_TOKEN_RE.match(ct["text"])
                s, gd, subj, seq = m.group(1), m.group(2), m.group(3), m.group(4)
                if s != stage:
                    continue
                code = f"{s}{gd}{subj}{seq}"

                # Collect words belonging to this row's y-band, grouped by column
                cells: list[list[tuple[float, float, str]]] = [[] for _ in col_ranges]
                for w in body_words:
                    if not (y_top <= w["top"] < y_bot):
                        continue
                    ci = _column_of(w["x0"], col_ranges)
                    if ci < 0:
                        continue
                    cells[ci].append((w["top"], w["x0"], w["text"]))

                def cell_text(col: list[tuple[float, float, str]]) -> str:
                    col = sorted(col, key=lambda t: (round(t[0] / 2), t[1]))
                    return re.sub(r"\s+", " ", " ".join(t[2] for t in col)).strip()

                disciplina = cell_text(cells[0])
                ano = cell_text(cells[1])
                if stage == "EF":
                    unidade = cell_text(cells[2])
                    objeto = cell_text(cells[3])
                else:
                    unidade, objeto = "", ""

                # Habilidade cell starts with the code token — strip it off
                hab_text = cell_text(cells[habilidade_col])
                hab_text = re.sub(r"^\(" + code + r"\)\s*", "", hab_text).strip()

                out.append(
                    Habilidade(
                        bncc_code=code,
                        stage=stage,
                        grade=ano or grade_label(stage, gd),
                        grade_code=gd,
                        subject=disciplina or subject_full_name(stage, subj),
                        subject_code=subj,
                        description_pt=hab_text,
                        thematic_unit=unidade,
                        knowledge_object=objeto,
                        source_pdf=pdf_path.name,
                    )
                )

    return out


# ---------------------------------------------------------------------------
# Merge MEC (primary) with API (enrichment)
# ---------------------------------------------------------------------------

def merge(primary: list[Habilidade], enrichment: list[Habilidade]) -> list[Habilidade]:
    """
    The API PDFs are tabular and yield cleaner data than the MEC PDF's
    column-aware text extraction.  When the API has a row, trust it for
    description / subject / grade / metadata.  The MEC parse is kept only
    as a fallback for codes the API lacks (e.g., Educação Física cross-grade
    codes outside the API export) and to wipe the MEC's noisy metadata
    heuristics when no API row backs them up.
    """
    enr_by_code = {h.bncc_code: h for h in enrichment}
    merged: list[Habilidade] = []
    for h in primary:
        enr = enr_by_code.get(h.bncc_code)
        if enr:
            merged.append(
                replace(
                    h,
                    description_pt=enr.description_pt or h.description_pt,
                    subject=enr.subject or h.subject,
                    grade=enr.grade or h.grade,
                    thematic_unit=enr.thematic_unit,
                    knowledge_object=enr.knowledge_object,
                )
            )
        else:
            # Keep the MEC row but drop the unreliable header-matched metadata
            merged.append(replace(h, thematic_unit="", knowledge_object=""))
    # Add any codes only in the API PDFs (e.g., Computação, Língua Inglesa)
    primary_codes = {h.bncc_code for h in primary}
    for h in enrichment:
        if h.bncc_code not in primary_codes:
            merged.append(h)
    return merged


# ---------------------------------------------------------------------------
# Save + summarise
# ---------------------------------------------------------------------------

def save_and_summarise(hab: list[Habilidade]) -> None:
    if not hab:
        log.error("No habilidades parsed.")
        sys.exit(1)
    df = pd.DataFrame([asdict(h) for h in hab])
    df = df.drop_duplicates(subset=["bncc_code"]).reset_index(drop=True)
    df = df.sort_values(["stage", "subject_code", "grade_code", "bncc_code"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d habilidades → %s", len(df), OUTPUT_CSV)

    pivot = df.groupby(["stage", "subject"]).size().unstack(fill_value=0)
    log.info("Counts by stage × subject:\n%s", pivot.to_string())

    bs = df["stage"].value_counts().to_dict()
    log.info("Summary: total=%d | EI=%d EF=%d EM=%d",
             len(df), bs.get("EI", 0), bs.get("EF", 0), bs.get("EM", 0))

    empty = df[df["description_pt"].fillna("").str.strip().str.len() < 5]
    if len(empty):
        log.warning("%d rows with near-empty descriptions: %s",
                    len(empty), empty["bncc_code"].tolist())
    with_tu = (df["thematic_unit"].fillna("") != "").sum()
    with_ko = (df["knowledge_object"].fillna("") != "").sum()
    log.info("Metadata coverage: thematic_unit on %d rows, knowledge_object on %d rows",
             with_tu, with_ko)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 1: Parse ALL BNCC standards (MEC primary + API enrichment) ===")

    mec_pdf = _find_file(MEC_PDF_CANDIDATES)
    if not mec_pdf:
        log.error("MEC PDF not found.  Expected one of: %s", MEC_PDF_CANDIDATES)
        sys.exit(1)

    primary = parse_mec_pdf(mec_pdf)
    log.info("[MEC] parsed %d habilidades", len(primary))

    enrichment: list[Habilidade] = []
    for stage, name in API_PDFS.items():
        p = _find_file(name)
        if not p:
            log.warning("[API %s] not found (%s) — skipping enrichment for this stage", stage, name)
            continue
        subset = parse_api_pdf(p, stage)
        log.info("[API %s] parsed %d habilidades", stage, len(subset))
        enrichment.extend(subset)

    merged = merge(primary, enrichment) if enrichment else primary
    save_and_summarise(merged)
    log.info("Step 1 complete.")


if __name__ == "__main__":
    main()
