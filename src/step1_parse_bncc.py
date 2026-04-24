"""
Step 1: Parse ALL BNCC standards from the API JSON files.

Inputs (place files anywhere in repo root or data/raw/):
  bncc_infantil.json    — from https://cientificar1992.pythonanywhere.com/bncc_infantil/
  bncc_fundamental.json — from https://cientificar1992.pythonanywhere.com/bncc_fundamental/
  bncc_medio.json       — from https://cientificar1992.pythonanywhere.com/bncc_medio/

Output:
  data/processed/bncc_standards.csv
  data/logs/step1_parsing.log
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
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

JSON_FILES = {
    "EI": ["bncc_infantil.json", "bncc_ei.json"],
    "EF": ["bncc_fundamental.json", "bncc_ef.json"],
    "EM": ["bncc_medio.json", "bncc_em.json"],
}

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

EI_SUBJECTS = {"EO", "CG", "TS", "EF", "ET"}
EF_SUBJECTS = {"MA", "LP", "CI", "GE", "HI", "AR", "EF", "ER", "CO", "LI"}
EM_SUBJECTS = {"LGG", "MAT", "CNT", "CHS", "LP", "LI", "ART", "EDF", "CO"}

SUBJECT_NAMES = {
    "MA": "Matemática",
    "LP": "Língua Portuguesa",
    "CI": "Ciências",
    "GE": "Geografia",
    "HI": "História",
    "AR": "Arte",
    "ER": "Ensino Religioso",
    "CO": "Computação",
    "LI": "Língua Inglesa",
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


def _grade_label(stage: str, grade_digits: str) -> str:
    table = {"EI": EI_GRADES, "EF": EF_GRADES, "EM": EM_GRADES}.get(stage, {})
    return table.get(grade_digits, grade_digits)


def _valid_subjects(stage: str) -> set[str]:
    return {"EI": EI_SUBJECTS, "EF": EF_SUBJECTS, "EM": EM_SUBJECTS}.get(stage, set())


def _valid_grades(stage: str) -> set[str]:
    table = {"EI": EI_GRADES, "EF": EF_GRADES, "EM": EM_GRADES}.get(stage, {})
    return set(table.keys())


def _subject_name(stage: str, subj_code: str) -> str:
    if subj_code == "EF":
        return "Escuta, Fala, Pensamento e Imaginação" if stage == "EI" else "Educação Física"
    return SUBJECT_NAMES.get(subj_code, subj_code)


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


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_file(candidates: list[str]) -> Path | None:
    for d in CANDIDATE_DIRS:
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# Field extraction — handles multiple naming conventions from the API
# ---------------------------------------------------------------------------

def _get(record: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = record.get(k)
        if v is not None:
            return str(v).strip()
    return default


def _extract_code(record: dict) -> str | None:
    """Pull the BNCC code from the record, regardless of field name."""
    raw = _get(record, "codigo", "code", "habilidade", "id", "codHabilidade")
    if not raw:
        # Scan all values for a code-shaped string
        for v in record.values():
            if isinstance(v, str):
                m = CODE_RE.search(v)
                if m:
                    return m.group(0)
        return None
    # Strip surrounding parens if present: "(EF01MA01)" → "EF01MA01"
    raw = raw.strip("()")
    m = CODE_RE.fullmatch(raw)
    return m.group(0) if m else None


def _extract_description(record: dict, code: str) -> str:
    """Pull the Portuguese description, stripping the leading code token."""
    raw = _get(
        record,
        "descricao", "description", "habilidade", "texto",
        "descricaoHabilidade", "objetivoAprendizagem",
    )
    # Remove leading "(EFXXYYXX) " or "EFXXYYXX " if present
    raw = re.sub(r"^\(?" + re.escape(code) + r"\)?\s*", "", raw).strip()
    return raw


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def parse_json(path: Path, stage: str) -> list[Habilidade]:
    log.info("[%s] reading %s", stage, path.name)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # The API may return a list directly, or wrap it: {"results": [...]} etc.
    if isinstance(data, dict):
        for key in ("results", "data", "habilidades", "items", stage.lower()):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            # Last resort: find first list value
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
    if not isinstance(data, list):
        log.error("[%s] unexpected JSON structure in %s — not a list", stage, path.name)
        return []

    valid_subjects = _valid_subjects(stage)
    valid_grades = _valid_grades(stage)
    out: list[Habilidade] = []
    skipped = 0

    for record in tqdm(data, desc=f"Parse {stage} JSON"):
        if not isinstance(record, dict):
            continue
        code = _extract_code(record)
        if not code:
            skipped += 1
            continue

        m = CODE_RE.fullmatch(code)
        if not m:
            skipped += 1
            continue
        s, gd, subj = m.group(1), m.group(2), m.group(3)
        if s != stage or subj not in valid_subjects or gd not in valid_grades:
            skipped += 1
            continue

        desc = _extract_description(record, code)

        # Thematic unit — EI calls it campo de experiências
        thematic = _get(
            record,
            "unidadeTematica", "unidade_tematica", "campoExperiencias",
            "campo_experiencias", "campo", "unidade", "area",
        )
        # Knowledge object — EI calls it objetivos de aprendizagem
        knowledge = _get(
            record,
            "objetoConhecimento", "objeto_conhecimento", "objetivosAprendizagem",
            "objetivo_aprendizagem", "objeto", "conhecimento",
        )
        # Subject / grade labels — fall back to code-derived values
        subject_label = _get(
            record,
            "componente", "disciplina", "area", "campoExperiencias", "campo",
        ) or _subject_name(stage, subj)
        grade_label = _get(record, "ano", "serie", "etapa", "turma") or _grade_label(stage, gd)

        out.append(
            Habilidade(
                bncc_code=code,
                stage=stage,
                grade=grade_label,
                grade_code=gd,
                subject=subject_label,
                subject_code=subj,
                description_pt=desc,
                thematic_unit=thematic,
                knowledge_object=knowledge,
            )
        )

    log.info("[%s] parsed %d habilidades (%d skipped/invalid)", stage, len(out), skipped)
    return out


# ---------------------------------------------------------------------------
# Save + summarise
# ---------------------------------------------------------------------------

def save_and_summarise(hab: list[Habilidade]) -> None:
    if not hab:
        log.error("No habilidades parsed — check your JSON files.")
        sys.exit(1)

    df = pd.DataFrame([asdict(h) for h in hab])
    df = df.drop_duplicates(subset=["bncc_code"]).reset_index(drop=True)
    df = df.sort_values(["stage", "subject_code", "grade_code", "bncc_code"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d habilidades → %s", len(df), OUTPUT_CSV)

    by_stage = df["stage"].value_counts().to_dict()
    log.info(
        "Total=%d | EI=%d | EF=%d | EM=%d",
        len(df),
        by_stage.get("EI", 0),
        by_stage.get("EF", 0),
        by_stage.get("EM", 0),
    )

    pivot = df.groupby(["stage", "subject"]).size().reset_index(name="count")
    log.info("By stage × subject:\n%s", pivot.to_string(index=False))

    empty = df[df["description_pt"].fillna("").str.strip().str.len() < 5]
    if len(empty):
        log.warning("%d rows with near-empty descriptions: %s", len(empty), empty["bncc_code"].tolist())

    with_tu = (df["thematic_unit"].fillna("") != "").sum()
    with_ko = (df["knowledge_object"].fillna("") != "").sum()
    log.info("Metadata: thematic_unit on %d rows, knowledge_object on %d rows", with_tu, with_ko)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 1: Parse ALL BNCC standards from JSON ===")

    all_habilidades: list[Habilidade] = []
    missing: list[str] = []

    for stage, candidates in JSON_FILES.items():
        path = _find_file(candidates)
        if not path:
            log.warning("[%s] JSON not found (tried: %s) — skipping", stage, candidates)
            missing.append(stage)
            continue
        subset = parse_json(path, stage)
        all_habilidades.extend(subset)

    if missing:
        log.warning(
            "Missing JSON files for: %s\n"
            "Download from:\n"
            "  https://cientificar1992.pythonanywhere.com/bncc_infantil/\n"
            "  https://cientificar1992.pythonanywhere.com/bncc_fundamental/\n"
            "  https://cientificar1992.pythonanywhere.com/bncc_medio/\n"
            "Save as bncc_infantil.json / bncc_fundamental.json / bncc_medio.json "
            "in the repo root or data/raw/",
            missing,
        )

    if not all_habilidades:
        log.error("No habilidades parsed at all. Add the JSON files and retry.")
        sys.exit(1)

    save_and_summarise(all_habilidades)
    log.info("Step 1 complete.")


if __name__ == "__main__":
    main()
