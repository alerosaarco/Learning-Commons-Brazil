"""
Step 1: Parse ALL BNCC standards from the API JSON files.

Inputs (in data/raw/):
  bncc_infantil.json    — from https://cientificar1992.pythonanywhere.com/bncc_infantil/
  bncc_fundamental.json — from https://cientificar1992.pythonanywhere.com/bncc_fundamental/
  bncc_medio.json       — from https://cientificar1992.pythonanywhere.com/bncc_medio/

Output:
  data/processed/bncc_standards.csv
  data/logs/step1_parsing.log

JSON structure (summary):
  EI: educacao_infantil.campos_experiencia[].faixas_etarias[].objetivos[]
        → {codigo, descricao}
  EF: <disciplina>.ano[].unidades_tematicas[].objeto_conhecimento[].habilidades[]
        → {nome_habilidade: "(CODE) text..."}
  EM: <disciplina>.ano[].codigo_habilidade[]
        → {nome_codigo, nome_habilidade}
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "data" / "logs"
for d in (PROCESSED_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS_DIR / "step1_parsing.log"
OUTPUT_CSV = PROCESSED_DIR / "bncc_standards.csv"

JSON_FILES = {
    "EI": RAW_DIR / "bncc_infantil.json",
    "EF": RAW_DIR / "bncc_fundamental.json",
    "EM": RAW_DIR / "bncc_medio.json",
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
CODE_FULL_RE = re.compile(r"^(EI|EF|EM)(\d{2})([A-Z]{2,3})(\d{2,3})$")

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


def _subject_name(stage: str, subj_code: str) -> str:
    if subj_code == "EF":
        return "Escuta, Fala, Pensamento e Imaginação" if stage == "EI" else "Educação Física"
    return SUBJECT_NAMES.get(subj_code, subj_code)


def _grade_label(stage: str, grade_digits: str) -> str:
    table = {"EI": EI_GRADES, "EF": EF_GRADES, "EM": EM_GRADES}[stage]
    return table.get(grade_digits, grade_digits)


def _valid(code: str, stage: str) -> bool:
    m = CODE_FULL_RE.match(code)
    if not m:
        return False
    s, gd, subj, _ = m.groups()
    if s != stage:
        return False
    if stage == "EI":
        return subj in EI_SUBJECTS and gd in EI_GRADES
    if stage == "EF":
        return subj in EF_SUBJECTS and gd in EF_GRADES
    return subj in EM_SUBJECTS and gd in EM_GRADES


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


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# ---------------------------------------------------------------------------
# Stage-specific parsers
# ---------------------------------------------------------------------------

def parse_ei(data: dict) -> list[Habilidade]:
    out: list[Habilidade] = []
    root = data.get("educacao_infantil", {})
    for campo in root.get("campos_experiencia", []):
        for faixa in campo.get("faixas_etarias", []):
            faixa_name = _clean(faixa.get("nome_faixa", ""))
            for obj in faixa.get("objetivos", []):
                code = _clean(obj.get("codigo", ""))
                desc = _clean(obj.get("descricao", ""))
                m = CODE_FULL_RE.match(code)
                if not m or not _valid(code, "EI"):
                    log.warning("[EI] skipping invalid code: %r", code)
                    continue
                _, gd, subj, _ = m.groups()
                out.append(
                    Habilidade(
                        bncc_code=code,
                        stage="EI",
                        grade=faixa_name or _grade_label("EI", gd),
                        grade_code=gd,
                        subject=_subject_name("EI", subj),
                        subject_code=subj,
                        description_pt=desc,
                        thematic_unit="",
                        knowledge_object="",
                    )
                )
    return out


def parse_ef(data: dict) -> list[Habilidade]:
    out: list[Habilidade] = []
    # Each top-level key is a discipline wrapper
    for disc_key, disc in data.items():
        if not isinstance(disc, dict):
            continue
        for ano_entry in disc.get("ano", []):
            # nome_ano is a list like ["1º"] or ["6º", "7º"]
            nome_ano_list = ano_entry.get("nome_ano", [])
            ano_label = ", ".join(_clean(a) for a in nome_ano_list) if isinstance(nome_ano_list, list) else _clean(nome_ano_list)
            for ut in ano_entry.get("unidades_tematicas", []):
                ut_name = _clean(ut.get("nome_unidade", ""))
                for ok in ut.get("objeto_conhecimento", []):
                    ok_name = _clean(ok.get("nome_objeto", ""))
                    for hab in ok.get("habilidades", []):
                        text = _clean(hab.get("nome_habilidade", ""))
                        # Code is embedded: "(EF01MA01) Utilizar..."
                        m = re.match(r"^\(([^)]+)\)\s*(.*)$", text, re.DOTALL)
                        if not m:
                            log.warning("[EF] no leading code in: %r", text[:80])
                            continue
                        code = _clean(m.group(1))
                        desc = _clean(m.group(2))
                        if not _valid(code, "EF"):
                            log.warning("[EF] skipping invalid code: %r", code)
                            continue
                        cm = CODE_FULL_RE.match(code)
                        _, gd, subj, _ = cm.groups()
                        out.append(
                            Habilidade(
                                bncc_code=code,
                                stage="EF",
                                grade=ano_label or _grade_label("EF", gd),
                                grade_code=gd,
                                subject=_subject_name("EF", subj),
                                subject_code=subj,
                                description_pt=desc,
                                thematic_unit=ut_name,
                                knowledge_object=ok_name,
                            )
                        )
    return out


def parse_em(data: dict) -> list[Habilidade]:
    out: list[Habilidade] = []
    for disc_key, disc in data.items():
        if not isinstance(disc, dict):
            continue
        for ano_entry in disc.get("ano", []):
            nome_ano_list = ano_entry.get("nome_ano", [])
            ano_label = ", ".join(_clean(a) for a in nome_ano_list) if isinstance(nome_ano_list, list) else _clean(nome_ano_list)
            for hab in ano_entry.get("codigo_habilidade", []):
                code = _clean(hab.get("nome_codigo", ""))
                desc = _clean(hab.get("nome_habilidade", ""))
                if not _valid(code, "EM"):
                    log.warning("[EM] skipping invalid code: %r", code)
                    continue
                cm = CODE_FULL_RE.match(code)
                _, gd, subj, _ = cm.groups()
                out.append(
                    Habilidade(
                        bncc_code=code,
                        stage="EM",
                        grade=ano_label or _grade_label("EM", gd),
                        grade_code=gd,
                        subject=_subject_name("EM", subj),
                        subject_code=subj,
                        description_pt=desc,
                        thematic_unit="",
                        knowledge_object="",
                    )
                )
    return out


STAGE_PARSERS = {"EI": parse_ei, "EF": parse_ef, "EM": parse_em}


# ---------------------------------------------------------------------------
# Save + summarise
# ---------------------------------------------------------------------------

def save_and_summarise(hab: list[Habilidade]) -> None:
    if not hab:
        log.error("No habilidades parsed.")
        sys.exit(1)

    df = pd.DataFrame([asdict(h) for h in hab])
    before = len(df)
    df = df.drop_duplicates(subset=["bncc_code"]).reset_index(drop=True)
    if before != len(df):
        log.info("Dropped %d duplicate codes", before - len(df))
    df = df.sort_values(["stage", "subject_code", "grade_code", "bncc_code"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Saved %d habilidades → %s", len(df), OUTPUT_CSV.relative_to(ROOT))

    by_stage = df["stage"].value_counts().to_dict()
    log.info(
        "Total=%d | EI=%d | EF=%d | EM=%d",
        len(df),
        by_stage.get("EI", 0),
        by_stage.get("EF", 0),
        by_stage.get("EM", 0),
    )

    pivot = df.groupby(["stage", "subject_code", "subject"]).size().reset_index(name="count")
    log.info("By stage × subject:\n%s", pivot.to_string(index=False))

    empty = df[df["description_pt"].fillna("").str.strip().str.len() < 5]
    if len(empty):
        log.warning("%d rows with near-empty descriptions: %s",
                    len(empty), empty["bncc_code"].tolist())
    with_tu = (df["thematic_unit"].fillna("") != "").sum()
    with_ko = (df["knowledge_object"].fillna("") != "").sum()
    log.info("Metadata: thematic_unit on %d rows, knowledge_object on %d rows", with_tu, with_ko)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 1: Parse ALL BNCC standards from JSON ===")

    all_habilidades: list[Habilidade] = []
    for stage, path in JSON_FILES.items():
        if not path.exists():
            log.error("Missing: %s", path)
            sys.exit(1)
        log.info("[%s] reading %s", stage, path.relative_to(ROOT))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        subset = STAGE_PARSERS[stage](data)
        log.info("[%s] parsed %d habilidades", stage, len(subset))
        all_habilidades.extend(subset)

    save_and_summarise(all_habilidades)
    log.info("Step 1 complete.")


if __name__ == "__main__":
    main()
