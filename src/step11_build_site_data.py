"""
Step 11: Build static JSON data files for the GitHub Pages website.

Produces:
  docs/data/index.json          - lightweight search index (all habilidades)
  docs/data/habilidades/{code}.json - full detail per habilidade

index.json structure (array):
  { code, stage, grade, subject, description_pt, has_prereq_data }

per-habilidade JSON:
  { code, stage, grade, subject, description_pt,
    components: [{id, description_pt, description_en, match_tier,
                  cc_component_id, cc_component_description}],
    cc_matches:  [{standard_id, standard_code, description, score, rank,
                   n_merge, n_link_regional, n_link}],
    prerequisites: [{code, description_pt}],
    unlocks:       [{code, description_pt}],
    has_prereq_data: bool  }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

MATH_SUBJECTS = {
    "Matemática",
    "Matemática e suas Tecnologias",
    "Espaços, Tempos, Quantidades, Relações e Transformações",
}

ROOT     = Path(__file__).resolve().parent.parent
PROC     = ROOT / "data" / "processed"
RAW      = ROOT / "data" / "raw"
DOCS     = ROOT / "docs"
DATA_DIR = DOCS / "data"
HAB_DIR  = DATA_DIR / "habilidades"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
HAB_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step11_build_site_data.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=== Step 11: Build site data ===")

    # -----------------------------------------------------------------------
    # Load all processed data
    # -----------------------------------------------------------------------
    bncc       = pd.read_csv(PROC / "bncc_translated.csv",          encoding="utf-8-sig")
    comps_pt   = pd.read_csv(PROC / "bncc_components_pt.csv",       encoding="utf-8-sig")
    comps_en   = pd.read_csv(PROC / "bncc_components.csv",          encoding="utf-8-sig")
    matches_c  = pd.read_csv(PROC / "bncc_component_matches.csv",   encoding="utf-8-sig")
    matches_s  = pd.read_csv(PROC / "bncc_cc_standard_matches.csv", encoding="utf-8-sig")
    prereqs    = pd.read_csv(PROC / "bncc_prerequisites.csv",       encoding="utf-8-sig")
    cc_comps   = pd.read_csv(PROC / "cc_components_full.csv",       encoding="utf-8-sig")
    cc_stds    = pd.read_csv(RAW  / "lc_cc_standards.csv",          encoding="utf-8-sig")

    log.info("Loaded all data files")

    # -----------------------------------------------------------------------
    # Deduplicate BNCC habilidades with identical (subject, description_pt).
    # The BNCC source contains pairs where the same skill appears under both
    # a single-grade code (e.g. EF06CO08) and a multi-grade code (EF69CO08).
    # Keep the entry whose grade field covers more years (widest range), so
    # the multi-grade code wins; fall back to the first occurrence.
    # -----------------------------------------------------------------------
    def _grade_width(grade_str: str) -> int:
        """Count how many distinct grade years a grade string covers."""
        import re
        return len(re.findall(r'\d+º', str(grade_str)))

    seen_desc: dict[tuple, str] = {}   # (subject, description_pt) → winning bncc_code
    for _, row in bncc.iterrows():
        key = (row["subject"], row["description_pt"])
        if key not in seen_desc:
            seen_desc[key] = row["bncc_code"]
        else:
            # Replace if this entry covers more grades
            existing_code = seen_desc[key]
            existing_row  = bncc[bncc["bncc_code"] == existing_code].iloc[0]
            if _grade_width(row["grade"]) > _grade_width(existing_row["grade"]):
                seen_desc[key] = row["bncc_code"]

    # Keep only the winning code per (subject, description_pt) pair
    winning_codes = set(seen_desc.values())
    dropped = len(bncc) - len(winning_codes)
    if dropped:
        log.info("Deduplicating %d habilidades with identical descriptions → keeping %d",
                 dropped, len(winning_codes))
        bncc = bncc[bncc["bncc_code"].isin(winning_codes)]

    # -----------------------------------------------------------------------
    # Build lookup tables
    # -----------------------------------------------------------------------
    # bncc_code → metadata
    bncc_meta: dict[str, dict] = {
        row["bncc_code"]: {
            "stage":          row.get("stage", ""),
            "grade":          row.get("grade", ""),
            "subject":        row.get("subject", ""),
            "description_pt": row.get("description_pt", ""),
        }
        for _, row in bncc.iterrows()
    }

    # component_id → Portuguese description (deduplicated by component_id)
    pt_map: dict[str, str] = (
        comps_pt.drop_duplicates("component_id")
        .set_index("component_id")["description_pt"]
        .to_dict()
    )

    # component_id → English description (deduplicated)
    en_map: dict[str, str] = (
        comps_en.drop_duplicates("component_id")
        .set_index("component_id")["description"]
        .to_dict()
    )

    # bncc_component_id → (cc_component_id, match_tier)
    match_c_map: dict[str, tuple[str, str]] = {
        row["bncc_component_id"]: (row["cc_component_id"], row["match_tier"])
        for _, row in matches_c.iterrows()
    }

    # cc_component_id → description
    cc_comp_desc: dict[str, str] = dict(zip(cc_comps["component_id"], cc_comps["description"]))

    # cc_standard_id → (statement_code, description)
    # statement_code can be NaN for some Science standards — normalise to ""
    cc_std_info: dict[str, tuple[str, str]] = {
        row["identifier"]: (
            "" if pd.isna(row["statement_code"]) else str(row["statement_code"]),
            row["description"],
        )
        for _, row in cc_stds.iterrows()
    }

    # habilidade_code → list of components rows (sorted consistently)
    hab_comps: dict[str, list] = {}
    for _, row in comps_en.iterrows():
        hab_comps.setdefault(row["habilidade_code"], []).append(row)

    # standard_id matches per habilidade
    std_matches_by_hab: dict[str, list] = {}
    for _, row in matches_s.sort_values("rank").iterrows():
        std_matches_by_hab.setdefault(row["habilidade_code"], []).append(row)

    # prerequisite edges
    prereq_by_hab: dict[str, list[str]] = {}   # hab → [prereq_codes]
    unlocks_by_hab: dict[str, list[str]] = {}  # prereq → [hab_codes it unlocks]
    for _, row in prereqs.iterrows():
        prereq_by_hab.setdefault(row["habilidade_code"], []).append(row["prerequisite_code"])
        unlocks_by_hab.setdefault(row["prerequisite_code"], []).append(row["habilidade_code"])

    # -----------------------------------------------------------------------
    # Build per-habilidade JSON files
    # -----------------------------------------------------------------------
    index_entries = []
    total = len(bncc_meta)
    log.info("Building %d habilidade JSON files…", total)

    for code, meta in bncc_meta.items():
        subject = meta["subject"]
        has_prereq_data = subject in MATH_SUBJECTS

        # Components
        components = []
        for comp_row in hab_comps.get(code, []):
            cid = str(comp_row["component_id"])
            cc_cid, tier = match_c_map.get(cid, (cid, "none"))
            components.append({
                "id":                     cid,
                "description_pt":         pt_map.get(cid, comp_row.get("description", "")),
                "description_en":         en_map.get(cid, comp_row.get("description", "")),
                "match_tier":             tier,
                "cc_component_id":        cc_cid if tier != "none" else None,
                "cc_component_description": cc_comp_desc.get(cc_cid, "") if tier != "none" else None,
            })

        # CC standard matches
        cc_matches = []
        for sm in std_matches_by_hab.get(code, [])[:5]:
            sid = sm["cc_standard_id"]
            std_code, std_desc = cc_std_info.get(sid, ("", ""))
            cc_matches.append({
                "standard_id":    sid,
                "standard_code":  std_code,
                "description":    std_desc,
                "score":          round(float(sm["score"]), 4),
                "rank":           int(sm["rank"]),
                "n_merge":        int(sm["n_merge"]),
                "n_link_regional": int(sm["n_link_regional"]),
                "n_link":         int(sm["n_link"]),
            })

        # Prerequisites and unlocks
        prerequisites = [
            {"code": p, "description_pt": bncc_meta.get(p, {}).get("description_pt", "")}
            for p in prereq_by_hab.get(code, [])
        ]
        unlocks = [
            {"code": u, "description_pt": bncc_meta.get(u, {}).get("description_pt", "")}
            for u in unlocks_by_hab.get(code, [])
        ]

        hab_json = {
            "code":            code,
            "stage":           meta["stage"],
            "grade":           meta["grade"],
            "subject":         subject,
            "description_pt":  meta["description_pt"],
            "has_prereq_data": has_prereq_data,
            "components":      components,
            "cc_matches":      cc_matches,
            "prerequisites":   prerequisites,
            "unlocks":         unlocks,
        }

        (HAB_DIR / f"{code}.json").write_text(
            json.dumps(hab_json, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        index_entries.append({
            "code":            code,
            "stage":           meta["stage"],
            "grade":           meta["grade"],
            "subject":         subject,
            "description_pt":  meta["description_pt"],
            "has_prereq_data": has_prereq_data,
        })

    # -----------------------------------------------------------------------
    # Write search index
    # -----------------------------------------------------------------------
    (DATA_DIR / "index.json").write_text(
        json.dumps(index_entries, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    log.info("Written %d habilidade JSON files to %s", total, HAB_DIR)
    log.info("Written search index to %s", DATA_DIR / "index.json")

    # Stats
    with_comps  = sum(1 for c in index_entries if c["code"] in hab_comps)
    with_cc     = sum(1 for c in index_entries if c["code"] in std_matches_by_hab)
    with_pre    = sum(1 for c in index_entries if c["code"] in prereq_by_hab)
    with_unlock = sum(1 for c in index_entries if c["code"] in unlocks_by_hab)
    log.info("Habilidades with components:    %d / %d", with_comps, total)
    log.info("Habilidades with CC matches:    %d / %d", with_cc, total)
    log.info("Habilidades with prerequisites: %d / %d", with_pre, total)
    log.info("Habilidades that unlock others: %d / %d", with_unlock, total)


if __name__ == "__main__":
    main()
