"""
Step 2: Extract Learning Commons Knowledge Graph data from local JSONL exports.

RUN THIS STEP ON YOUR LOCAL MACHINE (not in the sandbox) — the source files
are too large to commit to git and the CDN is blocked from the sandbox.

Inputs (in data/raw/, gitignored — download once):
  lc_nodes.jsonl         250 MB  — https://cdn.learningcommons.org/knowledge-graph/v1.8.0/exports/nodes.jsonl
  lc_relationships.jsonl 450 MB  — https://cdn.learningcommons.org/knowledge-graph/v1.8.0/exports/relationships.jsonl

Outputs (small, committed to data/raw/):
  lc_cc_standards.csv     — every Common Core standard (jurisdiction == "Multi-State")
  lc_components.csv       — every Learning Component
  lc_cc_components.csv    — which CC standards have which learning components
  lc_cc_prerequisites.csv — prerequisite graph between CC standards

After running, commit the 4 small CSVs and push.  All subsequent steps use
only those CSVs — they never need the raw JSONL files.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
LOGS_DIR = ROOT / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

NODES_FILE = RAW_DIR / "lc_nodes.jsonl"
RELS_FILE = RAW_DIR / "lc_relationships.jsonl"

OUT_STANDARDS = RAW_DIR / "lc_cc_standards.csv"
OUT_COMPONENTS = RAW_DIR / "lc_components.csv"
OUT_CC_COMPONENTS = RAW_DIR / "lc_cc_components.csv"
OUT_CC_PREREQS = RAW_DIR / "lc_cc_prerequisites.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step2_parsing.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node-label constants
# ---------------------------------------------------------------------------

LABEL_STANDARD = "StandardsFrameworkItem"
LABEL_COMPONENT = "LearningComponent"

# Relationship labels we care about
REL_HAS_COMPONENT = "hasLearningComponent"
REL_PREREQ_OF = "isPrerequisiteOf"
REL_HAS_PREREQ = "hasPrerequisite"
REL_RELEVANT = "relevantToStandard"

WANTED_REL_LABELS = {REL_HAS_COMPONENT, REL_PREREQ_OF, REL_HAS_PREREQ, REL_RELEVANT}


# ---------------------------------------------------------------------------
# Pass 1: stream nodes.jsonl
# ---------------------------------------------------------------------------

def _line_count(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def parse_nodes(path: Path) -> tuple[list[dict], list[dict]]:
    """
    Returns (cc_standards, learning_components).

    nodes.jsonl schema:
      {"type":"node","identifier":"<id>","labels":["Label",...],"properties":{...}}

    StandardsFrameworkItem properties we keep:
      statementCode, description, jurisdiction, academicSubject,
      gradeLevel, statementType, caseIdentifierUUID

    LearningComponent properties we keep:
      name, description
    """
    cc_standards: list[dict] = []
    components: list[dict] = []
    skipped = 0

    log.info("Pass 1: streaming %s …", path.name)
    with open(path, encoding="utf-8") as f:
        for raw in tqdm(f, desc="nodes", unit=" lines"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                node = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue

            labels = node.get("labels", [])
            props = node.get("properties", {})
            identifier = node.get("identifier", "")

            if LABEL_STANDARD in labels:
                if props.get("jurisdiction") == "Multi-State":
                    cc_standards.append({
                        "identifier": identifier,
                        "case_uuid": props.get("caseIdentifierUUID", identifier),
                        "statement_code": props.get("statementCode", ""),
                        "description": props.get("description", ""),
                        "academic_subject": props.get("academicSubject", ""),
                        "grade_level": props.get("gradeLevel", ""),
                        "statement_type": props.get("statementType", ""),
                    })

            elif LABEL_COMPONENT in labels:
                components.append({
                    "identifier": identifier,
                    "name": props.get("name", ""),
                    "description": props.get("description", ""),
                })

    if skipped:
        log.warning("Skipped %d malformed lines in nodes.jsonl", skipped)

    log.info(
        "Pass 1 done: %d CC standards, %d learning components",
        len(cc_standards), len(components),
    )
    return cc_standards, components


# ---------------------------------------------------------------------------
# Pass 2: stream relationships.jsonl
# ---------------------------------------------------------------------------

def parse_relationships(
    path: Path,
    cc_ids: set[str],
    component_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """
    Returns (cc_component_links, cc_prereq_links).

    relationships.jsonl schema:
      {
        "type": "relationship",
        "identifier": "<id>",
        "label": "<RelType>",          ← singular string, not array
        "source_identifier": "<id>",
        "source_labels": ["Label"],
        "target_identifier": "<id>",
        "target_labels": ["Label"],
        "properties": {...}
      }

    We capture:
      hasLearningComponent  source=StandardsFrameworkItem → target=LearningComponent
      isPrerequisiteOf      source=StandardsFrameworkItem → target=StandardsFrameworkItem
      hasPrerequisite       source=StandardsFrameworkItem → target=StandardsFrameworkItem
      relevantToStandard    source=LearningComponent/Factor → target=StandardsFrameworkItem
    """
    component_links: list[dict] = []
    prereq_links: list[dict] = []
    skipped = 0

    log.info("Pass 2: streaming %s …", path.name)
    with open(path, encoding="utf-8") as f:
        for raw in tqdm(f, desc="relationships", unit=" lines"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rel = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue

            label = rel.get("label", "")
            if label not in WANTED_REL_LABELS:
                continue

            src = rel.get("source_identifier", "")
            tgt = rel.get("target_identifier", "")

            if label == REL_HAS_COMPONENT:
                # CC standard → Learning Component
                if src in cc_ids and tgt in component_ids:
                    component_links.append({"standard_id": src, "component_id": tgt})

            elif label in (REL_PREREQ_OF, REL_HAS_PREREQ):
                # both endpoints should be CC standards
                if src in cc_ids and tgt in cc_ids:
                    if label == REL_PREREQ_OF:
                        prereq_links.append({"standard_id": tgt, "prerequisite_id": src})
                    else:
                        prereq_links.append({"standard_id": src, "prerequisite_id": tgt})

            elif label == REL_RELEVANT:
                # Factor/component → CC standard  (reversed: standard has factor)
                if tgt in cc_ids and src in component_ids:
                    component_links.append({"standard_id": tgt, "component_id": src})

    if skipped:
        log.warning("Skipped %d malformed lines in relationships.jsonl", skipped)

    # Deduplicate
    component_links = list({(r["standard_id"], r["component_id"]): r for r in component_links}.values())
    prereq_links = list({(r["standard_id"], r["prerequisite_id"]): r for r in prereq_links}.values())

    log.info(
        "Pass 2 done: %d CC→component links, %d CC prereq links",
        len(component_links), len(prereq_links),
    )
    return component_links, prereq_links


# ---------------------------------------------------------------------------
# Save + summarise
# ---------------------------------------------------------------------------

def save_and_summarise(
    cc_standards: list[dict],
    components: list[dict],
    component_links: list[dict],
    prereq_links: list[dict],
) -> None:
    df_std = pd.DataFrame(cc_standards).drop_duplicates(subset=["identifier"])
    df_cmp = pd.DataFrame(components).drop_duplicates(subset=["identifier"])
    df_cc_cmp = pd.DataFrame(component_links) if component_links else pd.DataFrame(columns=["standard_id", "component_id"])
    df_prereq = pd.DataFrame(prereq_links) if prereq_links else pd.DataFrame(columns=["standard_id", "prerequisite_id"])

    df_std.to_csv(OUT_STANDARDS, index=False, encoding="utf-8-sig")
    df_cmp.to_csv(OUT_COMPONENTS, index=False, encoding="utf-8-sig")
    df_cc_cmp.to_csv(OUT_CC_COMPONENTS, index=False, encoding="utf-8-sig")
    df_prereq.to_csv(OUT_CC_PREREQS, index=False, encoding="utf-8-sig")

    log.info("Saved:")
    log.info("  %d CC standards → %s", len(df_std), OUT_STANDARDS.relative_to(ROOT))
    log.info("  %d learning components → %s", len(df_cmp), OUT_COMPONENTS.relative_to(ROOT))
    log.info("  %d CC→component links → %s", len(df_cc_cmp), OUT_CC_COMPONENTS.relative_to(ROOT))
    log.info("  %d CC prereq links → %s", len(df_prereq), OUT_CC_PREREQS.relative_to(ROOT))

    # Coverage breakdown
    standards_with_components = df_cc_cmp["standard_id"].nunique() if len(df_cc_cmp) else 0
    standards_with_prereqs = df_prereq["standard_id"].nunique() if len(df_prereq) else 0
    total = len(df_std)

    log.info(
        "Coverage: %d/%d CC standards have components (%.0f%%)",
        standards_with_components, total,
        100 * standards_with_components / total if total else 0,
    )
    log.info(
        "Coverage: %d/%d CC standards have prereqs (%.0f%%)",
        standards_with_prereqs, total,
        100 * standards_with_prereqs / total if total else 0,
    )

    by_subject = df_std.groupby("academic_subject").size().sort_values(ascending=False)
    log.info("CC standards by subject:\n%s", by_subject.to_string())

    by_type = df_std.groupby("statement_type").size().sort_values(ascending=False)
    log.info("CC standards by type:\n%s", by_type.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Step 2: Parse Learning Commons Knowledge Graph v1.8.0 ===")

    for p in (NODES_FILE, RELS_FILE):
        if not p.exists():
            log.error(
                "Missing: %s\n"
                "Download from:\n"
                "  https://cdn.learningcommons.org/knowledge-graph/v1.8.0/exports/%s\n"
                "Run this step on your local machine, then commit the 4 output CSVs.",
                p, p.name,
            )
            sys.exit(1)

    cc_standards, components = parse_nodes(NODES_FILE)

    if not cc_standards:
        log.error("No CC standards found — check jurisdiction filter or schema.")
        sys.exit(1)
    if not components:
        log.error("No learning components found — check label filter.")
        sys.exit(1)

    cc_ids = {r["identifier"] for r in cc_standards}
    component_ids = {r["identifier"] for r in components}

    component_links, prereq_links = parse_relationships(RELS_FILE, cc_ids, component_ids)

    save_and_summarise(cc_standards, components, component_links, prereq_links)
    log.info("Step 2 complete. Commit the 4 CSVs in data/raw/ and push.")


if __name__ == "__main__":
    main()
