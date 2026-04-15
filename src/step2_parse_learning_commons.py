"""
Step 2: Parse the Learning Commons Knowledge Graph.

Expects these files already in data/raw/:
  - nodes.jsonl        (download from cdn.learningcommons.org)
  - relationships.jsonl

Extracts:
  - Common Core Math standards  → data/processed/cc_math_standards.csv
  - Learning Components         → data/processed/learning_components.csv
  - Relevant relationships      → data/processed/cc_relationships.csv

Also writes data/processed/kg_index.json — a compact index used by
later steps to avoid re-parsing the large JSONL files repeatedly.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NODES_PATH = RAW_DIR / "nodes.jsonl"
RELS_PATH = RAW_DIR / "relationships.jsonl"

RELEVANT_REL_TYPES = {
    "hasLearningComponent",
    "isPrerequisiteOf",
    "hasPrerequisite",
    "isRelatedTo",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    """Yield parsed JSON objects from a .jsonl file with a progress bar."""
    total = path.stat().st_size
    with open(path, encoding="utf-8") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=path.name
    ) as bar:
        for line in f:
            bar.update(len(line.encode()))
            line = line.strip()
            if line:
                yield json.loads(line)


def check_inputs():
    missing = [p for p in (NODES_PATH, RELS_PATH) if not p.exists()]
    if missing:
        log.error(
            "Missing input files:\n%s\n\n"
            "Download them with:\n"
            "  curl -L 'https://cdn.learningcommons.org/knowledge-graph/v1.7.0/exports/nodes.jsonl?ref=gh_curl' "
            "-o data/raw/nodes.jsonl\n"
            "  curl -L 'https://cdn.learningcommons.org/knowledge-graph/v1.7.0/exports/relationships.jsonl?ref=gh_curl' "
            "-o data/raw/relationships.jsonl",
            "\n".join(str(p) for p in missing),
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Node parsing
# ---------------------------------------------------------------------------

def parse_nodes():
    """
    Returns:
      cc_standards   : list of dicts — Common Core Math StandardsFrameworkItems
      lc_items       : list of dicts — LearningComponents
      all_nodes_index: dict  uuid → minimal node info (for relationship resolution)
    """
    cc_standards = []
    lc_items = []
    all_nodes_index = {}

    for node in iter_jsonl(NODES_PATH):
        uid = node.get("identifier", "")
        labels = node.get("labels", [])
        props = node.get("properties", {})

        # Index every node for relationship lookups
        all_nodes_index[uid] = {
            "labels": labels,
            "name": props.get("name", props.get("statementCode", uid[:8])),
        }

        if "StandardsFrameworkItem" in labels:
            # Filter: Common Core Math only
            jurisdiction = props.get("jurisdiction", "")
            subject = props.get("academicSubject", "")
            if jurisdiction == "Multi-State" and subject == "Mathematics":
                cc_standards.append(
                    {
                        "uuid": uid,
                        "code": props.get("statementCode", ""),
                        "description": props.get("description", ""),
                        "grade": props.get("gradeLevel", ""),
                        "subject": subject,
                        "jurisdiction": jurisdiction,
                        "statement_type": props.get("statementType", ""),
                    }
                )

        elif "LearningComponent" in labels:
            lc_items.append(
                {
                    "uuid": uid,
                    "name": props.get("name", ""),
                    "description": props.get("description", ""),
                }
            )

    return cc_standards, lc_items, all_nodes_index


# ---------------------------------------------------------------------------
# Relationship parsing
# ---------------------------------------------------------------------------

def parse_relationships(all_nodes_index: dict):
    """
    Returns relevant relationships filtered to those connecting known nodes.
    """
    rows = []
    for rel in iter_jsonl(RELS_PATH):
        rel_type = rel.get("type", "")
        if rel_type not in RELEVANT_REL_TYPES:
            continue

        start = rel.get("startIdentifier", "")
        end = rel.get("endIdentifier", "")

        rows.append(
            {
                "type": rel_type,
                "start_uuid": start,
                "end_uuid": end,
                "start_label": all_nodes_index.get(start, {}).get("labels", []),
                "end_label": all_nodes_index.get(end, {}).get("labels", []),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Build compact index
# ---------------------------------------------------------------------------

def build_kg_index(
    cc_standards: list[dict],
    lc_items: list[dict],
    relationships: list[dict],
) -> dict:
    """
    Build a compact JSON index for fast lookups in Steps 5–7:
      standard_to_components : cc_uuid → [lc_uuid, ...]
      prerequisites          : cc_uuid → [prerequisite_cc_uuid, ...]
      cc_by_uuid             : cc_uuid → {code, description, grade}
      lc_by_uuid             : lc_uuid → {name, description}
    """
    cc_uuids = {s["uuid"] for s in cc_standards}
    lc_uuids = {lc["uuid"] for lc in lc_items}

    standard_to_components: dict[str, list[str]] = {}
    prerequisites: dict[str, list[str]] = {}

    for rel in relationships:
        rtype = rel["type"]
        start = rel["start_uuid"]
        end = rel["end_uuid"]

        if rtype == "hasLearningComponent" and start in cc_uuids and end in lc_uuids:
            standard_to_components.setdefault(start, []).append(end)

        elif rtype == "isPrerequisiteOf" and end in cc_uuids and start in cc_uuids:
            # start isPrerequisiteOf end → end's prerequisite is start
            prerequisites.setdefault(end, []).append(start)

        elif rtype == "hasPrerequisite" and start in cc_uuids and end in cc_uuids:
            prerequisites.setdefault(start, []).append(end)

    index = {
        "cc_by_uuid": {s["uuid"]: {"code": s["code"], "description": s["description"], "grade": s["grade"]} for s in cc_standards},
        "lc_by_uuid": {lc["uuid"]: {"name": lc["name"], "description": lc["description"]} for lc in lc_items},
        "standard_to_components": standard_to_components,
        "prerequisites": prerequisites,
    }
    return index


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(cc_standards, lc_items, relationships, kg_index):
    cc_df = pd.DataFrame(cc_standards)
    cc_df.to_csv(PROCESSED_DIR / "cc_math_standards.csv", index=False, encoding="utf-8-sig")
    log.info("CC Math standards: %d → cc_math_standards.csv", len(cc_df))

    lc_df = pd.DataFrame(lc_items)
    lc_df.to_csv(PROCESSED_DIR / "learning_components.csv", index=False, encoding="utf-8-sig")
    log.info("Learning components: %d → learning_components.csv", len(lc_df))

    rel_df = pd.DataFrame(relationships)
    rel_df["start_label"] = rel_df["start_label"].apply(json.dumps)
    rel_df["end_label"] = rel_df["end_label"].apply(json.dumps)
    rel_df.to_csv(PROCESSED_DIR / "cc_relationships.csv", index=False, encoding="utf-8-sig")
    log.info("Relevant relationships: %d → cc_relationships.csv", len(rel_df))

    index_path = PROCESSED_DIR / "kg_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(kg_index, f, ensure_ascii=False, indent=2)
    log.info("KG index → %s", index_path)

    # Stats
    log.info(
        "Standards with learning components: %d / %d",
        len(kg_index["standard_to_components"]),
        len(cc_standards),
    )
    log.info(
        "Standards with prerequisites: %d / %d",
        len(kg_index["prerequisites"]),
        len(cc_standards),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Step 2: Parse Learning Commons Knowledge Graph ===")
    check_inputs()

    log.info("Parsing nodes...")
    cc_standards, lc_items, all_nodes_index = parse_nodes()
    log.info(
        "Found %d CC Math standards, %d learning components, %d total nodes.",
        len(cc_standards), len(lc_items), len(all_nodes_index),
    )

    log.info("Parsing relationships...")
    relationships = parse_relationships(all_nodes_index)
    log.info("Found %d relevant relationships.", len(relationships))

    log.info("Building KG index...")
    kg_index = build_kg_index(cc_standards, lc_items, relationships)

    save_outputs(cc_standards, lc_items, relationships, kg_index)
    log.info("Step 2 complete.")


if __name__ == "__main__":
    main()
