"""
Step 2: Fetch Learning Commons Knowledge Graph data via REST API.

Designed to run in a GitHub Actions runner (where api.learningcommons.org is
reachable — the Claude Code sandbox blocks it).

Environment:
  LC_API_KEY  — required; set as a GitHub repo secret

Outputs (committed to data/raw/):
  lc_cc_standards.csv     — every Multi-State (Common Core) standard
  lc_components.csv       — every learning component referenced by a CC standard
  lc_cc_components.csv    — (standard_id, component_id) edges
  lc_cc_prerequisites.csv — (standard_id, prerequisite_id) edges
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "https://api.learningcommons.org/knowledge-graph/v0"
API_KEY = os.environ.get("LC_API_KEY")
if not API_KEY:
    print("ERROR: LC_API_KEY environment variable is not set", file=sys.stderr)
    sys.exit(1)

HEADERS = {"x-api-key": API_KEY, "Accept": "application/json"}
TIMEOUT = 30
CONCURRENCY = 12

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
LOGS_DIR = ROOT / "data" / "logs"
for d in (RAW_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

OUT_STANDARDS = RAW_DIR / "lc_cc_standards.csv"
OUT_COMPONENTS = RAW_DIR / "lc_components.csv"
OUT_CC_COMPONENTS = RAW_DIR / "lc_cc_components.csv"
OUT_CC_PREREQS = RAW_DIR / "lc_cc_prerequisites.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step2_fetch_lc.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP — retry transient failures, fail fast on 4xx (except 429)
# ---------------------------------------------------------------------------

class Retryable(Exception):
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(Retryable),
    reraise=True,
)
def _get(endpoint: str, params: dict | None = None) -> dict:
    url = f"{API_BASE}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise Retryable(f"network error: {e}") from e

    if r.status_code in (429, 500, 502, 503, 504):
        raise Retryable(f"{r.status_code} on {url}")
    if r.status_code == 404:
        return {"data": []}
    r.raise_for_status()
    return r.json()


def _paged(endpoint: str, params: dict | None = None) -> list[dict]:
    """
    Collect all pages.

    Uses response metadata to determine when to stop, so we don't depend
    on guessing the server's effective page size.  Tries several common
    pagination envelope shapes:
      - {data:[...], meta:{totalPages, currentPage}}
      - {data:[...], meta:{total, pageSize}}
      - {data:[...], pagination:{totalPages}}
      - {data:[...], totalCount}
    Falls back to stopping when a page returns fewer items than the
    *actual* page size observed in the first response.
    """
    params = dict(params or {})
    params["pageSize"] = 100          # conservative — server may cap lower
    params["page"] = 1
    out: list[dict] = []
    observed_page_size: int | None = None

    for page in range(1, 5001):      # absolute safety cap
        params["page"] = page
        resp = _get(endpoint, params)
        data = resp.get("data", [])

        if not data:
            break
        out.extend(data)

        if observed_page_size is None:
            observed_page_size = len(data)

        # Try to read total pages from response envelope
        meta = resp.get("meta") or resp.get("pagination") or {}
        total_pages = (
            meta.get("totalPages")
            or meta.get("total_pages")
        )
        if total_pages is not None:
            if page >= int(total_pages):
                break
            continue

        # Try total count + observed page size
        total_count = (
            meta.get("total")
            or meta.get("totalCount")
            or resp.get("totalCount")
        )
        if total_count is not None and observed_page_size:
            import math
            if page >= math.ceil(int(total_count) / observed_page_size):
                break
            continue

        # Fallback: stop when the page is smaller than the first page
        if len(data) < observed_page_size:
            break

    return out


# ---------------------------------------------------------------------------
# Fetch phases
# ---------------------------------------------------------------------------

def _log_envelope(label: str, resp: dict) -> None:
    """Log the non-data keys of a response so we can see pagination metadata."""
    keys = {k: v for k, v in resp.items() if k != "data"}
    log.info("Response envelope [%s]: %s", label, keys)


def fetch_cc_frameworks() -> list[dict]:
    log.info("Fetching Multi-State (Common Core) frameworks…")
    raw = _get("/standards-frameworks", {"jurisdiction": "Multi-State", "pageSize": 100})
    _log_envelope("/standards-frameworks", raw)
    frameworks = raw.get("data", [])
    for f in frameworks:
        log.info(
            "  framework: %s — %s (uuid=%s)",
            f.get("academicSubject", "?"),
            f.get("title", f.get("name", "")),
            f.get("caseIdentifierUUID", f.get("identifier", "")),
        )
    return frameworks


def fetch_standards_for_framework(framework: dict) -> list[dict]:
    uuid = framework.get("caseIdentifierUUID") or framework.get("identifier")
    if not uuid:
        return []
    subject = framework.get("academicSubject", "")
    title = framework.get("title") or framework.get("name") or ""
    log.info("Fetching standards for '%s' — %s (uuid=%s)…", title, subject, uuid)

    # Log first-page envelope to confirm pagination shape
    first = _get("/academic-standards",
                 {"standardsFrameworkCaseIdentifierUUID": uuid, "pageSize": 100, "page": 1})
    _log_envelope(f"/academic-standards [{subject}]", first)

    items = _paged(
        "/academic-standards",
        {"standardsFrameworkCaseIdentifierUUID": uuid},
    )
    for item in items:
        item["_framework_subject"] = subject
    log.info("  → %d items", len(items))
    return items


def fetch_prereqs_and_components(standard: dict) -> tuple[list[dict], list[dict]]:
    """
    Returns (prereq_rows, component_rows) for one standard.
    Each prereq_row: {standard_id, prerequisite_id, prerequisite_code}
    Each component_row: {standard_id, component_id, component_name,
                         component_description}
    """
    uuid = standard.get("caseIdentifierUUID") or standard.get("identifier")
    if not uuid:
        return [], []

    try:
        prereq_resp = _get(f"/academic-standards/{uuid}/prerequisites")
    except requests.HTTPError:
        prereq_resp = {"data": []}
    try:
        comp_resp = _get(f"/academic-standards/{uuid}/learning-components")
    except requests.HTTPError:
        comp_resp = {"data": []}

    prereq_rows = [
        {
            "standard_id": uuid,
            "prerequisite_id": p.get("caseIdentifierUUID") or p.get("identifier", ""),
            "prerequisite_code": p.get("statementCode", ""),
        }
        for p in prereq_resp.get("data", [])
    ]
    comp_rows = [
        {
            "standard_id": uuid,
            "component_id": c.get("identifier", ""),
            "component_name": c.get("name", ""),
            "component_description": c.get("description", ""),
        }
        for c in comp_resp.get("data", [])
    ]
    return prereq_rows, comp_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _std_row(s: dict) -> dict:
    return {
        "identifier": s.get("caseIdentifierUUID") or s.get("identifier", ""),
        "statement_code": s.get("statementCode", ""),
        "description": s.get("description", ""),
        "academic_subject": s.get("academicSubject", s.get("_framework_subject", "")),
        "grade_level": s.get("gradeLevel", ""),
        "statement_type": s.get("normalizedStatementType", s.get("statementType", "")),
    }


def main() -> None:
    log.info("=== Step 2: Fetch Learning Commons KG via REST API ===")

    frameworks = fetch_cc_frameworks()
    if not frameworks:
        log.error("No Multi-State frameworks returned — check API key / endpoint.")
        sys.exit(1)

    all_standards: list[dict] = []
    for fw in frameworks:
        all_standards.extend(fetch_standards_for_framework(fw))
    log.info("Total CC standards: %d", len(all_standards))

    prereq_rows: list[dict] = []
    component_rows: list[dict] = []
    component_dedup: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(fetch_prereqs_and_components, s): s for s in all_standards}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="standards"):
            pr, cp = fut.result()
            prereq_rows.extend(pr)
            for row in cp:
                cid = row["component_id"]
                if cid and cid not in component_dedup:
                    component_dedup[cid] = {
                        "identifier": cid,
                        "name": row["component_name"],
                        "description": row["component_description"],
                    }
                component_rows.append({"standard_id": row["standard_id"], "component_id": cid})

    # Write outputs
    df_std = pd.DataFrame([_std_row(s) for s in all_standards]).drop_duplicates(
        subset=["identifier"]
    )
    df_cmp = pd.DataFrame(list(component_dedup.values()))
    df_cc_cmp = (
        pd.DataFrame(component_rows).drop_duplicates(subset=["standard_id", "component_id"])
        if component_rows
        else pd.DataFrame(columns=["standard_id", "component_id"])
    )
    df_prereq = (
        pd.DataFrame(prereq_rows).drop_duplicates(subset=["standard_id", "prerequisite_id"])
        if prereq_rows
        else pd.DataFrame(columns=["standard_id", "prerequisite_id", "prerequisite_code"])
    )

    df_std.to_csv(OUT_STANDARDS, index=False, encoding="utf-8-sig")
    df_cmp.to_csv(OUT_COMPONENTS, index=False, encoding="utf-8-sig")
    df_cc_cmp.to_csv(OUT_CC_COMPONENTS, index=False, encoding="utf-8-sig")
    df_prereq.to_csv(OUT_CC_PREREQS, index=False, encoding="utf-8-sig")

    log.info("Saved:")
    log.info("  %5d CC standards         → %s", len(df_std), OUT_STANDARDS.relative_to(ROOT))
    log.info("  %5d learning components  → %s", len(df_cmp), OUT_COMPONENTS.relative_to(ROOT))
    log.info("  %5d CC→component links   → %s", len(df_cc_cmp), OUT_CC_COMPONENTS.relative_to(ROOT))
    log.info("  %5d CC prereq links      → %s", len(df_prereq), OUT_CC_PREREQS.relative_to(ROOT))

    total = len(df_std) or 1
    std_with_cmp = df_cc_cmp["standard_id"].nunique() if len(df_cc_cmp) else 0
    std_with_prereq = df_prereq["standard_id"].nunique() if len(df_prereq) else 0
    log.info(
        "Coverage: %d/%d (%.0f%%) have components | %d/%d (%.0f%%) have prereqs",
        std_with_cmp, total, 100 * std_with_cmp / total,
        std_with_prereq, total, 100 * std_with_prereq / total,
    )
    by_subject = df_std.groupby("academic_subject").size().sort_values(ascending=False)
    log.info("CC standards by subject:\n%s", by_subject.to_string())


if __name__ == "__main__":
    main()
