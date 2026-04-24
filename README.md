# BNCC × Learning Commons

Maps Brazil's **BNCC** (Base Nacional Comum Curricular — all 1,710 habilidades across
Educação Infantil, Ensino Fundamental, and Ensino Médio) to the
**Learning Commons Knowledge Graph**, which organises US Common Core standards into atomic
learning components with prerequisite relationships.

**→ [Explore the website](https://alerosaarco.github.io/Learning-Commons-Brazil/)**

## What this produces

For every BNCC habilidade:

- **Learning components** (3–5 atomic sub-skills in Brazilian Portuguese)
- **Common Core matches** — weighted Jaccard similarity scores against CC standards
- **Prerequisite graph** — which habilidades must be mastered first (Math only; see note below)

## Pipeline

The pipeline uses Common Core as a *bridge*: both curricula are decomposed into atomic
components in the same action-verb style, then matched at component level. This avoids
the mismatch between coarse CC standards and fine-grained BNCC habilidades.

| Step | Script | What it does | Key output |
|------|--------|-------------|------------|
| 1 | `step1_parse_bncc.py` | Parse BNCC JSON → CSV | `bncc_standards.csv` (1,710 rows) |
| 2 | `step2_parse_learning_commons.py` | Parse LC Knowledge Graph | `lc_standards.csv`, `lc_components.csv` |
| 3 | `step3_translate.py` | Translate BNCC PT→EN (for embedding compatibility) | `bncc_translated.csv` |
| 4 | `step4_generate_cc_components.py` | Generate 3–5 atomic components per CC standard | `cc_components_full.csv` (10,242 rows) |
| 4b | `step4b_dedup_components.py` | Deduplicate CC components within each standard (SequenceMatcher ≥ 0.82) | `cc_components_full.csv` (→ 10,136 after dedup) |
| 6 | `step6_generate_bncc_components.py` | Generate 3–5 atomic components per BNCC habilidade, using CC components as style anchors | `bncc_components.csv` (5,494 rows) |
| 6b | `step6b_dedup_bncc_components.py` | Deduplicate BNCC components within each habilidade | `bncc_components.csv` (→ 5,494 after dedup) |
| 7 | `step7_match_components.py` | For each BNCC component: TF-IDF retrieves top-20 subject-filtered CC candidates; Claude classifies each pair into 4 tiers | `bncc_component_matches.csv`, `bncc_components.csv` updated |
| 8 | `step8_jaccard_matching.py` | Weighted Jaccard similarity: BNCC habilidade ↔ CC standard | `bncc_cc_standard_matches.csv` |
| 9a | `step9a_inherit_prerequisites.py` | Inherit LC prerequisite edges into BNCC graph (Math only; score ≥ 0.08; no grade inversions) | `bncc_prerequisites.csv` (290 edges) |
| 10 | `step10_translate_components.py` | Translate BNCC components EN→PT-BR | `bncc_components_pt.csv` |
| 11 | `step11_build_site_data.py` | Build static JSON for website | `docs/data/` |

> **Step 5 was retired.** An early design assigned CC components directly to BNCC habilidades,
> causing many habilidades to receive hundreds of weakly-related components. Steps 6–7 replace
> this with independent component generation + component-level matching.

### Matching tiers (Step 7)

| Tier | Weight | Meaning |
|------|--------|---------|
| **merge** | 1.0 | Identical atomic skill — components are interchangeable |
| **link_regional** | 0.7 | Same skill, Brazilian framing (e.g. modalização, cordel, aglutinação) |
| **link** | 0.5 | Related but genuinely distinct in scope or demand |
| **none** | 0.0 | No meaningful match; Brazil-specific content |

Results: 572 merge · 145 link_regional · 2,927 link · 1,850 none

### Weighted Jaccard formula (Step 8)

```
score(H, S) = Σ weight(tier) / (|H components| + |S components| − Σ weight(tier))
```

Coverage: 1,372 / 1,710 habilidades (80.2%) matched at least one CC standard.
338 unmatched are genuinely Brazil-specific (History, Geography, PE, Religious Ed, Computing).

### Prerequisites note

The Learning Commons only defines prerequisite edges for **Mathematics** (757 edges, all within Math).
ELA and Science have zero prerequisite edges in the LC dataset.
Consequently, this project only shows prerequisites for Math habilidades.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python src/step1_parse_bncc.py
python src/step2_parse_learning_commons.py
python src/step3_translate.py
python src/step4_generate_cc_components.py
python src/step4b_dedup_components.py
# step 5 is retired
python src/step6_generate_bncc_components.py
python src/step6b_dedup_bncc_components.py
python src/step7_match_components.py
python src/step8_jaccard_matching.py
python src/step9a_inherit_prerequisites.py
python src/step10_translate_components.py
python src/step11_build_site_data.py
```

Each step is resume-safe: re-running picks up where it left off.

## Data sources

Committed under `data/raw/`:

- `bncc_infantil.json`, `bncc_fundamental.json`, `bncc_medio.json` — from the BNCC API
- `lc_cc_standards.csv`, `lc_cc_prerequisites.csv`, `lc_cc_components.csv`, `lc_components.csv`
  — Learning Commons Knowledge Graph exports

## Repo layout

```
.
├── README.md
├── requirements.txt
├── src/                    — pipeline scripts (step1 … step11)
├── data/
│   ├── raw/                — committed source files
│   ├── processed/          — generated CSVs (gitignored)
│   └── logs/               — per-step logs (gitignored)
└── docs/                   — static GitHub Pages site
    ├── index.html          — fuzzy search across all 1,710 habilidades
    ├── habilidade.html     — detail page (components, CC matches, prerequisites)
    ├── about.html          — methodology and pipeline description
    └── data/               — pre-built JSON (index + 1,710 per-habilidade files)
```

## Stack

- **Claude Sonnet 4.6** — component generation, translation, matching
- **Claude Haiku 4.5** — PT-BR translation (fast + cheap)
- **scikit-learn** TF-IDF — candidate retrieval for component matching
- **pandas** — all tabular work
- **Fuse.js** — client-side fuzzy search on the website
- **tenacity** — API retry logic; **tqdm** — progress bars

## Website

Enable **GitHub Pages** from the `/docs` folder on the `main` branch.
No build step, no server — all data is pre-computed JSON served as static files.
