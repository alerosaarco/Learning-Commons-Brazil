# BNCC → Learning Commons

Mapping Brazil's **BNCC** (Base Nacional Comum Curricular — the national curriculum for
Educação Infantil, Ensino Fundamental, and Ensino Médio) to the **Learning Commons
Knowledge Graph** via Common Core (US) as a bridge.

The final deliverable is a single `.xlsx` in `data/output/` with, for every BNCC habilidade:

- its Learning Components (granular skills it requires), and
- its Prerequisites (earlier habilidades it depends on),

each row tagged as `Learning Commons` (sourced directly), `AI-matched` (mapped via CC),
or `AI-generated` (synthesised when the KG has no coverage).

Full scope — all 1,710 habilidades across EI, EF, and EM.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/step1_parse_bncc.py` | `data/processed/bncc_standards.csv` — all habilidades |
| 2 | `src/step2_parse_learning_commons.py` | `data/processed/lc_standards.csv` — all CC standards |
| 3 | `src/step3_translate.py` | `data/processed/bncc_translated.csv` — PT → EN |
| 4 | `src/step4_match_bncc_to_cc.py` | `data/processed/bncc_cc_candidates.csv` |
| 5 | `src/step5_learning_components.py` | `data/processed/components.csv` |
| 6 | `src/step6_best_cc_match.py` | `data/processed/bncc_cc_best.csv` |
| 7 | `src/step7_prerequisites.py` | `data/processed/prerequisites.csv` |
| 8 | `src/step8_assemble_xlsx.py` | `data/output/bncc_learning_commons.xlsx` |

Each step reads from `data/processed/` and appends its own checkpoint there,
so re-running a later step never re-does earlier work.

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
# …then step 2, 3, … once each script is written
```

## Data sources

Committed under `data/raw/`:

- `bncc_infantil.json`, `bncc_fundamental.json`, `bncc_medio.json` — from
  `https://cientificar1992.pythonanywhere.com/bncc_{infantil,fundamental,medio}/`.
  These are the primary input for Step 1.
- `BNCC_EI_EF_110518_versaofinal_site.pdf` — the official MEC reference PDF
  (kept for manual verification; no longer used by the pipeline).
- `API_BNCC_{EI,EF,EM}.pdf` — tabular exports from the same API (reference only).

Learning Commons Knowledge Graph data (`nodes.jsonl`, `relationships.jsonl`) is downloaded
by Step 2 from `cdn.learningcommons.org` at run time.

## Repo layout

```
.
├── BNCC_Learning_Commons_Technical_Spec.md   — full spec
├── README.md
├── requirements.txt
├── .env                                      — gitignored
├── src/
│   ├── step1_parse_bncc.py
│   └── … (remaining steps to come)
└── data/
    ├── raw/                                  — committed source files
    ├── processed/                            — generated CSVs (gitignored)
    ├── output/                               — final .xlsx (gitignored)
    └── logs/                                 — per-step logs (gitignored)
```

## Stack

- **Claude Opus 4.7** (`claude-opus-4-7`) for translation, matching, component
  generation, and prerequisite generation.
- `pandas` for tabular work, `openpyxl` for the final `.xlsx`.
- `tenacity` for retry on API calls, `tqdm` for progress bars.

## Current status

- [x] Step 1 — parses 1,710 habilidades (EI=93, EF=1,408, EM=209)
- [ ] Steps 2–8 — in progress
