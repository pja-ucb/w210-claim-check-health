# End-to-end demo (1–2 claims)

This walkthrough uses the **sample 100** outpatient claims file that matches both the ML batch scorer and the RAG claims lookup.

## Prerequisites

- Python env with `claimcheck_health/backend` dependencies installed (see `requirements.txt`).
- `OPENAI_API_KEY` set for the RAG pipeline (e.g. `w210/RAG/.env` or shell).
- Chroma DB and RAG data present under `w210/RAG/`.

## Start servers

From `claimcheck_health/backend`:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Serve the static UI (from `claimcheck_health/frontend`), for example:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080` (or your chosen port).

## Demo flow

### Minimal two-claim contrast (recommended)

1. Upload `w210/nn/two_claim_demo.csv` (two rows: **low** vs **high** `High_Risk`).
2. **Run scoring**, then **Generate Context** on each row.

| Claim ID (CLM_ID) | Role | Overlap + contrast |
|-------------------|------|---------------------|
| `542832281268681` | Low risk | Shares **ICD-9 71596** (OA lower leg) and **HCPCS 73562, 97110, 97530** with the high claim — and has **only** those three HCPCS (all map in CMS). |
| `542052281361022` | High risk | Same MSK/therapy/imaging codes **plus** dozens of additional HCPCS (44 distinct lines). |

Set `RAG_CLAIMS_CSV` to this same file if your API default does not already include both IDs (default `..._rag_default.csv` does).

### Full sample (100 rows)

1. **Upload CSV:** choose  
   `w210/nn/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1_labeled_sample_100.csv`
2. Click **Run scoring.** Wait for the table (combined score, NN probability, flag).
3. **Generate Context** on either of these rows (both exist in the claims file used by RAG):

   | Claim ID (CLM_ID) | Note |
   |-------------------|------|
   | `542192281063886` | First row in sample; lab-style HCPCS |
   | `542272281166593` | Multi-HCPCS row |

4. The page scrolls to **Generate context** and runs the pipeline automatically (~minutes). When it finishes, you should see **Recommended action**, per-HCPCS sections, and expandable full JSON.

Optional: enter the same ID manually and click **Generate Context** in the panel to repeat without rescoring.

## Troubleshooting

- **Claim not in claims file:** use IDs from `..._rag_default.csv` (default for RAG), or set `RAG_CLAIMS_CSV` to the full labeled CSV / your upload extract.
- **429 / quota:** OpenAI billing or key limits; pipeline will fail until resolved.
- **Data paths:** use **Check data paths** in the UI or `GET http://localhost:8000/policy-review/health`.
