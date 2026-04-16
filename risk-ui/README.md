# ClaimCheck Health — app package

This folder is a **clean copy** of only the **ClaimCheck Health** web app: FastAPI backend, static frontend, offline toy HTML, and deployment notes.

**Not included here (by design):**

- `risk_ui/backend/.venv/` — recreate locally with `python -m venv .venv` and `pip install -r requirements.txt`

**Still required from the parent repo** (paths assume this package lives at `w210/claimcheck_health/`):

- `w210/nn/` — outpatient model artifacts (`outpatient_nn.keras`, encoders, thresholds, etc.) and demo CSVs
- `w210/RAG/` — policy RAG pipeline data and scripts (for `/policy-review`)

---

## Contents

| Path | Purpose |
|------|--------|
| `backend/` | FastAPI app (`app/main.py`, scoring, policy review) |
| `frontend/` | Batch UI: `index.html`, `app.js`, `style.css` |
| `simple_app.html` | Offline heuristic demo (no NN; no backend) |
| `deploy/aws.md` | AWS deployment notes |
| `DEMO.md` | Short end-to-end demo walkthrough |

## Quick start

### 1) Backend

```bash
cd claimcheck_health/backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2) Frontend

```bash
cd claimcheck_health/frontend
python3 -m http.server 8001
```

### 3) Open UI

- **http://localhost:8001/index.html**
- API base URL is set in `frontend/app.js` (default `http://localhost:8000`).

## Model directory

Default artifact path in code points at `w210/nn`. Override with:

```bash
export OUTPATIENT_MODEL_DIR=/path/to/nn
```

## API (summary)

- `GET /health`
- `POST /risk-score-batch` — NN batch scoring
- `GET /policy-review/health` — RAG paths check
- `POST /policy-review` — full policy pipeline for one `claim_id`

See original project docs under `w210/risk_ui/README.md` for CSV format and full detail (this package mirrors that behavior).
