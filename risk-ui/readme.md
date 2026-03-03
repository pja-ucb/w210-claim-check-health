# Risk UI Prototype (Local)

UI-focused README for the `risk_ui/` folder. This prototype provides:
- A batch upload UI
- A FastAPI backend that scores claims
- Local model integration (outpatient default)
- Optional RAG evidence stubbed in results

## Folder Layout

- `frontend/` UI (`index.html`, `app.js`, `style.css`)
- `backend/` FastAPI service
- `simple_app.html` single-file local demo (no backend)
- `deploy/aws.md` AWS deployment notes

## Quick Start (Demo)

### 1) Start backend

```bash
cd /Users/pedro.josealvarez/w210/risk_ui/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

### 2) Start frontend

```bash
cd /Users/pedro.josealvarez/w210/risk_ui/frontend
python3 -m http.server 8001
```

### 3) Open UI

- URL: `http://localhost:8001/index.html`
- API Base URL: `http://localhost:8002`

## CSV Format

Minimal example (SynPUF columns also supported):

```
claim_id,policy_id,claim_amount,claim_type
CLM0001,POL1001,12500,outpatient
CLM0002,POL1002,98000,outpatient
```

If `claim_type` is missing, the UI uses the dropdown default.

## Backend Model Integration

The backend loads local outpatient artifacts by default:

- `outpatient_nn.keras`
- `outpatient_encoder.pkl`
- `outpatient_scaler.pkl`
- `outpatient_feature_cols.pkl`
- `outpatient_thresholds.json`

Default location:
```
/Users/pedro.josealvarez/w210/nn
```

Override with env vars:
```
OUTPATIENT_MODEL_DIR=/path/to/outpatient
INPATIENT_MODEL_DIR=/path/to/inpatient
```

## API

`POST /risk-score-batch`

Body:
```
{"claims":[...], "return_evidence": true}
```

Response:
```
{"results":[...], "summary": {...}}
```

## Notes

- `simple_app.html` runs locally without a backend and uses placeholder scoring.
- Current RAG integration is stubbed in `backend/app/clients/rag.py`.
