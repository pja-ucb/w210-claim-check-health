# Welcome to the ClaimCheck Health GitHub Repository
-----

Central reference repo for the ClaimCheck Health capstone group, created in 2026 by **Tim Platz, Shruti Mahadevan, Jagriti Singh, and Pedro Alvarez**.

This project is intentionally modular so each workflow owner (Modeling, RAG Architecture, UI) can evolve their piece independently while integrating into a shared end-to-end demo.

## Workstreams & DRIs

This repo is designed to be **customizable by DRIs** for each workflow:

1) **Modeling DRI: Jagriti Singh**
   - Owns model training, export artifacts, and thresholds.
   - Defines model input schema and feature requirements.

2) **RAG Architecture DRI: Shruti Mahadevan**
   - Owns retrieval pipeline, policy documents, and evidence formatting.
   - Defines RAG query/response contract.

3) **UI DRI: Pedro Alvarez**
   - Owns the reviewer-facing UX, CSV validation, and results display.
   - Defines upload template and reporting requirements.

## Repository Layout

The following directories exist in this repo. They store:

- `risk_ui/` End-to-end prototype (backend + frontend + AWS notes).
- `models/` Model artifacts, thresholds, and evaluation summaries.
- `rag/` Retrieval configs, prompts, and policy sources.

## Notes

This README is intentionally high-level. Each workstream can expand its own
documentation and requirements as the MVP evolves.
