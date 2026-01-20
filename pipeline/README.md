# Pipeline (Stages 1–4)

This folder contains the executable pipeline steps for the project.

## Files
- `data_generator.py` — Stage 1: generate synthetic customer, usage, and billing data.
- `run_sql_pipeline.py` — Stage 2: run SQL staging, quality checks, and feature builds.
- `train_model.py` — Stage 4: train churn models and persist artifacts.

## Run Order

```bash
python pipeline/data_generator.py
python pipeline/run_sql_pipeline.py
python pipeline/train_model.py
```
