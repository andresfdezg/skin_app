# vibe_skin

Skin lesion classifier for phone photos. MVP stage: a deep learning pipeline trained on HAM10000.

> Research and learning project. Not a medical device and not a diagnostic tool.

## Setup

```bash
uv sync
uv run python scripts/check_env.py
```

## Data

Put HAM10000 (Harvard Dataverse version) in `datasets/ham10000/`. This folder is ignored by git. See [DATA_LICENSE.md](DATA_LICENSE.md) for license and citation.

## Train and evaluate

```bash
uv run python scripts/prepare_data.py                         # splits + data report
uv run python -m vibe_skin.train --config configs/baseline.yaml
uv run python -m vibe_skin.evaluate runs/<run_dir>             # official ISIC 2018 test set
```

Each run writes to `runs/<date>_<name>/`: config, best checkpoint, history, metrics and confusion matrices.

## API

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health`: model name and device.
- `POST /predict`: multipart upload with field `file` (JPEG/PNG). Returns per-class probabilities, top class and malignant probability (mel + bcc + akiec).
- Interactive docs at `http://<pc-ip>:8000/docs`. The served run is set in `configs/api.yaml`.

Single images from the command line: `uv run python -m vibe_skin.predict runs/<run_dir> photo.jpg`

## Tests

```bash
uv run pytest
```
