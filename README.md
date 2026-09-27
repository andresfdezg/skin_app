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
