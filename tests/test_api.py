"""API tests. They need a trained run (configs/api.yaml) and the HAM10000 test images."""
import io

import pytest
from PIL import Image

from vibe_skin.config import load_config, resolve

api_cfg = load_config("configs/api.yaml")
data_cfg = load_config("configs/data_ham10000.yaml")
test_dir = resolve(data_cfg["root"]) / data_cfg["test_image_dir"]

pytestmark = pytest.mark.skipif(
    not (resolve(api_cfg["run_dir"]) / "best.pt").exists() or not test_dir.exists(),
    reason="needs a trained run and the dataset",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_real_image(client):
    path = next(test_dir.glob("*.jpg"))
    r = client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["prediction"]["class"] in body["probabilities"]
    assert 0 <= body["malignant_probability"] <= 1


def test_predict_non_square_png(client):
    buf = io.BytesIO()
    Image.new("RGB", (300, 500), (180, 120, 100)).save(buf, format="PNG")
    r = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_matches_offline_evaluation(client):
    """The API must give the same probabilities as evaluate.py (no train/serve skew).
    Small differences (~1e-3) come from TF32 GPU math with batch size 1 vs 128."""
    import pandas as pd

    preds_file = resolve(api_cfg["run_dir"]) / "predictions_test.csv"
    if not preds_file.exists():
        pytest.skip("run evaluate.py first")
    offline = pd.read_csv(preds_file).head(20)
    for row in offline.itertuples():
        path = test_dir / f"{row.image_id}.jpg"
        r = client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
        probs = r.json()["probabilities"]
        for c, p in probs.items():
            assert abs(p - getattr(row, c)) < 5e-3, (row.image_id, c, p, getattr(row, c))
