import uuid
import warnings
from pathlib import Path

import pod5
import pytest

from trnazap.inference import SingleReadInference
from trnazap.storages.read_results import ReadResultDetailed

POD5 = Path(__file__).resolve().parents[1] / "example" / "data" / "ecoli" / "ecoli.pod5"
pytestmark = pytest.mark.skipif(not POD5.exists(), reason="example POD5 not available")

TINY_CONFIG = dict(
    chunk_size=18, max_seq_len=200, num_classification_classes=5,
    num_segmentation_classes=4, num_heads=2, num_layers=1, hidden_size=16,
    dim_feedforward=32, norm_first=False, float_dtype="float32",
)


def _make(device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no checkpoint: random weights are fine here
        return SingleReadInference(TINY_CONFIG, device=device)


@pytest.fixture(scope="module")
def read_id():
    with pod5.Reader(POD5) as r:
        return str(next(r.reads()).read_id)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_predict(device, read_id):
    result = _make(device).predict(POD5, read_id)
    assert isinstance(result, ReadResultDetailed)
    assert result.read_id == read_id
    assert result.classification_logits.shape == (5,)


def test_predict_with_attention_and_saliency(read_id):
    result, aux = _make("cpu").predict(
        POD5, read_id, return_attention_scores=True, return_token_saliency=True)
    assert result.read_id == read_id
    assert set(aux) == {"attention", "saliency"}
    assert aux["saliency"][0].shape == (result.num_chunks,)


def test_missing_read_raises_key_error():
    with pytest.raises(KeyError, match="not found"):
        _make("cpu").predict(POD5, str(uuid.uuid4()))
