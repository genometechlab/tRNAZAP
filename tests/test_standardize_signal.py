"""standardize_signal must give exactly the numbers SequenceStandardizer gave.

The fast version relies on numpy summing a flat array in the same order as an
(N, 1) array along axis 0. That holds today but isn't guaranteed by numpy, so
this compares bit for bit on real reads: if a numpy upgrade changes it, this
fails instead of model inputs drifting silently.
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pod5
import pytest

from trnazap.feeders import SequenceStandardizer, standardize_signal

DATA = Path(__file__).resolve().parents[1] / "example" / "data"
POD5S = [DATA / "ecoli" / "ecoli.pod5", DATA / "yeast" / "yeast.pod5"]


def _old(x):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return SequenceStandardizer().fit_transform([x.reshape(-1, 1)])[0].ravel()


def _same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture(scope="module")
def signals():
    found = [p for p in POD5S if p.exists()]
    if not found:
        pytest.skip("example POD5 not available")
    out = []
    for p in found:
        with pod5.Reader(p) as r:
            out += [rec.signal.copy() for rec in itertools.islice(r.reads(), 300)]
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bit_identical_on_real_reads(signals, dtype):
    for sig in signals:
        x = sig.astype(dtype)
        assert _same_bits(standardize_signal(x), _old(x))
        # and after the cast preprocessing applies before the model
        assert _same_bits(standardize_signal(x).astype(dtype), _old(x).astype(dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("raw", [
    np.array([], dtype=np.int16),            # empty: mean is NaN
    np.full(5000, 731, dtype=np.int16),      # constant: variance ~0, may round below 0
    np.full(3, -32768, dtype=np.int16),
    np.array([0], dtype=np.int16),
    np.array([1, 2], dtype=np.int16),
])
def test_bit_identical_on_edge_cases(raw, dtype):
    x = raw.astype(dtype)
    assert _same_bits(standardize_signal(x), _old(x))
