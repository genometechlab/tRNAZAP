"""Length-bucketed batching: helpers, and that the pipeline gives the same results."""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pod5
import pytest
import torch

from trnazap.inference.pod5_inferer import (
    DEFAULT_LENGTH_BUCKETS,
    Inference,
    _LengthBucketer,
    bucket_batch_sizes,
    normalize_length_buckets,
    parse_length_buckets,
)

POD5 = Path(__file__).resolve().parents[1] / "example" / "data" / "ecoli" / "ecoli.pod5"

# Long enough max_seq_len that the example reads (~300-700 tokens at stride 18)
# are not all cropped to the same length and spread over several buckets.
CONFIG = dict(
    chunk_size=18, max_seq_len=1000, num_classification_classes=5,
    num_segmentation_classes=4, num_heads=2, num_layers=1, hidden_size=16,
    dim_feedforward=32, norm_first=False, float_dtype="float32",
    label_names={i: f"tRNA-{i}" for i in range(5)},
)
BUCKETS = (350, 450, 550, 650)


# ---------------------------------------------------------------- helpers

def test_default_buckets_are_valid():
    assert normalize_length_buckets(DEFAULT_LENGTH_BUCKETS) == DEFAULT_LENGTH_BUCKETS
    assert len(DEFAULT_LENGTH_BUCKETS) == 20


@pytest.mark.parametrize("bad", [[0, 10], [-5], [10, 10], [20, 10]])
def test_normalize_rejects_bad_boundaries(bad):
    with pytest.raises(ValueError):
        normalize_length_buckets(bad)


@pytest.mark.parametrize("empty", [None, [], ()])
def test_no_boundaries_means_single_bucket(empty):
    assert normalize_length_buckets(empty) == ()


def test_parse_cli_value():
    assert parse_length_buckets("128,192, 256") == (128, 192, 256)
    assert parse_length_buckets("none") == parse_length_buckets("NONE") == ()
    with pytest.raises(argparse.ArgumentTypeError):
        parse_length_buckets("256,128")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_length_buckets("abc")


def test_batch_sizes_without_scaling_are_all_batch_size():
    assert bucket_batch_sizes((100, 200), 64, 1200) == [64, 64, 64]
    assert bucket_batch_sizes((), 64, 1200) == [64]


def test_scaling_one_equalizes_tokens_per_batch():
    bounds, B, L = (96, 200, 600), 1024, 1200
    sizes = bucket_batch_sizes(bounds, B, L, scaling=1.0)
    uppers = list(bounds) + [L]
    for size, upper in zip(sizes, uppers):
        assert size * upper == pytest.approx(B * L, rel=0.01)


@pytest.mark.parametrize("m", [0.0, 0.3, 1.0, 2.5])
def test_longest_bucket_always_uses_batch_size(m):
    sizes = bucket_batch_sizes((96, 200), 1024, 1200, scaling=m)
    assert sizes[-1] == 1024
    assert sizes == sorted(sizes, reverse=True)  # shorter buckets never get smaller batches


def test_scaling_interpolates():
    # bucket [0,100) with max_seq_len 1000: full scale x10
    assert bucket_batch_sizes((100,), 10, 1000, scaling=0.5) == [55, 10]


def test_boundaries_above_max_seq_len_are_clamped():
    # U is capped at max_seq_len, so such a bucket never gets a batch size below batch_size
    assert bucket_batch_sizes((500, 2000), 10, 1000, scaling=1.0) == [20, 10, 10]


def test_negative_scaling_rejected():
    with pytest.raises(ValueError):
        bucket_batch_sizes((100,), 10, 1000, scaling=-0.1)


def _sample(tokens, tag=None):
    return {"inputs": {}, "metadata": {"num_tokens": tokens, "read_id": tag}}


def test_bucketer_assigns_by_half_open_ranges():
    b = _LengthBucketer((10, 20), (100, 100, 100))
    for t in (0, 9, 10, 19, 20, 5000):
        b.add(_sample(t))
    assert [[s["metadata"]["num_tokens"] for s in bk] for bk in b.buckets] == [[0, 9], [10, 19], [20, 5000]]


def test_bucketer_returns_full_batches_and_flushes_rest():
    b = _LengthBucketer((10,), (2, 3))
    assert b.add(_sample(1, "a")) is None
    assert b.add(_sample(50, "x")) is None
    full = b.add(_sample(2, "b"))
    assert [s["metadata"]["read_id"] for s in full] == ["a", "b"]
    assert b.add(_sample(3, "c")) is None
    leftovers = list(b.flush())
    assert [[s["metadata"]["read_id"] for s in bk] for bk in leftovers] == [["c"], ["x"]]
    assert list(b.flush()) == []  # flushing empties the buckets


# ---------------------------------------------------------------- pipeline

needs_pod5 = pytest.mark.skipif(not POD5.exists(), reason="example POD5 not available")


def _engine(**kw):
    """CPU engine without a checkpoint; seeded so every engine has the same random weights."""
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Inference(CONFIG, device="cpu", save_raw=True, **kw)


@pytest.fixture(scope="module")
def plain():
    return _engine(length_buckets=None)


@pytest.fixture(scope="module")
def bucketed():
    return _engine(length_buckets=BUCKETS)


@pytest.fixture(scope="module")
def read_ids():
    with pod5.Reader(POD5) as r:
        return [str(rec.read_id) for _, rec in zip(range(60), r.reads())]


def _logits(results):
    return {r.read_id: r.logits for r in results}


def _assert_same(a, b):
    """Same reads and outputs. Different batch shapes can change float32 rounding
    (~1e-8 on CPU), so logits are compared within tolerance and predictions exactly."""
    assert a.keys() == b.keys()
    for rid in a:
        for task, v in a[rid].items():
            if v is None:
                assert b[rid][task] is None
                continue
            np.testing.assert_allclose(v, b[rid][task], rtol=1e-5, atol=1e-6,
                                       err_msg=f"{rid} {task}")
            np.testing.assert_array_equal(v.argmax(-1), b[rid][task].argmax(-1))


@needs_pod5
@pytest.mark.parametrize("scaling", [0.0, 1.0])
def test_bucketing_gives_identical_results(plain, read_ids, scaling):
    kw = dict(read_ids=read_ids, batch_size=8, show_progress=False)
    engine = _engine(length_buckets=BUCKETS, bucket_batch_scaling=scaling)
    _assert_same(_logits(plain.predict(POD5, **kw)), _logits(engine.predict(POD5, **kw)))


@needs_pod5
def test_batches_hold_one_bucket_each(bucketed, read_ids, monkeypatch):
    seen = []
    original = Inference._run_gpu_batch

    def record(self, batch):
        seen.append(batch["metadata"]["num_tokens"].tolist())
        return original(self, batch)

    monkeypatch.setattr(Inference, "_run_gpu_batch", record)
    bucketed.predict(POD5, read_ids=read_ids, batch_size=4, show_progress=False)
    buckets_per_batch = [set(np.searchsorted(BUCKETS, toks, side="right")) for toks in seen]
    assert all(len(b) == 1 for b in buckets_per_batch)
    assert len(set().union(*buckets_per_batch)) > 1  # the test data spans several buckets
    assert sum(map(len, seen)) == len(read_ids)
    assert all(len(t) <= 4 for t in seen)


@needs_pod5
def test_fewer_reads_than_batch_size_are_flushed(bucketed, read_ids):
    results = bucketed.predict(POD5, read_ids=read_ids[:5], batch_size=1024,
                               show_progress=False)
    assert sorted(results.read_ids) == sorted(read_ids[:5])


@needs_pod5
def test_predict_iter_with_buckets(plain, read_ids):
    engine = _engine(length_buckets=BUCKETS, bucket_batch_scaling=1.0)
    got = _logits(engine.predict_iter(POD5, read_ids=read_ids, batch_size=8))
    ref = _logits(plain.predict(POD5, read_ids=read_ids, batch_size=8, show_progress=False))
    _assert_same(ref, got)


def test_default_engine_uses_default_buckets():
    engine = _engine()
    assert engine.length_buckets == DEFAULT_LENGTH_BUCKETS
    assert engine.bucket_batch_scaling == 0.0


@pytest.mark.parametrize("kw", [dict(length_buckets=[5, 3]), dict(bucket_batch_scaling=-1)])
def test_bad_bucketing_settings_rejected_at_construction(kw):
    with pytest.raises(ValueError):
        _engine(**kw)


# ---------------------------------------------------------------- loader workers & outputs

@needs_pod5
def test_save_raw_segmentation_is_trimmed_to_the_read(bucketed, read_ids, tmp_path):
    from trnazap.io import ZIRReader

    out = tmp_path / "run.zir"
    results = bucketed.predict(POD5, read_ids=read_ids, batch_size=8, output_path=out,
                               show_progress=False)
    for r in results:
        assert r._logits["segmentation"].shape[0] == r.num_chunks
    with ZIRReader(out) as reader:
        for r in reader:
            assert r._logits["segmentation"].shape[0] == r.num_chunks


@needs_pod5
@pytest.mark.parametrize("num_workers", [0, 1, 3])
def test_every_read_processed_exactly_once(num_workers):
    engine = _engine(length_buckets=BUCKETS, num_workers=num_workers)
    with pod5.Reader(POD5) as r:
        all_ids = sorted(str(rec.read_id) for rec in r.reads())
    seen = [res.read_id for res in engine.predict_iter(POD5, batch_size=16)]
    assert sorted(seen) == all_ids  # no read missing, none twice


@needs_pod5
@pytest.mark.parametrize("num_workers", [0, 3])
def test_worker_count_does_not_change_results(plain, read_ids, num_workers):
    engine = _engine(length_buckets=BUCKETS, num_workers=num_workers)
    kw = dict(read_ids=read_ids, batch_size=8, show_progress=False)
    _assert_same(_logits(plain.predict(POD5, **kw)), _logits(engine.predict(POD5, **kw)))


@needs_pod5
def test_early_stop_shuts_down_worker_processes(read_ids):
    import multiprocessing

    engine = _engine(length_buckets=BUCKETS, num_workers=3)
    it = engine.predict_iter(POD5, read_ids=read_ids, batch_size=1)
    next(it)
    assert multiprocessing.active_children()
    it.close()
    assert not multiprocessing.active_children()


def test_negative_num_workers_rejected():
    with pytest.raises(ValueError):
        _engine(num_workers=-1)


# ---------------------------------------------------------------- settings logging


def _table_lines(caplog):
    return [r for r in caplog.records if "batch size per bucket" in r.getMessage()]


@needs_pod5
def test_bucket_table_logged_once_per_engine(read_ids, caplog):
    engine = _engine(length_buckets=BUCKETS, bucket_batch_scaling=1.0, num_workers=0)
    kw = dict(read_ids=read_ids[:10], batch_size=8, show_progress=False)
    with caplog.at_level("INFO", logger="trnazap.inference.pod5_inferer"):
        for _ in range(3):  # like infer-multi running several chunks on one GPU
            engine.predict(POD5, **kw)
        assert len(_table_lines(caplog)) == 1
        engine.predict(POD5, **{**kw, "batch_size": 4})  # new batch size → new table
        assert len(_table_lines(caplog)) == 2


@needs_pod5
def test_settings_logging_can_be_turned_off(read_ids, caplog):
    class Quiet(Inference):
        _log_settings = False

    torch.manual_seed(0)
    with caplog.at_level("INFO", logger="trnazap.inference.pod5_inferer"):
        engine = Quiet(CONFIG, device="cpu", length_buckets=None,
                       bucket_batch_scaling=2.0, num_workers=0)
        engine.predict(POD5, read_ids=read_ids[:10], batch_size=8, show_progress=False)
    assert not [r for r in caplog.records if "bucket_batch_scaling" in r.getMessage()]
    assert not _table_lines(caplog)
