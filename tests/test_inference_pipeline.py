import threading
import warnings
from pathlib import Path

import pod5
import pytest

from trnazap.inference.pod5_inferer import Inference
from trnazap.io import ZIRReader

POD5 = Path(__file__).resolve().parents[1] / "example" / "data" / "ecoli" / "ecoli.pod5"
pytestmark = pytest.mark.skipif(not POD5.exists(), reason="example POD5 not available")

TINY_CONFIG = dict(
    chunk_size=18, max_seq_len=200, num_classification_classes=5,
    num_segmentation_classes=4, num_heads=2, num_layers=1, hidden_size=16,
    dim_feedforward=32, norm_first=False, float_dtype="float32",
    label_names={i: f"tRNA-{i}" for i in range(5)},
)


@pytest.fixture(scope="module")
def inferer():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no checkpoint: random weights are fine here
        return Inference(TINY_CONFIG, device="cpu")


@pytest.fixture(scope="module")
def read_ids():
    with pod5.Reader(POD5) as r:
        return [str(rec.read_id) for _, rec in zip(range(40), r.reads())]


def _pipeline_threads():
    return [t for t in threading.enumerate() if t.name in ("pod5-producer", "gpu-consumer")]


def test_predict_writes_readable_archive(tmp_path, inferer, read_ids):
    out = tmp_path / "run.zir"
    results = inferer.predict(POD5, output_path=out, read_ids=read_ids,
                              batch_size=8, show_progress=False)
    assert len(results) == len(read_ids)
    with ZIRReader(out) as reader:
        assert len(reader) == len(read_ids)
    assert not list(tmp_path.glob(".tmp_*"))  # temp dir removed on success
    assert not _pipeline_threads()


def test_batch_failure_keeps_partial_archive_and_raises(tmp_path, inferer, read_ids, monkeypatch):
    calls = {"n": 0}
    original = Inference._run_gpu_batch

    def flaky(self, batch):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated CUDA OOM")
        return original(self, batch)

    monkeypatch.setattr(Inference, "_run_gpu_batch", flaky)
    out = tmp_path / "run.zir"
    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        # batch_size=1 keeps the queue small, so the producer is blocked on it
        inferer.predict(POD5, output_path=out, read_ids=read_ids,
                        batch_size=1, show_progress=False)

    # No .zir that looks complete; the two finished batches stay readable in the temp dir
    assert not out.exists()
    [partial] = tmp_path.glob(".tmp_*/run.zir.partial")
    with ZIRReader(partial) as reader:
        assert len(reader) == 2
        assert len(list(reader)) == 2
    assert not _pipeline_threads()


def test_failed_sharded_run_leaves_only_partials(tmp_path, inferer, read_ids, monkeypatch):
    calls = {"n": 0}
    original = Inference._run_gpu_batch

    def flaky(self, batch):
        calls["n"] += 1
        if calls["n"] == 6:
            raise RuntimeError("simulated failure")
        return original(self, batch)

    monkeypatch.setattr(Inference, "_run_gpu_batch", flaky)
    out = tmp_path / "shards"
    with pytest.raises(RuntimeError):
        inferer.predict(POD5, output_path=out, read_ids=read_ids, batch_size=1,
                        shard_size=2, show_progress=False)

    # Earlier shards were complete, but none may be published as .zir
    assert sorted(p.name for p in out.iterdir()) == [p.name for p in out.glob(".tmp_*")]
    assert len(list(out.glob(".tmp_*/*.zir.partial"))) == 3
    with pytest.raises(ValueError, match="No ZIR file"):
        ZIRReader(out)  # recursive scan must not pick up the temp dir


def test_producer_failure_is_raised(tmp_path, inferer):
    with pytest.raises(RuntimeError, match="Inference worker failed"):
        inferer.predict(POD5, output_path=tmp_path / "run.zir", read_ids=["not-a-uuid"],
                        batch_size=8, show_progress=False)
    assert not (tmp_path / "run.zir").exists()
    assert not _pipeline_threads()


def test_predict_iter_early_stop_shuts_down_threads(inferer, read_ids):
    it = inferer.predict_iter(POD5, read_ids=read_ids, batch_size=1)
    first = next(it)
    assert first.read_id in read_ids
    it.close()  # what `break` in a for-loop does
    assert not _pipeline_threads()


def test_predict_iter_raises_consumer_error(inferer, read_ids, monkeypatch):
    def broken(self, batch):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(Inference, "_run_gpu_batch", broken)
    with pytest.raises(RuntimeError, match="simulated model failure"):
        list(inferer.predict_iter(POD5, read_ids=read_ids, batch_size=4))
    assert not _pipeline_threads()


def test_archive_header_records_run_provenance(tmp_path, inferer, read_ids):
    out = tmp_path / "shards"
    inferer.predict(POD5, output_path=out, read_ids=read_ids, batch_size=8,
                    shard_size=15, show_progress=False, return_results=False)
    shards = sorted(out.glob("*.zir"))
    assert len(shards) == 3
    metas = []
    for p in shards:
        with ZIRReader(p) as reader:
            metas.append(reader.metadata)
    # Written at close: run totals, identical in every shard of the run
    assert {m.num_reads_processed for m in metas} == {len(read_ids)}
    assert all(m.total_inference_time and m.total_inference_time > 0 for m in metas)
    assert len({m.run_id for m in metas}) == 1 and metas[0].run_id
    m = metas[0]
    assert m.tasks == ["classification", "fragmentation", "segmentation"]
    assert m.head_type == "mlp"
    assert m.trnazap_version
    assert m.checkpoint_sha256 is None  # no checkpoint
    assert m.config_name is None        # built from a dict, not a file


def test_staging_dir_run_id_is_shared(tmp_path, inferer, read_ids):
    from trnazap.io.archive_writer import finalize_tmp_dir, make_tmp_dir, run_id_of_tmp_dir

    run_tmp = make_tmp_dir(tmp_path)
    for i, chunk in enumerate((read_ids[:20], read_ids[20:])):
        inferer.predict(POD5, output_path=tmp_path / f"results_{i:05d}.zir", read_ids=chunk,
                        batch_size=8, show_progress=False, return_results=False,
                        staging_dir=run_tmp)
    finalize_tmp_dir(run_tmp, tmp_path)
    for p in sorted(tmp_path.glob("results_*.zir")):
        with ZIRReader(p) as reader:
            assert reader.metadata.run_id == run_id_of_tmp_dir(run_tmp)
            assert reader.metadata.num_reads_processed == 20  # reads of that chunk


def test_model_without_fragmentation_stores_none(tmp_path, read_ids):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inf = Inference(dict(TINY_CONFIG, tasks=["classification", "segmentation"]), device="cpu")
    out = tmp_path / "run.zir"
    inf.predict(POD5, output_path=out, read_ids=read_ids[:8], batch_size=8,
                show_progress=False, return_results=False)
    with ZIRReader(out) as reader:
        assert reader.metadata.tasks == ["classification", "segmentation"]
        reads = list(reader)
    assert all(r.fragmented is None for r in reads)
    assert all(r.variable_region_range is not None for r in reads)


def test_config_name_and_checkpoint_hash_recorded():
    import hashlib

    cfg = Path(__file__).resolve().parents[1] / "configs" / "zap_s54_c51_BIOecoli_fragmented.yaml"
    if not cfg.exists():
        pytest.skip("config not available")
    try:
        inf = Inference(str(cfg), device="cpu")
    except FileNotFoundError:
        pytest.skip("checkpoint not available")
    md = inf._build_metadata(inf._resolve_paths(POD5), batch_size=8)
    assert md.config_name == cfg.name
    ckpt = Path(inf.model_loader.checkpoint_path)
    assert md.checkpoint_sha256 == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert "model_checkpoint" not in md.__dict__  # no machine-specific path


def test_named_config_records_its_yaml_name():
    from trnazap.config.model_config import config_file

    name = "zap_s54_c51_BIOecoli_fragmented"
    try:
        path = config_file(name)  # uses the cached configs/<name>.yaml
    except ValueError:
        pytest.skip("named config not cached locally")
    assert path.name == f"{name}.yaml"
    assert config_file(str(path)).name == path.name  # same name when given the path
