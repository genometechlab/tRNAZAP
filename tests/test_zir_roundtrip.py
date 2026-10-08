import numpy as np
import pytest

from trnazap.io import ZIRReader, ZIRShardManager, ZIRWriter
from trnazap.io.archive_writer import finalize_tmp_dir, make_tmp_dir, write_zir
from trnazap.storages.inference_metadata import InferenceMetadata
from trnazap.storages.read_results import ReadResult, ReadResultDetailed


def _metadata():
    return InferenceMetadata(
        chunk_size=18, max_seq_len=200, model_type="test", model_name="tRNAZAPFormer",
        num_classification_classes=5, num_segmentation_classes=4,
        label_names={str(i): f"tRNA-{i}" for i in range(5)},
        batch_size=2, device="cpu", float_dtype="float32",
    )


def _detailed(i, rng):
    n = 10 + i
    return ReadResultDetailed(
        read_id=f"read-{i:04d}",
        _logits={
            "classification": rng.standard_normal(5).astype(np.float32),
            "segmentation": rng.standard_normal((n, 4)).astype(np.float32),
            "fragmentation": rng.standard_normal(2).astype(np.float32),
        },
        num_chunks=n, chunk_size=18, cropped=bool(i % 2),
    )


@pytest.fixture
def reads():
    rng = np.random.default_rng(0)
    return [_detailed(i, rng) for i in range(7)]


def test_detailed_roundtrip(tmp_path, reads):
    path = tmp_path / "out.zir"
    with ZIRWriter(path, _metadata()) as w:
        for r in reads:
            w.add_result(r)

    with ZIRReader(path) as reader:
        assert len(reader) == len(reads)
        assert reader.metadata.label_names == _metadata().label_names
        back = list(reader)

    assert [r.read_id for r in back] == [r.read_id for r in reads]
    for orig, got in zip(reads, back):
        assert (got.num_chunks, got.chunk_size, got.cropped) == (
            orig.num_chunks, orig.chunk_size, orig.cropped)
        for key in ("classification", "segmentation", "fragmentation"):
            np.testing.assert_array_equal(got.logits[key], orig.logits[key])


def test_compressed_roundtrip(tmp_path, reads):
    compressed = [r.to_compressed() for r in reads]
    path = tmp_path / "out.zir"
    with ZIRWriter(path, _metadata()) as w:
        for r in compressed:
            w.add_result(r)

    with ZIRReader(path) as reader:
        back = list(reader)

    for orig, got in zip(compressed, back):
        assert isinstance(got, ReadResult)
        np.testing.assert_array_equal(got.top3_classes, orig.top3_classes)
        assert tuple(got.variable_region_range) == tuple(orig.variable_region_range)
        assert got.fragmented == orig.fragmented


def test_indexed_lookup(tmp_path, reads):
    path = tmp_path / "out.zir"
    with ZIRWriter(path, _metadata()) as w:
        for r in reads:
            w.add_result(r)

    with ZIRReader(path, index=True) as reader:
        target = reads[3]
        assert target.read_id in reader
        got = reader.get(target.read_id)
        np.testing.assert_array_equal(got.logits["segmentation"], target.logits["segmentation"])
        assert reader.get("missing") is None


def test_sharded_roundtrip(tmp_path, reads):
    manager = ZIRShardManager(tmp_path / "shards" / "run", _metadata(), 3)
    try:
        for r in reads:
            manager.add_result(r)
    finally:
        manager.close()

    shards = sorted((tmp_path / "shards").rglob("*.zir"))
    assert len(shards) == 3
    with ZIRReader(tmp_path / "shards") as reader:
        assert sorted(r.read_id for r in reader) == sorted(r.read_id for r in reads)


def test_get_during_iteration_and_nested_iteration(tmp_path, reads):
    for shard in ("a", "b"):
        with ZIRWriter(tmp_path / f"{shard}.zir", _metadata()) as w:
            for r in reads:
                w.add_result(ReadResultDetailed(
                    read_id=f"{shard}-{r.read_id}", _logits=r._logits,
                    num_chunks=r.num_chunks, chunk_size=r.chunk_size, cropped=r.cropped))

    with ZIRReader(tmp_path, index=True) as reader:
        expected = [r.read_id for r in reader]
        seen = []
        for r in reader:
            seen.append(r.read_id)
            # random access and a nested pass must not disturb the outer iterator
            assert reader.get(expected[-1]).read_id == expected[-1]
            assert sum(1 for _ in reader.reads(selection={expected[0]})) == 1
        assert seen == expected
        assert len(seen) == 2 * len(reads)


def test_shard_manager_publishes_only_on_success(tmp_path, reads):
    out = tmp_path / "shards"
    with pytest.raises(RuntimeError):
        with ZIRShardManager(out, _metadata(), 3) as manager:
            for r in reads[:5]:
                manager.add_result(r)
            raise RuntimeError("boom")
    assert not list(out.glob("*.zir"))
    [failed_tmp] = out.glob(".tmp_*")
    partials = sorted(failed_tmp.glob("*.zir.partial"))
    assert len(partials) == 2
    # A partial named explicitly is still readable (salvage); directory scans skip it
    with ZIRReader(partials[0]) as reader:
        assert len(reader) == 3
    with pytest.raises(ValueError, match="No ZIR file"):
        ZIRReader(out)

    # A successful re-run publishes .zir, removes its own temp dir, and leaves the
    # failed run's temp dir untouched for the user to inspect or delete
    with ZIRShardManager(out, _metadata(), 3) as manager:
        for r in reads:
            manager.add_result(r)
    assert sorted(p.name for p in out.glob("*.zir")) == [
        "shard0000.zir", "shard0001.zir", "shard0002.zir"]
    assert list(out.glob(".tmp_*")) == [failed_tmp]
    with ZIRReader(out) as reader:
        assert len(reader) == len(reads)


def test_concurrent_managers_do_not_share_files(tmp_path, reads):
    a = ZIRShardManager(tmp_path / "a.zir", _metadata())
    b = ZIRShardManager(tmp_path / "a.zir", _metadata())
    assert a.tmp_dir != b.tmp_dir
    a.add_result(reads[0])
    b.add_result(reads[1])
    a.close()
    b.close()
    with ZIRReader(tmp_path / "a.zir") as reader:  # last finisher wins, file intact
        assert [r.read_id for r in reader] == [reads[1].read_id]
    assert not list(tmp_path.glob(".tmp_*"))


def test_write_zir_returns_output_location(tmp_path, reads):
    with ZIRWriter(tmp_path / "src.zir", _metadata()) as w:
        for r in reads:
            w.add_result(r)
    with ZIRReader(tmp_path / "src.zir") as store:
        single = write_zir(store, tmp_path / "copy.zir")
        sharded = write_zir(store, tmp_path / "resharded", shard_size=4)
    assert single == tmp_path / "copy.zir" and single.exists()
    assert sharded == tmp_path / "resharded"
    with ZIRReader(sharded) as reader:
        assert len(reader) == len(reads)


def test_managers_sharing_a_staging_dir_finalize_together(tmp_path, reads):
    # The infer-multi layout: the caller owns one temp dir, each chunk's manager
    # writes into it, and nothing is published until the caller finalizes it.
    run_tmp = make_tmp_dir(tmp_path)
    for i, chunk in enumerate((reads[:4], reads[4:])):
        with ZIRShardManager(tmp_path / f"results_{i:05d}.zir", _metadata(),
                             staging_dir=run_tmp) as manager:
            for r in chunk:
                manager.add_result(r)
    assert list(tmp_path.glob(".tmp_*")) == [run_tmp]  # no per-manager temp dirs
    assert not list(tmp_path.glob("*.zir"))
    assert sorted(p.name for p in run_tmp.iterdir()) == [
        "results_00000.zir.partial", "results_00001.zir.partial"]

    finals = finalize_tmp_dir(run_tmp, tmp_path)
    assert [p.name for p in finals] == ["results_00000.zir", "results_00001.zir"]
    assert not run_tmp.exists()
    with ZIRReader(tmp_path) as reader:
        assert [r.read_id for r in reader] == [r.read_id for r in reads]


def test_staging_dir_abort_leaves_partials_without_publishing(tmp_path, reads):
    run_tmp = make_tmp_dir(tmp_path)
    with pytest.raises(RuntimeError):
        with ZIRShardManager(tmp_path / "results_00000.zir", _metadata(),
                             staging_dir=run_tmp) as manager:
            manager.add_result(reads[0])
            raise RuntimeError("boom")
    assert not list(tmp_path.glob("*.zir"))
    with ZIRReader(run_tmp / "results_00000.zir.partial") as reader:
        assert len(reader) == 1


def test_staging_dir_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        ZIRShardManager(tmp_path / "a.zir", _metadata(), staging_dir=tmp_path / "missing")
