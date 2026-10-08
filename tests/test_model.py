from pathlib import Path

import pytest
import torch

from trnazap.config.model_config import ModelConfig, ModelLoader
from trnazap.model import tRNAZAPFormer

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))

SMALL = dict(hidden_size=32, num_heads=2, dim_feedforward=64, num_layers=2,
             chunk_size=18, max_seq_len=200, num_classification_classes=7)


def _inputs(n=900, lengths=(900, 400), dtype=torch.float32):
    torch.manual_seed(0)
    return torch.randn(len(lengths), n, dtype=dtype), torch.tensor(lengths)


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.stem)
def test_config_builds_and_runs(cfg_path):
    try:
        cfg = ModelConfig.from_yaml(cfg_path)
    except ValueError as e:
        pytest.skip(f"invalid config: {e}")
    model = ModelLoader(cfg, device=torch.device("cpu")).build_model().eval()
    dtype = next(model.parameters()).dtype
    stride = model.signal_encoder.effective_stride
    n = stride * 20
    signal, length = _inputs(n, (n, n // 2), dtype=dtype)

    with torch.no_grad():
        out = model(signal, length)

    assert set(out) >= model.enabled_tasks
    if "classification" in out:
        assert out["classification"].shape == (2, cfg.num_classification_classes)
    if "segmentation" in out:
        assert out["segmentation"].shape == (2, n // stride, cfg.num_segmentation_classes)
    if "fragmentation" in out:
        assert out["fragmentation"].shape == (2, 2)


@pytest.mark.parametrize("head_type", ["mlp", "linear"])
def test_padding_does_not_change_outputs(head_type):
    model = tRNAZAPFormer(**SMALL, head_type=head_type).eval()
    signal, length = _inputs(900, (540,))
    padded = torch.cat([signal, torch.randn(1, 360)], dim=1)

    with torch.no_grad():
        a = model(signal, length)
        b = model(padded, length)

    for key in ("classification", "fragmentation"):
        torch.testing.assert_close(a[key], b[key], atol=1e-5, rtol=1e-5)
    valid_tokens = 540 // SMALL["chunk_size"]
    torch.testing.assert_close(
        a["segmentation"][:, :valid_tokens], b["segmentation"][:, :valid_tokens],
        atol=1e-5, rtol=1e-5,
    )


def test_head_types():
    mlp = tRNAZAPFormer(**SMALL)
    linear = tRNAZAPFormer(**SMALL, head_type="linear")
    assert isinstance(mlp.classifier, torch.nn.Sequential)
    assert isinstance(linear.classifier, torch.nn.Linear)
    assert linear.frag_classifier.in_features == 2 * SMALL["hidden_size"]
    with pytest.raises(ValueError):
        tRNAZAPFormer(**SMALL, head_type="bogus")


def test_boundary_task():
    tasks = ("classification", "segmentation", "boundary")
    model = tRNAZAPFormer(**SMALL, head_type="linear", enabled_tasks=tasks).eval()
    signal, length = _inputs()
    with torch.no_grad():
        out = model(signal, length)
    # Boundaries are cumulative region fractions: monotone, within [0, 1]
    b = out["boundary"]
    assert b.shape == (2, model.boundary_head.out_features - 1)
    assert torch.all(b[:, 1:] >= b[:, :-1]) and torch.all((b >= 0) & (b <= 1 + 1e-6))
    torch.testing.assert_close(out["boundary_fracs"].sum(-1), torch.ones(2))


def test_task_subset_and_unknown_task():
    model = tRNAZAPFormer(**SMALL, enabled_tasks=("classification",)).eval()
    assert model.frag_classifier is None and model.token_classifier is None
    signal, length = _inputs()
    with torch.no_grad():
        assert set(model(signal, length)) == {"classification"}
        with pytest.raises(ValueError):
            model(signal, length, tasks=("segmentation",))


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.stem)
def test_config_head_type_is_applied(cfg_path):
    try:
        cfg = ModelConfig.from_yaml(cfg_path)
    except ValueError as e:
        pytest.skip(f"invalid config: {e}")
    assert "head_type:" in cfg_path.read_text(), "configs should declare head_type explicitly"
    model = ModelLoader(cfg, device=torch.device("cpu")).build_model()
    assert model.head_type == cfg.head_type


def test_config_head_type_override_and_validation(tmp_path):
    base = dict(chunk_size=18, max_seq_len=200, num_classification_classes=7,
                num_segmentation_classes=4, num_heads=2, num_layers=1, hidden_size=32,
                dim_feedforward=64, norm_first=False)
    default = ModelLoader(ModelConfig(**base), device=torch.device("cpu")).build_model()
    assert default.head_type == "mlp"
    linear = ModelLoader(ModelConfig(**base, head_type="linear"),
                         device=torch.device("cpu")).build_model()
    assert isinstance(linear.classifier, torch.nn.Linear)
    with pytest.raises(ValueError):
        ModelConfig(**base, head_type="bogus")


def test_load_weights_strict(tmp_path):
    from trnazap.utils import load_weights

    model = tRNAZAPFormer(**SMALL)
    state = model.state_dict()
    path = tmp_path / "ok.pth"

    # The identity stem's frozen conv kernel may be absent from checkpoints
    torch.save({k: v for k, v in state.items() if k != "signal_encoder.conv.weight"}, path)
    load_weights(tRNAZAPFormer(**SMALL), path)

    # A head-type mismatch must fail instead of leaving heads randomly initialized
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="does not match"):
        load_weights(tRNAZAPFormer(**SMALL, head_type="linear"), path)

    torch.save({k: v for k, v in state.items() if not k.startswith("classifier.")}, path)
    with pytest.raises(RuntimeError, match="Missing keys"):
        load_weights(tRNAZAPFormer(**SMALL), path)
