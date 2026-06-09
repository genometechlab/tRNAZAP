import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ModelURLS:
    yaml_url: Optional[str]
    weights_url: Optional[str]

YEAST_IVT_URLS = ModelURLS(None, None)
YEAST_BIO_URLS = ModelURLS(None, None)
ECOLI_IVT_URLS = ModelURLS(None, None)
ECOLI_BIO_URLS = ModelURLS(None, None)

AVAILABLE_CONFIGS = {
    "zap_s54_c79_IVTyeast_fragmented": YEAST_IVT_URLS,
    "zap_s54_c79_BIOyeast_fragmented": YEAST_BIO_URLS,
    "zap_s54_c51_IVTecoli_fragmented": ECOLI_IVT_URLS,
    "zap_s54_c51_BIOecoli_fragmented": ECOLI_BIO_URLS,
}

# default_configs.py lives at <root>/src/trnazap/config/default_configs.py
# parents[3] = project root
_PACKAGE_ROOT    = Path(__file__).resolve().parents[3]
_CONFIGS_DIR     = _PACKAGE_ROOT / "configs"
_CHECKPOINTS_DIR = _PACKAGE_ROOT / "checkpoints"

def _download(url: str, dest: Path, label: str) -> None:
    if url is None:
        raise ValueError(
            f"No URL configured for '{label}'. "
            f"This named config is not yet available for download."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} -> {dest}")
    
    urllib.request.urlretrieve(url, dest)


def resolve_named_config(name: str) -> Tuple[Path, Path]:
    """
    Resolve a named config to local yaml and weights paths.
    Downloads both if not already cached.

    Parameters
    ----------
    name : str
        One of AVAILABLE_CONFIGS keys e.g. 'YEAST_IVT'.

    Returns
    -------
    yaml_path, weights_path : Tuple[Path, Path]
    """
    urls = AVAILABLE_CONFIGS[name]

    yaml_path    = _CONFIGS_DIR     / f"{name}.yaml"
    weights_path = _CHECKPOINTS_DIR / f"{name}.pth"

    if not yaml_path.exists():
        _download(urls.yaml_url, yaml_path, label=f"{name} config")
    else:
        print(f"Using cached config: {yaml_path}")

    if not weights_path.exists():
        _download(urls.weights_url, weights_path, label=f"{name} weights")
    else:
        print(f"Using cached weights: {weights_path}")

    return yaml_path