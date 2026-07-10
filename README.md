<img src="images/logo.png" alt="tRNAZAP" width="300"/>

tRNA ionic current model and alignment tools

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Available models](#available-models)
- [1. Inference](#1-inference)
  - [1.1 Running inference](#11-running-inference)
    - [Python API](#python-api)
    - [CLI — `infer` and `infer-multi`](#cli--infer-and-infer-multi)
  - [1.2 Working with results](#12-working-with-results)
    - [In memory — `InferenceResults`](#in-memory--inferenceresults)
    - [On disk — `ZIRReader`](#on-disk--zirreader)
    - [The read object — `ReadResultDetailed` vs `ReadResult`](#the-read-object--readresultdetailed-vs-readresult)
  - [1.3 Inference visualization](#13-inference-visualization)
- [2. Alignment](#2-alignment)
  - [2.1 `trnazap align`](#21-trnazap-align)
  - [2.2 `trnazap alignment_visualize`](#22-trnazap-alignment_visualize)
- [End-to-end example](#end-to-end-example)

# Installation

```bash
git clone https://github.com/genometechlab/tRNA_zap.git
cd tRNA_zap
pip install -e .
```

> ⚠️ The `trnazap align` and `trnazap alignment_visualize` commands require
> [`samtools`](https://www.htslib.org) on your `PATH`.

# Overview

`trnazap` classifies and segments nanopore ionic-current signals into
biologically relevant regions (variable region, ONT adapter, 3′ and 5′ splint),
and aligns the resulting reads against reference tRNA databases.

The package exposes two interfaces that share the same underlying code:

1. A **command-line interface** (`trnazap <command>`) — the recommended entry
   point, used by [`example/run_example.sh`](./example/run_example.sh).
2. A **Python API** (`from trnazap import ...`) for notebook / programmatic use.

A typical workflow is: **infer** (POD5 → `.zir` inference archive) → **align**
(`.zir` + basecalled BAM → aligned BAM) → **visualize** the alignment.

# Available models

Download a model's config (`.yaml`) and weights (`.pth`) and place them like so:

```
<your_project_root>/
├── configs/
│   └── <model_config>.yaml
├── checkpoints/
│   └── <model_weights>.pth
├── your_script.py
└── ...
```

| Model / config                                                                                          | Substrate        | Weights            |
|---------------------------------------------------------------------------------------------------------|------------------|--------------------|
| [`zap_s54_c79_BIOyeast_fragmented.yaml`](./configs/zap_s54_c79_BIOyeast_fragmented.yaml)                | Yeast (biological) | _(add link)_     |
| [`zap_s54_c79_IVTyeast_fragmented.yaml`](./configs/zap_s54_c79_IVTyeast_fragmented.yaml)                | Yeast (IVT)      | _(add link)_       |
| [`zap_s54_c51_BIOecoli_fragmented.yaml`](./configs/zap_s54_c51_BIOecoli_fragmented.yaml)                | E. coli (biological) | _(add link)_   |
| [`zap_s54_c51_IVTecoli_fragmented.yaml`](./configs/zap_s54_c51_IVTecoli_fragmented.yaml)                | E. coli (IVT)    | _(add link)_       |

> 💡 The path to the weights is set inside the YAML config (`checkpoint_path`).
> If you move the weights, update that path accordingly.

# 1. Inference

Inference classifies each read and segments it into its regions
(variable region, ONT adapter, 3′/5′ splint). Running it involves three
independent choices:

1. **How you run it** — the [Python API](#python-api) or the
   [CLI](#cli--infer-and-infer-multi).
2. **Where results go** — held [in memory](#in-memory--inferenceresults)
   (Python API only) and/or written to a `.zir` archive
   [on disk](#on-disk--zirreader) (both interfaces).
3. **How much detail per read** — the full
   [`ReadResultDetailed`](#the-read-object--readresultdetailed-vs-readresult)
   or the lightweight `ReadResult`, selected with a single `save_raw` flag.

Whichever combination you pick, the storage containers (`InferenceResults`,
`ZIRReader`) share one access interface, and the per-read objects
(`ReadResultDetailed`, `ReadResult`) share another — so downstream code is the
same regardless of how inference was run. Those shared types are documented once
in [§1.2](#12-working-with-results).

## 1.1 Running inference

### Python API

```python
from trnazap import Inference

# POD5 input: a single file, a directory, or a list of either
pod5_pth = ["Path/to/pod5/dir1", "Path/to/file.pod5"]

# device: "cuda" for GPU, "cpu" for a few reads, or None to auto-select CUDA.
# save_raw=True keeps full ReadResultDetailed reads; the default compresses to ReadResult.
infer_engine = Inference(
    "configs/zap_s54_c79_BIOyeast_fragmented.yaml",
    device="cuda",
    save_raw=False,
)

results = infer_engine.predict(
    pod5_paths=pod5_pth,
    read_ids=None,     # None = all reads in the POD5(s); or pass a list of read-id strings
    batch_size=32,     # reads per batch
    output_path=None,  # set to a .zir path to also write results to disk
    return_results=True,
)
```

`predict(...)` can deliver its output **in RAM**, **on disk**, or **both** —
controlled by `return_results` and `output_path`:

| `return_results` | `output_path` | Result |
|------------------|---------------|--------|
| `True` (default) | `None` | Returns an `InferenceResults` held in memory. Best for notebooks / small runs. |
| `False` | `"run.zir"` | Streams straight to a `.zir` on disk, returns `None`. Memory-efficient for large datasets. |
| `True` | `"run.zir"` | Does both: returns `InferenceResults` **and** writes the `.zir`. |

> ⚠️ `return_results=False` with `output_path=None` does nothing and raises a
> `ValueError`.

### CLI — `infer` and `infer-multi`

The CLI always writes results to a `.zir` archive on disk (there is no in-memory
mode); load them back later with a [`ZIRReader`](#on-disk--zirreader), or feed
them straight into [`trnazap align`](#21-trnazap-align).

#### `trnazap infer` (single device)

```bash
trnazap infer \
  --input   ./data/yeast/yeast.pod5 \
  --config  ./configs/zap_s54_c79_BIOyeast_fragmented.yaml \
  --output  ./data/yeast/yeast_zap_inference.zir \
  --batch-size 1024
```

| Flag | Description |
|------|-------------|
| `--input`, `-i` | POD5 file, several POD5 files, or a directory of POD5s (**required**) |
| `--config`, `-c` | Model config YAML (**required**) |
| `--output`, `-o` | Output `.zir` file (or directory, if sharding) |
| `--batch-size` | Reads per inference batch (default: `32`) |
| `--shard-size` | Reads per output shard when writing `.zir` (default: single file) |
| `--save-raw` | Keep full `ReadResultDetailed` reads (default: compress to `ReadResult`) |
| `--no-progress` | Disable progress bars |
| `--force-cpu` | Force CPU-only inference (default: use CUDA if available) |

#### `trnazap infer-multi` (multi-GPU)

For large datasets on a multi-GPU machine. It collects all read IDs, splits them
into shards, and runs one worker per GPU. Each worker writes its own
`results_<shard_idx>.zir` into the results directory.

```bash
trnazap infer-multi \
  --input       ./data/big_run/ \
  --config      ./configs/zap_s54_c79_BIOyeast_fragmented.yaml \
  --results-dir ./data/big_run/zap_inference/ \
  --shard-size  500000 \
  --batch-size  1024
```

| Flag | Description |
|------|-------------|
| `--input`, `-i` | POD5 file or directory of POD5s (**required**) |
| `--config`, `-c` | Model config YAML (**required**) |
| `--results-dir`, `-o` | Directory for the output shard `.zir` files (default: current dir) |
| `--shard-size` | Reads per shard handed to a GPU worker (default: `1000000`) |
| `--batch-size` | Reads per batch, per worker (default: `1024`) |
| `--gpus` | Number of GPUs to use (default: all detected via `torch.cuda.device_count()`) |
| `--save-raw` | Keep full `ReadResultDetailed` reads (default: compress to `ReadResult`) |
| `--no-progress` | Disable per-worker progress bars |

> 💡 The result is several `.zir` shards. Feed them all to `trnazap align` at
> once (`--inference shard0.zir shard1.zir ...`), or open them together with a
> single `ZIRReader([...])`.

## 1.2 Working with results

Inference produces a **container** of reads (in RAM or on disk) holding
**per-read objects**. Both come in two interchangeable forms.

- **Containers** — `InferenceResults` (in memory) and `ZIRReader` (on disk) share
  the same access interface: `read_ids`, `get(read_id)`, membership, and
  iteration over read objects.
- **Read objects** — `ReadResultDetailed` (full) and `ReadResult` (compressed)
  share the `ReadResultLike` interface: `read_id`, `classification_pred`,
  `variable_region_range`, `copy()`.

### In memory — `InferenceResults`

Returned by `predict(..., return_results=True)`.

```python
# Access / membership
read_result = results["read_abc123"]      # -> read object (raises KeyError if absent)
maybe       = results.get("read_abc123")  # -> read object or None
"read_abc123" in results                  # -> bool

# Enumerate
results.read_ids            # List[str] of all read IDs
len(results)                # number of reads
for r in results: ...       # iterates over read OBJECTS (not IDs)
for rid, r in results.items(): ...

# Label index -> class-name mapping (from the config)
results.label_names

# Summary statistics
results.summary()           # {num_reads, chunk_size, model_name, inference_time}

# Persist to the ZIR archive format for later
results.save_zir("/path/to/results.zir", shard_size=None)
```

### On disk — `ZIRReader`

Reads back a `.zir` written by `predict(output_path=...)`, `save_zir(...)`, or the
CLI. Same access interface as `InferenceResults`, so the same loops work.

```python
from trnazap import ZIRReader

with ZIRReader("/path/to/results.zir") as store:
    rids        = store.read_ids               # List[str]
    read_result = store.get("read_abc123")     # -> read object or None
    strict      = store.get_read("read_abc123")# same, but raises on a missing read

    for r in store.reads():                    # stream all reads (optionally reads(selection={...}))
        ...
```

`ZIRReader` also accepts a **list of `.zir` files** (e.g. the shards produced by
`infer-multi`), presenting them as one combined store:

```python
store = ZIRReader(["results_0.zir", "results_1.zir", "results_2.zir"])
```

### The read object — `ReadResultDetailed` vs `ReadResult`

Which form the reads take is set at inference time by `save_raw`
(`save_raw=True` / `--save-raw` → detailed; default → compressed). Both are
yielded transparently by `InferenceResults` and `ZIRReader`.

**Shared members** (`ReadResultLike` — available on either form):

```python
r.read_id
r.classification_pred     # top predicted class index, or None
r.variable_region_range   # (start, end) signal indices; (-1, -1) if none found
r.copy()

results.label_names[r.classification_pred]   # map index -> class name
```

**`ReadResultDetailed`** (full — `save_raw=True`) additionally exposes logits,
probabilities and predictions for all three tasks (**classification**,
**segmentation**, **fragmentation**), plus CRF smoothing:

```python
# Classification (read-level)
r.classification_probs         # softmax probabilities
r.topk_classes(k=3)            # top-k class indices, largest first

# Segmentation (per-chunk)
r.segmentation_preds           # np.ndarray of class indices per chunk
r.segmentation_probs           # per-chunk softmax probabilities

# Fragmentation
r.fragmentation_pred
r.fragmentation_probs

# Grouped views (dicts keyed by 'classification' / 'segmentation' / 'fragmentation')
r.preds
r.probs
r.logits

# CRF-smoothed segmentation predictions
smoothed = r.get_smoothed_segmentation_preds()             # device='cpu' by default
smoothed, (start, end) = r.get_smoothed_segmentation_preds(
    return_variable_region_range=True,
)
```

**`ReadResult`** (compressed — the default) keeps only a small summary and drops
the per-chunk arrays, probabilities, logits, and smoothing:

```python
r.top3_classes            # np.ndarray of the top-3 class indices (largest first)
r.classification_pred     # == top3_classes[0]
r.variable_region_range   # from the raw (argmax) segmentation
r.fragmented              # bool: was the read predicted fragmented?
```

> 💡 To use detailed-only members after inference, guard with
> `isinstance(r, ReadResultDetailed)` — compressed reads won't have them.

## 1.3 Inference visualization

Inference visualization is provided through the **Python API** class
`ZAPInferenceVisualizer`, which plots a read's raw signal together with the
model's predicted segments. The constructor takes the **POD5 path(s)** (so it
can load the raw signal), and `visualize(...)` takes **read objects**
(`ReadResultDetailed` / `ReadResult`), not read-id strings.

```python
from trnazap import Inference, ZAPInferenceVisualizer

pod5_pth   = "data.pod5"
config_pth = "configs/zap_s54_c79_BIOyeast_fragmented.yaml"

results = Inference(config_pth, device="cuda", save_raw=True).predict(pod5_pth)

vis = ZAPInferenceVisualizer(pod5_pth, config=config_pth)

# Single read -> one Figure; list of reads -> list of Figures
fig  = vis.visualize(results["read_abc123"])
figs = vis.visualize([results[r] for r in results.read_ids[:5]])
```

`visualize(...)` accepts:

- `apply_crf_smoothing` (default `True`) — overlay CRF-smoothed segmentation
- `plot_probabilities` (default `True`)
- `plot_signal` (default `True`)
- `ground_truth_segmentations` — optional list of per-chunk ground-truth labels
- `figure_size` (default `(16, 8)`)

> ℹ️ Run inference with `save_raw=True` for the richest plots — the segmentation
> overlay and probability track need the detailed reads. A `trnazap
> inference_visualize` CLI subcommand is registered but **not yet implemented**;
> use the Python API above for now.

# 2. Alignment

## 2.1 `trnazap align`

Aligns inferred reads against the reference tRNA database, combining a
basecalled BAM with one or more `.zir` inference archives (e.g. the shards from
`infer-multi`).

```bash
trnazap align \
  --unaligned_bam ./data/yeast/yeast_unaligned_dorado_1.0.0.emit_moves.calmd.bam \
  --inference     ./data/yeast/yeast_zap_inference.zir \
  --out_dir       ./data/yeast \
  --out_pre       yeast_aligned_zap \
  --model         yeast \
  --threads       4
```

| Flag | Description |
|------|-------------|
| `--unaligned_bam`, `-ub` | Basecalled BAM paired with the POD5 input (**required**) |
| `--inference`, `-i` | One or more `.zir` inference results (**required**; accepts multiple runs/shards) |
| `--out_dir`, `-od` | Output directory (created if missing) (**required**) |
| `--out_pre`, `-op` | Prefix prepended to output files (**required**) |
| `--model`, `-m` | Target substrate: `yeast` or `e_coli` (default: `e_coli`) |
| `--threads`, `-t` | Worker threads (default: `8`) |
| `--secondary`, `-s` | Also align the second-highest classification and keep the better alignment |
| `--ident_threshold` | Minimum identity for a read to be kept (default: `0.75`) |
| `--wf_gap_open` / `--wf_gap_extend` | Wagner-Fisher gap penalties (defaults: `2.0` / `0.5`) |
| `--sw_gap_open` / `--sw_gap_extend` | Smith-Waterman gap penalties (defaults: `-6.0` / `-1.0`) |
| `--sw_match` / `--sw_mismatch` | Smith-Waterman match / mismatch scores (defaults: `3.0` / `1.0`) |

## 2.2 `trnazap alignment_visualize`

Generates comparison figures between an existing aligner (e.g. `bwa`) and the
tRNA-zap alignment.

```bash
trnazap alignment_visualize \
  --model      yeast \
  --out_dir    ./comparison_results/yeast/ \
  --out_prefix yeast_ \
  --bwa_path   ./data/yeast/yeast_aligned_bwa.bam \
  --zap_path   ./data/yeast/yeast_aligned_zap.sorted.bam \
  --threads    8
```

| Flag | Description |
|------|-------------|
| `--model`, `-m` | `yeast` or `e_coli` (**required**) |
| `--out_dir`, `-od` | Output directory (**required**) |
| `--out_prefix`, `-op` | Output file prefix (**required**) |
| `--bwa_path` | Existing BWA alignment BAM |
| `--zap_path` | tRNA-zap alignment BAM |
| `--threads`, `-t` | Worker threads (default: `8`) |
| `--reference` | Custom BWA reference FASTA (overrides the bundled reference for the selected model) |

# End-to-end example

A complete, runnable pipeline (infer → align → visualize) for bundled yeast and
E. coli data lives in [`example/run_example.sh`](./example/run_example.sh).
