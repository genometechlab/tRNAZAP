![tRNAZAP](images/logo.png)
# tRNA_zap
tRNA ionic current model and alignment tools

# Installation

To install the repository, refer to the following instructions:

```bash
git clone https://github.com/genometechlab/tRNA_zap.git
cd tRNA_zap
pip install -e .
```


# Splitter Module

The **Splitter Module** enables classification and segmentation of nanopore ionic current signals into biologically relevant regions.

To use the module, 1) download a model and its configuration from the available models section and 2) refer to the example to learn how to use the module

## Available models


| Model Name           | Config File                             | Model Weights                          | Description                                 |
|----------------------|------------------------------------------|----------------------------------------|---------------------------------------------|
| `zap_s54_c127`       | [`zap_s54_c127.yaml`](./configs/zap_s54_c127.yaml) | [`zap_s54_c127.pth`](./checkpoints/zap_s54_c127.pth) | Standard classifier trained on yeast and ecoli

---

Please download one of the specified models and its config file from the table above and place them in the following structure:


```
<your_project_root>/
├── configs/
│   └── <model_config>.yaml
├── checkpoints/
│   └── <model_weights>.pt
├── your_script.py
└── ...
```

> 💡 **Tip:** The path to the model weights file is specified inside the YAML configuration (`checkpoint_path`). If you want to move the weights to a different directory, be sure to update that path in the config file accordingly.

## Inference run example

```python
# From the splitter module, import Inference and ResultsVisualizer classes
from tRNA_zap.splitter import Inference, ResultsVisualizer

# Specify your pod5 paths. This can be a single file or a list of directories
pod5_pth = ['Path/To/pod5/file', 'Path/to/pod5/dir1', 'Path/to/pod5/dir2', ...]

# Specify the reads you want to run inference on
desired_reads = [...]  # A list of read IDs as strings, 

# Load inference engine from a configuration file
config_pth = "/patch/to/config.yaml"

# Device used for inference. For fast inference, use a cuda-enabled GPU. 
#Can use cpu for a small number of read IDs
device = "cuda" 

infer_engine = Inference(config_pth, device=device)

# Run inference
results = infer_engine.predict(
    pod5_paths=pod5_pth,
    read_ids=desired_reads, # if not provided, will perform inference on every read ID in pod5s
    batch_size=2048, # Number of read IDs to be processed in one batch
)
```

You will get an `InferenceResults` object as the return value of `Inference.predict(...)`
An explanation on how to use and interact with this isntance is provided below


## `InferenceResults`

The `InferenceResults` object is a lightweight container that stores all outputs from an inference run, indexed by read ID. It also includes relevant metadata and supports basic persistence and inspection.

```python
results = infer_engine.predict(...)
```


- #### `results[read_id] -> ReadResult`
    Returns the inference result for a specific read ID. Raises `KeyError` if not found.

    ```python
    read_result = results["read_abc123"]
    ```

    To Check if a read is present
    ```python
    if "read_abc123" in results:
        ...
    ```
- #### `results.read_ids -> List[str]`
    Returns a list of all read IDs in the result set.
    ```python
    all_ids = results.read_ids
    ```

- #### `results.save(path: str | Path) -> None`
    Saves the full results object to a `.pkl` file.
    ```python
    results.save("/path/to/save/results.pkl")
    ```

- #### `InferenceResults.load(path: str | Path) -> InferenceResults` *(class method)*
    Loads a previously saved results object from disk.
    ```python
    results = InferenceResults.load("/path/to/saved/results.pkl")
    ```

- #### `results.summary() -> Dict[str, Any]`

    Returns a dictionary with summary statistics about the inference run:
    - number of reads
    - total chunks
    - chunk size
    - model type
    - device
    - inference timestamp
    - total inference time

    ```python
    summary = results.summary()
    ```

- #### `results.label_names -> Dict`

    Returns a mappind of label indices to class names

    ```python
    summary = results.label_names
    ```

## ReadResult

Each value corresponding to a read_id key in InferenceResults is a ReadResult object. It stores the model outputs for a single read. Probabilities and predictions for both sequence-level and read-level tasks can be accessed directly from this object.

You do not need to create this class manually — it is returned when you access a read ID from InferenceResults:

```python
read_result = inference_results["read_id"]
```

- #### read_result.seq2seq_preds -> Optional[np.ndarray]

    Predicted class indices for each chunk in the read (from the seq2seq task).

    ```python
        chunk_predictions = read_result.seq2seq_preds
    ```

- #### read_result.classification_pred -> Optional[int]

    Predicted class index for the whole read.

    ```python
        label_index = read_result.classification_pred
    ```

- #### read_result.classification_pred_cls -> Optional[str]

    Predicted class label (name) for the whole read. Uses the label names from InferenceMetadata.

    ```python
        label_name = read_result.classification_pred_cls
    ```

- #### read_result.seq2seq_probs -> Optional[np.ndarray]

    Softmax probabilities for each class at each chunk position (seq2seq task).

    ```python
        probs = read_result.seq2seq_probs
    ```

- #### read_result.classification_probs -> Optional[np.ndarray]

    Softmax probabilities for the read-level classification task.

    ```python
        probs = read_result.classification_probs
    ```

- #### read_result.preds -> Dict[str, Any]

    Dictionary containing predictions for both tasks.

    ```python
        {
            "seq_class": read_result.classification_pred,
            "seq2seq": read_result.seq2seq_preds
        }
    ```

- #### read_result.probs -> Dict[str, np.ndarray]

    Dictionary containing probability outputs for both tasks.

    ```python
        {
            "seq_class": read_result.classification_probs,
            "seq2seq": read_result.seq2seq_probs
        }
    ```

- #### read_result.variable_region_range -> Tuple[int, int]

    Start and end indices (inclusive) of the predicted variable region in the chunked signal. Returns (-1, -1) if no region is found.

    ```python
        start, end = read_result.variable_region_range
    ```
    
- #### read_result.get_smoothed_seq2seq_preds(...) -> Optional[np.ndarray]

    Returns smoothed seq2seq predictions using CRF-based smoothing (if available).

    Parameters:
    - device (default: 'cpu'): Device to run the CRF on
    - return_variable_region_range (default: False): Whether to also return (start, end) range

    ```python
        smoothed_preds = read_result.get_smoothed_seq2seq_preds()

        # or with variable region range:
        smoothed_preds, (start, end) = read_result.get_smoothed_seq2seq_preds(return_variable_region_range=True)
    ```
