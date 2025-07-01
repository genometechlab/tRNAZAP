# tRNA_zap
tRNA ionic current model and alignment tools


# Splitter Module

The **Splitter Module** enables classification and segmentation of nanopore ionic current signals into biologically relevant regions.

To use the module, 1) download a model and its configuration from the available models section and 2) refer to the example to learn how to use the module

## Available models


---

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

## Inference + Visualization Example

```python
# From the splitter module, import Inference and ResultsVisualizer classes
from tRNA_zap.splitter import Inference, ResultsVisualizer

# Specify your pod5 paths. This can be a single file or a list of directories
pod5_pth = ['Path/To/pod5/file', 'Path/to/pod5/dir1', 'Path/to/pod5/dir2', ...]

# Specify the reads you want to run inference on
desired_reads = [...]  # A list of read UUIDs as strings

# Load inference engine from a configuration file
config_pth = "./configs/zap_s54_c127.yaml"
infer_engine = Inference(config_pth, device="cuda")

# Run inference
results = infer_engine.predict(
    pod5_paths=pod5_pth,
    read_ids=desired_reads,
    batch_size=2048,
    num_workers=4
)
```

You will get an `InferenceResults` object as the return value of `Inference.predict(...)`
An explanation on how to use and interact with this isntance is provided below


## `InferenceResults`

The `InferenceResults` object is a lightweight container that stores all outputs from an inference run, indexed by read ID. It also includes relevant metadata and supports basic persistence and inspection.


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
---

## `ReadResult`

Each value corresponds to a read_id key in `InferenceResults` is a `ReadResult` object. It contains the model outputs for a single read. Probabilities and predictions can be directly accessed from this object.

You do not need to create this class manually — it is returned when you access a read ID from `InferenceResults`.


- #### `read_results.seq2seq_preds -> List[int]`

    Predicted class indices for each chunk in the read (from the seq2seq task).

    ```python
    chunk_predictions = read_result.seq2seq_preds
    ```

- #### `read_results.classification_pred -> int`

    Predicted class index for the whole read.
    > 💡 **Tip:** If you would prefer to get the exact class names instead of the class label index, use the label_names from the InferenceResults instance


    ```python
    label = read_result.classification_pred # Returns class label index
    cls_name = results.label_names[label] # Return the class name
    ```

- #### `read_results.seq2seq_probs -> Optional[np.ndarray]`

    Softmax probabilities for each class at each chunk position.

    ```python
    probs = read_result.seq2seq_probs
    ```

- #### `read_results.classification_probs -> Optional[np.ndarray]`

    Softmax probabilities for the read-level classification task.

    ```python
    probs = read_result.classification_probs
    ```

- #### `read_result.preds`

    Returns a dictionary with both prediction types:
    ```python
    {
        "seq_class": read_result.classification_pred,
        "seq2seq": read_result.seq2seq_preds
    }
    ```

- #### `read_result.probs`

    Returns a dictionary with both probability outputs:
    ```python
    {
        "seq_class": read_result.classification_probs,
        "seq2seq": read_result.seq2seq_probs
    }
    ```