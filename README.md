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

# Initialize the visualizer
visualizer = ResultsVisualizer(results, device="cuda")

# Visualize one of the reads
visualizer.visualize(desired_reads[0]);
