import os
import warnings
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import tqdm

from ..feeders import Pod5IterDataset, SequenceStandardizer
from .config import ModelConfig, ModelLoader
from ..storages import InferenceResults, InferenceMetadata, ReadResult

warnings.filterwarnings("ignore", category=UserWarning)


class Inference:
    """Main inference engine for running predictions on Pod5 files."""
    
    def __init__(
        self,
        config: Union[ModelConfig, str, Dict[str, Any]],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize InferenceEngine.
        
        Args:
            config: ModelConfig instance, path to config file, or config dictionary
            device: torch device to run inference on
        """
        # Load configuration
        if isinstance(config, str):
            if config.endswith('.yaml'):
                self.config = ModelConfig.from_yaml(config)
            elif config.endswith('.json'):
                self.config = ModelConfig.from_json(config)
            else:
                raise ValueError(f"Unsupported config file format: {config}")
        elif isinstance(config, dict):
            self.config = ModelConfig.from_dict(config)
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            raise TypeError("config must be ModelConfig, path to config file, or dict")
        
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model loader
        self.model_loader = ModelLoader(self.config, self.device)
        self.model = None
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize and load model."""
        self.model = self.model_loader.get_model(load_checkpoint=True)
        self.model.eval()
        
    @staticmethod
    def _local_standardizer(signal):
        """Standardize the input signal."""
        standardizer = SequenceStandardizer()
        return standardizer.fit_transform([signal.reshape(-1, 1)])[0].flatten()
    
    @staticmethod
    def _collate_fn(batch):
        """Collate function for DataLoader."""
        assert len(batch) == 1
        batch = batch[0]
        
        signals = [torch.tensor(item["inputs"]["signal"]) for item in batch]
        lengths_list = [item["inputs"]["length"] for item in batch]
        lengths_tensor = torch.tensor(lengths_list)
        lengths_numpy = np.array(lengths_list)
        read_ids = [item["metadata"]['read_id'] for item in batch]
        
        padded_signals = pad_sequence(signals, batch_first=True, padding_value=-1)
        lengths_tensor = lengths_tensor.clip(1000)
        
        return {
            "inputs": {
                "signal": padded_signals,
                "length": lengths_tensor,
            },
            "metadata": {
                "read_id": read_ids,
                "num_tokens": lengths_numpy,
            }
        }
    
    def _prepare_dataset(self, pod5_paths: Union[str, List[str]], read_ids: List[str], batch_size: int) -> Pod5IterDataset:
        """Prepare dataset for inference."""
        dtype = "float64" if self.config.float_dtype == "float64" else "float32"
        
        dataset_params = {
            "batch_size": batch_size,
            "window_size": self.config.chunk_size,
            "step_size": self.config.chunk_size,
            "max_seq_len": self.config.max_seq_len,
            "transform": self._local_standardizer,
            "dtype": dtype,
        }
        
        dataset = Pod5IterDataset(
            read_ids=read_ids,
            pod5_paths=pod5_paths,
            random_crop=False,
            load_labels=False,
            **dataset_params,
        )
        
        return dataset
    
    def predict(
        self,
        pod5_paths: Union[str, List[str]],
        read_ids: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        save_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> InferenceResults:
        """
        Run inference on Pod5 files.
        
        Args:
            pod5_paths: Path(s) to Pod5 files or directories
            read_ids: List of read IDs to process
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
            save_path: Path to save results (optional)
            show_progress: Whether to show progress bar
            
        Returns:
            InferenceResults object containing all results and metadata
        """
        start_time = time.time()
        
        # Create metadata
        metadata = InferenceMetadata(
            # Model configuration
            chunk_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            model_type=self.config.model_type if hasattr(self.config, 'model_type') else 'transformer_zam',
            num_classes=self.config.num_classes,
            num_classes_seq2seq=getattr(self.config, 'num_classes_seq2seq', 4),
            
            # Inference settings
            batch_size=batch_size,
            device=str(self.device),
            float_dtype=self.config.float_dtype,
            
            # Run information
            model_checkpoint=str(self.config.checkpoint_path) if self.config.checkpoint_path else None,
            pod5_paths=[str(p) for p in (pod5_paths if isinstance(pod5_paths, list) else [pod5_paths])],
        )
        
        # Create results container
        results = InferenceResults(metadata=metadata)
        
        # Prepare dataset
        dataset = self._prepare_dataset(pod5_paths, read_ids, batch_size)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            shuffle=False,
        )
        
        # Run inference
        print(f"Running inference on {len(read_ids)} reads...")
        use_amp = self.config.float_dtype == "float16"
        
        with torch.no_grad():
            iterator = tqdm.tqdm(dataloader, desc="Processing") if show_progress else dataloader
            
            for batch in iterator:
                # Move batch to device
                inputs = {
                    key: tensor.to(self.device) 
                    for key, tensor in batch["inputs"].items()
                }
                
                # Run model
                with torch.amp.autocast(device_type=self.device, enabled=use_amp):
                    outputs = self.model(**inputs)
                
                # Process each read in the batch
                for i, read_id in enumerate(batch["metadata"]["read_id"]):
                    # Extract logits for this read
                    logits = {}
                    for key, tensor in outputs.items():
                        logits[key] = tensor[i].cpu().numpy()
                    
                    # Calculate number of chunks
                    num_tokens = batch["metadata"]["num_tokens"][i]
                    num_chunks = int(np.ceil(num_tokens / self.config.chunk_size))
                    
                    # Add to results
                    results.add(
                        read_id=read_id,
                        logits=logits,
                        num_chunks=num_chunks
                    )
        
        # Update timing
        results.metadata.total_inference_time = time.time() - start_time
        
        # Save if requested
        if save_path:
            results.save(save_path)
            print(f"Results saved to {save_path}")
        
        print(f"Done! Processed {len(results)} reads in {results.metadata.total_inference_time:.2f}s")
        return results
    
    def predict_from_file(
        self,
        pod5_paths: Union[str, List[str]],
        read_ids_file: str,
        **kwargs
    ) -> InferenceResults:
        """
        Run inference using read IDs from a file.
        
        Args:
            pod5_paths: Path(s) to Pod5 files or directories
            read_ids_file: Path to file containing read IDs (one per line)
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            InferenceResults object
        """
        with open(read_ids_file, 'r') as f:
            read_ids = [line.strip() for line in f.readlines()]
            
        return self.predict(pod5_paths, read_ids, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "config": self.config.__dict__,
            "num_parameters": self.model_loader.get_num_parameters(),
            "device": str(self.device),
            "dtype": self.config.float_dtype,
        }