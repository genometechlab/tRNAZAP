import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import pod5
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from uuid import UUID

from ..feeders import SequenceScaler
from ..storages import ReadResult

logger = logging.getLogger(__name__)

# Constants
DEFAULT_COLOR_MAP = {
    0: "indianred",
    1: "yellowgreen",
    2: "lightskyblue",
    3: "orange",
}

DEFAULT_CLASS_LABELS = {
    0: "tRNA",
    1: "ONT Adapter",
    2: "3' Splint",
    3: "5' Splint",
}

# Visualization parameters
VIZ_PARAMS = {
    "pred_y": 1100,
    "pred_smooth_y": 1250,
    "prob_y": 1400,
    "bar_height": 100,
    "prob_scale": 200,
}


class ResultsVisualizer:
    """
    Visualizes inference results from a tRNA-zap model.
    
    Displays signal, predictions, CRF-smoothed outputs, and class probabilities.
    """
    
    def __init__(
        self,
        pod5_paths: Union[str, Path, List[Union[str, Path]]],
        signal_scale: float = 1000.0,
        device: Optional[torch.device] = None,
        color_map: Optional[Dict[int, str]] = None,
        class_labels: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Initialize the visualizer.
        
        Args:
            pod5_paths: Path(s) to POD5 files
            signal_scale: Multiplier for signal visualization
            device: Device for CRF smoothing (defaults to CUDA if available)
            color_map: Class index to color mapping
            class_labels: Class index to label mapping
        """
        self.signal_scale = signal_scale
        self.color_map = color_map or DEFAULT_COLOR_MAP
        self.class_labels = class_labels or DEFAULT_CLASS_LABELS
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pod5_paths = self._validate_pod5_paths(pod5_paths)
        self._reader = None

    def _validate_pod5_paths(self, pod5_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        """Validate and convert POD5 paths to Path objects."""
        if isinstance(pod5_paths, (str, Path)):
            pod5_paths = [pod5_paths]
        
        paths = [Path(p) for p in pod5_paths]
        missing = [str(p) for p in paths if not p.exists()]
        
        if missing:
            raise FileNotFoundError(f"POD5 paths not found: {missing}")
        
        return paths

    def visualize(
        self,
        read_results: Union[List[ReadResult], ReadResult],
        apply_crf_smoothing: bool = True,
        plot_probabilities: bool = True,
        plot_signal: bool = True,
        figure_size: Tuple[int, int] = (16, 8),
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """
        Create visualization(s) for read results.
        
        Args:
            read_results: Single result or list of results to visualize
            apply_crf_smoothing: Whether to apply CRF smoothing
            plot_probabilities: Whether to plot class probabilities
            plot_signal: Whether to plot the raw signal
            figure_size: Figure dimensions (width, height)
            
        Returns:
            Single figure or list of figures
        """
        # Handle single vs multiple inputs
        single_input = isinstance(read_results, ReadResult)
        if single_input:
            read_results = [read_results]
        
        # Load all signals at once
        read_ids = [r.read_id for r in read_results]
        signals = self._load_signals(read_ids)
        
        # Create visualizations
        figures = []
        for read_result in read_results:
            signal = signals[read_result.read_id]
            fig = self._create_visualization(
                read_result=read_result,
                signal=signal,
                apply_crf_smoothing=apply_crf_smoothing,
                plot_probabilities=plot_probabilities,
                plot_signal=plot_signal,
                figure_size=figure_size,
            )
            figures.append(fig)
        
        return figures[0] if single_input else figures

    def _create_visualization(
        self,
        read_result: ReadResult,
        signal: np.ndarray,
        apply_crf_smoothing: bool,
        plot_probabilities: bool,
        plot_signal: bool,
        figure_size: Tuple[int, int],
    ) -> plt.Figure:
        """Create a single visualization."""
        if read_result.segmentation_logits is None:
            raise ValueError(f"No segmentation logits for read {read_result.read_id}")
        
        # Prepare data
        signal_scaled = self._prepare_signal(signal)
        predictions_smooth = self._apply_crf_smoothing(read_result.segmentation_logits) if apply_crf_smoothing else None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Plot components
        if plot_signal:
            self._plot_signal(ax, signal_scaled)
        
        self._plot_predictions(ax, read_result.segmentation_preds, read_result.chunk_size, "Predictions", VIZ_PARAMS["pred_y"])
        
        if predictions_smooth is not None:
            self._plot_predictions(ax, predictions_smooth, read_result.chunk_size, "Predictions (CRF Smoothed)", VIZ_PARAMS["pred_smooth_y"])
        
        if plot_probabilities and read_result.segmentation_probs is not None:
            self._plot_probabilities(ax, read_result.segmentation_probs, read_result.chunk_size)
        
        # Format figure
        self._format_figure(ax, read_result.read_id, len(signal_scaled))
        
        plt.tight_layout()
        plt.close()
        return fig

    def _load_signals(self, read_ids: List[str]) -> Dict[str, np.ndarray]:
        """Load signals for multiple reads efficiently."""
        if self._reader is None:
            self._reader = pod5.DatasetReader(
                self._pod5_paths, 
                recursive=True, 
                max_cached_readers=8
            )
        
        signals = {}
        selection = [UUID(read_id) for read_id in read_ids]
        
        for read_record in self._reader.reads(selection=selection):
            if read_record.signal is None:
                raise ValueError(f"Empty signal for read {read_record.read_id}")
            signals[str(read_record.read_id)] = read_record.signal
        
        missing = set(read_ids) - set(signals.keys())
        if missing:
            raise RuntimeError(f"Failed to load signals for reads: {missing}")
        
        return signals

    def _prepare_signal(self, signal: np.ndarray) -> np.ndarray:
        """Scale and prepare signal for visualization."""
        signal = signal.astype(np.float32).reshape(-1, 1)
        scaler = SequenceScaler(scale=self.signal_scale, offset=0)
        return scaler.fit_transform([signal])[0].reshape(-1)

    def _apply_crf_smoothing(self, logits: np.ndarray) -> Optional[np.ndarray]:
        """Apply CRF smoothing with error handling."""
        try:
            from ..utils import crf_smoothing
            return crf_smoothing(logits, device=self.device)
        except (ImportError, Exception) as e:
            logger.warning(f"CRF smoothing failed: {e}")
            return None

    def _plot_signal(self, ax: plt.Axes, signal: np.ndarray) -> None:
        """Plot the raw signal."""
        ax.plot(signal, label="Signal", color='black', alpha=1, linewidth=0.5)

    def _plot_predictions(
        self, 
        ax: plt.Axes, 
        predictions: np.ndarray, 
        chunk_size: int, 
        label: str, 
        y_position: float
    ) -> None:
        """Plot prediction bars."""
        for i, pred in enumerate(predictions):
            x = i * chunk_size
            color = self.color_map.get(pred, 'gray')
            rect = Rectangle(
                (x, y_position), 
                chunk_size, 
                VIZ_PARAMS["bar_height"], 
                edgecolor=color, 
                facecolor=color
            )
            ax.add_patch(rect)
        
        ax.text(0, y_position + 50, label, fontsize=12, fontweight='bold')

    def _plot_probabilities(
        self, 
        ax: plt.Axes, 
        probabilities: np.ndarray, 
        chunk_size: int
    ) -> None:
        """Plot class probability curves."""
        y_base = VIZ_PARAMS["prob_y"]
        y_scale = VIZ_PARAMS["prob_scale"]
        
        x_positions = np.arange(len(probabilities)) * chunk_size + chunk_size // 2
        
        # Plot probability curves
        for class_idx, color in self.color_map.items():
            y_values = probabilities[:, class_idx] * y_scale + y_base
            ax.plot(
                x_positions, 
                y_values, 
                color=color, 
                alpha=0.7, 
                linewidth=2, 
                label=self.class_labels[class_idx]
            )
        
        # Add grid lines and labels
        self._add_probability_grid(ax, x_positions[-1], y_base, y_scale)
        ax.text(0, y_base + y_scale + 20, "Class Probabilities", fontsize=12, fontweight='bold')

    def _add_probability_grid(
        self, 
        ax: plt.Axes, 
        x_max: float, 
        y_base: float, 
        y_scale: float
    ) -> None:
        """Add grid lines and labels for probability plot."""
        y_levels = [y_base, y_base + y_scale * 0.5, y_base + y_scale]
        labels = ["0.0", "0.5", "1.0"]
        
        ax.hlines(y_levels, 0, x_max, color='black', linestyle='--', alpha=0.3)
        
        for y, label in zip(y_levels, labels):
            ax.text(-x_max * 0.02, y, label, fontsize=10)

    def _format_figure(self, ax: plt.Axes, read_id: str, signal_length: int) -> None:
        """Apply formatting to the figure."""
        ax.set_title(f"Read: {read_id}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (samples)", fontsize=14)
        ax.set_xlim(0, signal_length)
        ax.set_yticks([])
        
        # Remove spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        
        # Add legend
        self._add_legend(ax)

    def _add_legend(self, ax: plt.Axes) -> None:
        """Add legend with class colors and signal."""
        handles = [
            Rectangle((0, 0), 1, 1, fc=color, label=self.class_labels[idx])
            for idx, color in self.color_map.items()
        ]
        handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label='Signal'))
        
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.075),
            ncol=len(handles),
            frameon=False,
            fontsize=10,
            columnspacing=2.0,
        )

    def close(self) -> None:
        """Close the POD5 reader and release resources."""
        if self._reader is not None:
            self._reader.clear_index()
            self._reader.clear_readers()
            self._reader = None

    def __enter__(self) -> "ResultsVisualizer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()