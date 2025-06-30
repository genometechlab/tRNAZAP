"""
Visualization function for tRNA-zap inference results.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pod5
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from uuid import UUID

try:
    from torchcrf import CRF
except ImportError:
    CRF = None

from ..feeders import SequenceScaler
from ..storages import InferenceResults


# Color scheme for different classes
COLOR_MAP = {
    0: "indianred",      # tRNA
    1: "yellowgreen",    # Adapter 1
    2: "lightskyblue",   # Adapter 2
    3: "orange",         # Adapter 3
}

CLASS_LABELS = {
    0: "tRNA",
    1: "ONT Adapter",
    2: "3' Splint", 
    3: "5' Splint",
}


def visualize_from_results(
    read_id: str,
    results: InferenceResults,
    pod5_paths: Union[str, Path, List[Union[str, Path]]],
    apply_crf_smoothing: bool = True,
    plot_probabilities: bool = True,
    plot_signal: bool = True,
    figure_size: Tuple[int, int] = (16, 8),
    signal_scale: float = 1000.0,
    device: Optional[torch.device] = None
) -> plt.Figure:
    """Visualize a single read from inference results."""
    
    if read_id not in results:
        raise ValueError(f"Read ID {read_id} not found in results")
    
    # Get read result
    read_result = results[read_id]
    
    # Get chunk size from results metadata
    chunk_size = results.chunk_size
    
    # Get seq2seq logits
    logits = read_result.seq2seq_logits
    if logits is None:
        raise ValueError(f"No seq2seq logits found for read {read_id}")
    
    # Load signal from Pod5
    try:
        signal = _load_signal_from_pod5(pod5_paths, read_id)
    except Exception as e:
        print(f"[ERROR] Failed to load signal: {e}")
        raise
    
    # Prepare signal for visualization
    signal_scaled = _prepare_signal(signal, signal_scale)
    
    # Calculate probabilities and predictions
    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = probabilities.argmax(axis=-1)
    
    # Apply CRF smoothing if requested
    predictions_smooth = None
    if apply_crf_smoothing and CRF is not None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            predictions_smooth = _apply_crf_smoothing(logits, device)
        except Exception as e:
            print(f"[WARNING] CRF smoothing failed: {e}")
            predictions_smooth = None
    
    # Create and return figure
    try:
        fig = _create_figure(
            read_id=read_id,
            signal=signal_scaled,
            predictions=predictions,
            predictions_smooth=predictions_smooth,
            probabilities=probabilities,
            chunk_size=chunk_size,
            plot_probabilities=plot_probabilities,
            plot_signal=plot_signal,
            figure_size=figure_size,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create figure: {e}")
        raise
    
    return fig

# Helper functions (private)

def _load_signal_from_pod5(pod5_paths: Union[str, Path, List[Union[str, Path]]], read_id: str) -> np.ndarray:
    """Load signal from Pod5 file."""
    with pod5.DatasetReader(pod5_paths, recursive=True) as pod5_reader:
        read_record = next(pod5_reader.reads(selection=[UUID(read_id)]))
        signal = read_record.signal
    
    if signal is None:
        raise ValueError(f"Could not load signal for read {read_id} from Pod5 files")
    
    return signal


def _prepare_signal(signal: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """Prepare signal for visualization."""
    signal = signal.astype(np.float32)
    
    # Apply scaling for visualization
    scaler = SequenceScaler(scale=scale, offset=0)
    signal = signal.reshape(-1, 1)
    signal = scaler.fit_transform([signal])[0].reshape(-1)
    
    return signal


def _apply_crf_smoothing(logits: np.ndarray, device: torch.device) -> np.ndarray:
    """Apply CRF smoothing to predictions."""
    crf = CRF(4, batch_first=True)
    
    # Set transition constraints
    crf.transitions = torch.nn.Parameter(
        torch.tensor([
            [0, -1e8, -1e8, 0],      # 0 -> 0,3
            [-1e8, 0, 0, -1e8],      # 1 -> 1,2
            [0, -1e8, 0, -1e8],      # 2 -> 2,0
            [-1e8, -1e8, -1e8, 0],   # 3 -> 3
        ])
    )
    crf.to(device)
    
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        logits_tensor = torch.tensor(logits).unsqueeze(0).to(device)
        mask = torch.ones(logits_tensor.shape[:2], dtype=torch.bool).to(device)
        
        # Decode
        smoothed = crf.decode(logits_tensor, mask)
    
    return np.array(smoothed[0])


def _create_figure(
    read_id: str,
    signal: np.ndarray,
    predictions: np.ndarray,
    predictions_smooth: Optional[np.ndarray],
    probabilities: np.ndarray,
    chunk_size: int,
    plot_probabilities: bool,
    plot_signal: bool,
    figure_size: Tuple[int, int],
) -> plt.Figure:
    """Create the visualization figure."""
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Y-axis positions for different elements
    pred_y = 1100
    pred_smooth_y = 1250
    prob_y = 1400
    
    # Plot signal if requested
    if plot_signal:
        ax.plot(signal, label="Signal", color='black', alpha=1, linewidth=0.5)
    
    # Plot prediction bars
    _create_prediction_bar(ax, predictions, chunk_size, pred_y, width=100)
    ax.text(0, pred_y + 50, "Predictions", fontsize=12, fontweight='bold')
    
    # Plot smoothed predictions if available
    if predictions_smooth is not None:
        _create_prediction_bar(ax, predictions_smooth, chunk_size, pred_smooth_y, width=100)
        ax.text(0, pred_smooth_y + 50, "Predictions (CRF Smoothed)", fontsize=12, fontweight='bold')
    
    # Plot probabilities if requested
    if plot_probabilities:
        _plot_probabilities(ax, probabilities, chunk_size, prob_y)
    
    # Styling
    ax.set_title(f"Read: {read_id}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (samples)", fontsize=14)
    ax.set_xlim(0, len(signal))
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    
    # Add legend for classes
    _add_class_legend(ax)
    
    plt.tight_layout()
    return fig


def _create_prediction_bar(
    ax: plt.Axes,
    predictions: np.ndarray,
    chunk_size: int,
    y_position: float,
    width: float = 100
):
    """Create colored bar for predictions."""
    x_position = 0
    
    for pred in predictions:
        color = COLOR_MAP.get(pred, 'gray')
        rect = Rectangle(
            (x_position, y_position),
            chunk_size,
            width,
            edgecolor=color,
            facecolor=color,
            alpha=1
        )
        ax.add_patch(rect)
        x_position += chunk_size


def _plot_probabilities(
    ax: plt.Axes,
    probabilities: np.ndarray,
    chunk_size: int,
    y_base: float,
    y_scale: float = 200
):
    """Plot probability distributions for each class."""
    x_positions = np.arange(len(probabilities)) * chunk_size + chunk_size // 2
    
    # Plot each class probability
    for class_idx, color in COLOR_MAP.items():
        y_values = probabilities[:, class_idx] * y_scale + y_base
        ax.plot(
            x_positions,
            y_values,
            color=color,
            alpha=0.7,
            linewidth=2,
            label=CLASS_LABELS[class_idx]
        )
    
    # Add reference lines
    ax.hlines(y_base, 0, x_positions[-1], color='black', linestyle='--', alpha=0.3)
    ax.hlines(y_base + y_scale * 0.5, 0, x_positions[-1], color='black', linestyle='--', alpha=0.3)
    ax.hlines(y_base + y_scale, 0, x_positions[-1], color='black', linestyle='--', alpha=0.3)
    
    # Add probability labels
    ax.text(-len(x_positions) * chunk_size * 0.02, y_base, "0.0", fontsize=10)
    ax.text(-len(x_positions) * chunk_size * 0.02, y_base + y_scale * 0.5, "0.5", fontsize=10)
    ax.text(-len(x_positions) * chunk_size * 0.02, y_base + y_scale, "1.0", fontsize=10)
    ax.text(0, y_base + y_scale + 20, "Class Probabilities", fontsize=12, fontweight='bold')


def _add_class_legend(ax: plt.Axes):
    """Add legend for class colors horizontally below the plot."""
    handles = []
    for class_idx, color in COLOR_MAP.items():
        handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=1, label=CLASS_LABELS[class_idx])
        )
    signal_line = plt.Line2D([0], [0], color='black', linewidth=2, alpha=1, label='Signal')
    handles.append(signal_line)
    
    # Place legend horizontally below the plot
    ax.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.075),  # Position below the x-axis
        ncol=len(handles),  # Number of columns = number of classes (horizontal)
        frameon=False,
        fontsize=10,
        columnspacing=2.0,  # Space between legend items
    )