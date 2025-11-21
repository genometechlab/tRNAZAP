import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import pod5
import warnings
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from uuid import UUID

from ..feeders import SequenceScaler
from ..storages import ReadResult, ReadResultCompressed


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
    "gt": 1550,
    "bar_height": 100,
    "prob_scale": 200,
}

logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """
    Visualizes inference results from a tRNA-zap model
    with publication-ready styling and Illustrator-friendly output.
    """

    def __init__(
        self,
        pod5_paths: Union[str, Path, List[Union[str, Path]]],
        signal_scale: float = 1000.0,
        device: Optional[torch.device] = None,
        color_map: Optional[Dict[int, str]] = None,
        class_labels: Optional[Dict[int, str]] = None,
        rasterize_signal: bool = False,   # NEW: reduce PDF size while keeping annotations vector
        signal_linewidth: float = 0.6,    # NEW: consistent stroke widths
    ) -> None:
        self.signal_scale = signal_scale
        self.color_map = color_map or DEFAULT_COLOR_MAP
        self.class_labels = class_labels or DEFAULT_CLASS_LABELS
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pod5_paths = self._validate_pod5_paths(pod5_paths)
        self._reader = None

        # Styling knobs
        self.rasterize_signal = rasterize_signal
        self.signal_linewidth = signal_linewidth

    def _validate_pod5_paths(self, pod5_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
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
        ground_truth_segmentations: List[int] = None,
        figure_size: Tuple[int, int] = (16, 8),
        title_prefix: str = "Read",
    ) -> Union[plt.Figure, List[plt.Figure]]:
        single_input = isinstance(read_results, ReadResult) or isinstance(read_results, ReadResultCompressed)
        if single_input:
            read_results = [read_results]

        # Load all signals at once
        read_ids = [r.read_id for r in read_results]
        signals = self._load_signals(read_ids)

        figures = []
        for read_result in read_results:
            signal = signals[read_result.read_id]
            fig = self._create_visualization(
                read_result=read_result,
                signal=signal,
                apply_crf_smoothing=apply_crf_smoothing,
                plot_probabilities=plot_probabilities,
                plot_signal=plot_signal,
                ground_truth_segmentations=ground_truth_segmentations,
                figure_size=figure_size,
                title_prefix=title_prefix,
            )
            figures.append(fig)
        return figures[0] if single_input else figures

    def _create_visualization(
        self,
        read_result: Union[ReadResult,ReadResultCompressed],
        signal: np.ndarray,
        apply_crf_smoothing: bool,
        plot_probabilities: bool,
        plot_signal: bool,
        ground_truth_segmentations: List[int],
        figure_size: Tuple[int, int],
        title_prefix: str,
    ) -> plt.Figure:
        # Prepare data
        signal_scaled = self._prepare_signal(signal)

        # Figure
        fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)

        # Signal
        if plot_signal:
            self._plot_signal(ax, signal_scaled)
            
        if isinstance(read_result, ReadResult):
            plot_segmentation = True
            if read_result.segmentation_logits is None:
                warnings.warn(f"No segmentation logits for read {read_result.read_id}")
                plot_segmentation = False

            if plot_segmentation:
                # Predictions
                self._plot_segmentations(ax, read_result.segmentation_preds, read_result.chunk_size,
                                    "Pred.", VIZ_PARAMS["pred_y"])
                if apply_crf_smoothing:
                    predictions_smooth = read_result.get_smoothed_segmentation_preds(device=self.device)
                    self._plot_segmentations(ax, predictions_smooth, read_result.chunk_size,
                                        "Pred. (CRF)", VIZ_PARAMS["pred_smooth_y"])

            # Probabilities
            if plot_probabilities and read_result.segmentation_probs is not None:
                self._plot_probabilities(ax, read_result.segmentation_probs, read_result.chunk_size)
                    
            if ground_truth_segmentations is not None:
                self._plot_segmentations(ax, ground_truth_segmentations, read_result.chunk_size,
                        "GT", VIZ_PARAMS["gt"])
        
        elif isinstance(read_result, ReadResultCompressed):
            if apply_crf_smoothing:
                warnings.warn("Cannot apply the CRF smoothing on ReadResultcompressed."+
                            " Requires raw ReadResult,"+
                            " Pass save_raw=True to inference to get raw ReadResults")
        
            variable_region_range = read_result.variable_region_range
            start_ = variable_region_range[0]//read_result.chunk_size
            end_ = ((variable_region_range[1]+1)//read_result.chunk_size)-1
            preds_array = np.zeros((read_result.num_chunks))-1
            preds_array[start_:end_+1]=0
            self._plot_segmentations(ax, preds_array, read_result.chunk_size,
                                    "Pred.", VIZ_PARAMS["pred_y"])
            
            
        # Format
        self._format_figure(ax, read_result.read_id, len(signal_scaled), title_prefix=title_prefix)

        plt.close()
        return fig
            

    def _load_signals(self, read_ids: List[str]) -> Dict[str, np.ndarray]:
        if self._reader is None:
            self._reader = pod5.DatasetReader(self._pod5_paths, recursive=True, max_cached_readers=8)

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
        signal = signal.astype(np.float32).reshape(-1, 1)
        scaler = SequenceScaler(scale=self.signal_scale, offset=0)
        return scaler.fit_transform([signal])[0].reshape(-1)

    def _apply_crf_smoothing(self, logits: np.ndarray) -> Optional[np.ndarray]:
        try:
            from ..utils import crf_smoothing
            return crf_smoothing(logits, device=self.device)
        except (ImportError, Exception) as e:
            logger.warning(f"CRF smoothing failed: {e}")
            return None

    def _plot_signal(self, ax: plt.Axes, signal: np.ndarray) -> None:
        # Optionally rasterize the heavy line to keep PDF file size light
        ax.plot(
            signal,
            label="Signal",
            color="black",
            alpha=1,
            linewidth=self.signal_linewidth,
            zorder=1,
            rasterized=self.rasterize_signal,
        )

    def _plot_segmentations(
        self,
        ax: plt.Axes,
        predictions: np.ndarray,
        chunk_size: int,
        label: str,
        y_position: float
    ) -> None:
        # Draw colored blocks
        for i, pred in enumerate(predictions):
            if pred==-1: continue
            x = i * chunk_size
            color = self.color_map.get(pred, '0.6')
            rect = Rectangle(
                (x, y_position),
                chunk_size,
                VIZ_PARAMS["bar_height"],
                edgecolor=color,
                facecolor=color,
                linewidth=0,
                zorder=3,
            )
            ax.add_patch(rect)

        # Inline section label with a subtle white box for clarity
        ax.text(
            0, y_position + VIZ_PARAMS["bar_height"]//3, label,
            fontsize=10, fontweight='bold', va="bottom",
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none", pad=2),
            zorder=4
        )

    def _plot_probabilities(
        self,
        ax: plt.Axes,
        probabilities: np.ndarray,
        chunk_size: int
    ) -> None:
        y_base = VIZ_PARAMS["prob_y"]
        y_scale = VIZ_PARAMS["prob_scale"]
        x_positions = np.arange(len(probabilities)) * chunk_size + chunk_size // 2

        for class_idx, color in self.color_map.items():
            y_values = probabilities[:, class_idx] * y_scale + y_base
            ax.plot(
                x_positions,
                y_values,
                color=color,
                alpha=0.9,
                linewidth=1.6,
                label=self.class_labels[class_idx],
                zorder=5,
            )

        # Clean probability grid with proper transforms
        self._add_probability_grid(ax, x_positions[-1], y_base, y_scale)

        ax.text(
            0, y_base + y_scale + 24, "Class probabilities",
            fontsize=10, fontweight='bold', va="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            zorder=6
        )

    def _add_probability_grid(
        self,
        ax: plt.Axes,
        x_max: float,
        y_base: float,
        y_scale: float
    ) -> None:
        # Horizontal guides at 0, 0.5, 1.0
        levels = [0.0, 0.5, 1.0]
        y_levels = [y_base + y_scale * lv for lv in levels]
        ax.hlines(y_levels, 0, x_max, color='0.2', linestyle='--', linewidth=0.6, alpha=0.25, zorder=2)

        # Left-anchored labels (no negative x)
        trans = blended_transform_factory(ax.transAxes, ax.transData)  # x in axes, y in data
        for lv, y in zip(levels, y_levels):
            ax.text(
                0.002, y, f"{lv:.1f}",
                transform=trans, ha="left", va="center",
                fontsize=9, color='0.3',
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5),
                zorder=6
            )

    def _format_figure(self, ax: plt.Axes, read_id: str, signal_length: int, title_prefix: str) -> None:
        # Title & axes
        ax.set_title(f"{title_prefix}: {read_id}", fontweight='bold', pad=8)
        ax.set_xlabel("Time (samples)")
        ax.set_xlim(0, signal_length)

        # No y-ticks (composite visual scale)
        ax.set_yticks([])

        # Clean spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)

        # Ticks: outward, subtle
        ax.tick_params(axis='x', direction='out', length=4, width=0.8)

        # Compact margins
        ax.margins(x=0)

        # Legend (single row, outside below)
        self._add_legend(ax)

    def _add_legend(self, ax: plt.Axes) -> None:
        handles = [
            Rectangle((0, 0), 1, 1, fc=color, label=self.class_labels[idx], clip_on=False)
            for idx, color in self.color_map.items()
        ]
        handles.append(plt.Line2D([0], [0], color='black', linewidth=1.4, label='Signal'))

        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(handles),
            frameon=False,
            fontsize=9,
            handlelength=1.5,
            columnspacing=1.8,
            handletextpad=0.6,
            borderaxespad=0.8,
        )

    def close(self) -> None:
        if self._reader is not None:
            self._reader.clear_index()
            self._reader.clear_readers()
            self._reader = None

    def __enter__(self) -> "ResultsVisualizer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
