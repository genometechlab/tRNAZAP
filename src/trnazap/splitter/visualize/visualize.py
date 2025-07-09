import numpy as np
import torch
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pod5
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List, Generator
from uuid import UUID

from ..feeders import SequenceScaler
from ..storages import InferenceResults

logging.basicConfig(level=logging.INFO)

COLOR_MAP = {
    0: "indianred",
    1: "yellowgreen",
    2: "lightskyblue",
    3: "orange",
}

CLASS_LABELS = {
    0: "tRNA",
    1: "ONT Adapter",
    2: "3' Splint",
    3: "5' Splint",
}


class ResultsVisualizer:
    """
    Visualizes inference results from a tRNA-zap model, including signal, predictions, 
    CRF-smoothed outputs, and class probabilities.

    Parameters:
    -----------
    results : InferenceResults
        The results object containing logits, predictions, etc.

    signal_scale : float, default=1000.0
        Multiplier used to rescale the signal for visualization.

    device : torch.device, optional
        Device to run CRF smoothing on. Defaults to CUDA if available.

    color_map : dict, optional
        Mapping from class index to matplotlib color string.

    class_labels : dict, optional
        Mapping from class index to human-readable label.

    pod5_paths : str or list, optional
        Path(s) to POD5 files. If not provided, inferred from results.metadata.
    """
    def __init__(
        self,
        results: InferenceResults,
        signal_scale: float = 1000.0,
        device: Optional[torch.device] = None,
        color_map: Dict[int, str] = COLOR_MAP,
        class_labels: Dict[int, str] = CLASS_LABELS,
        pod5_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None
    ) -> None:
        self.results = results
        self.signal_scale = signal_scale
        self.color_map = color_map
        self.class_labels = class_labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pod5_paths_handler(pod5_paths)

    def _pod5_paths_handler(self, pod5_paths):
        if pod5_paths is not None:
            pod5_paths = [Path(p) for p in (pod5_paths if isinstance(pod5_paths, (list, tuple)) else [pod5_paths])]
            if not all(p.exists() for p in pod5_paths):
                missing = [str(p) for p in pod5_paths if not p.exists()]
                raise FileNotFoundError(f"The following pod5 paths do not exist: {missing}")
            self._pod5_paths = pod5_paths

        else:
            try:
                meta_paths = self.results.metadata.pod5_paths
            except AttributeError:
                raise ValueError(
                    "No pod5_paths were provided, and the results metadata does not contain any. "
                    "Please specify pod5_paths manually."
                )

            meta_paths = [Path(p) for p in (meta_paths if isinstance(meta_paths, (list, tuple)) else [meta_paths])]
            if all(p.exists() for p in meta_paths):
                self._pod5_paths = meta_paths
            else:
                missing = [str(p) for p in meta_paths if not p.exists()]
                raise FileNotFoundError(
                    f"No pod5_paths were provided, and the paths stored in results metadata no longer exist:\n"
                    f"{missing}\n"
                    "Please provide updated pod5_paths manually (e.g., if the files were moved or renamed)."
                )

    def _visualize(
        self,
        read_id: str,
        signal: np.ndarray,
        apply_crf_smoothing: bool = True,
        plot_probabilities: bool = True,
        plot_signal: bool = True,
        figure_size: Tuple[int, int] = (16, 8),
    ) -> plt.Figure:
        
        read_result = self.results[read_id]
        logits = read_result.seq2seq_logits
        probabilities = read_result.seq2seq_probs
        predictions = read_result.seq2seq_preds
        chunk_size = self.results.metadata.chunk_size

        if logits is None:
            raise ValueError(f"No seq2seq logits found for read {read_id}")

        signal_scaled = self._prepare_signal(signal)

        predictions_smooth = None
        if apply_crf_smoothing:
            try:
                from ..utils import crf_smoothing
                predictions_smooth = crf_smoothing(logits, device=self.device)
            except ImportError as e:
                print(f"[WARNING] Failed to import crf_smoothing: {e}")
            except Exception as e:
                print(f"[WARNING] CRF smoothing failed: {e}")

        fig = self._create_figure(
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

        return fig
    
    def visualize(        
        self,
        read_ids: Union[List[str], str],
        apply_crf_smoothing: bool = True,
        plot_probabilities: bool = True,
        plot_signal: bool = True,
        figure_size: Tuple[int, int] = (16, 8),
    ) -> Union[plt.Figure, List[plt.Figure]]:
        
        return_single_fig = False
        if isinstance(read_ids, str):
            return_single_fig = True
            read_ids = [read_ids,]

        if not all(read_id in self.results for read_id in read_ids):
            missing = [read_id for read_id in read_ids if read_id not in self.results]
            raise ValueError(f"Read ID(s) {', '.join(missing)} not found in results")
        
        figs = []
        signals = self._load_signals(read_ids)
        for read_id in read_ids:
            fig_ = self._visualize(read_id,
                                   signals[read_id],
                                   apply_crf_smoothing,
                                   plot_probabilities,
                                   plot_signal,
                                   figure_size)
            figs.append(fig_)
            
        return figs[0] if return_single_fig else figs 
        
    def _load_signals(self, read_ids: List[str]) -> Dict:
        if not hasattr(self, "_reader") or self._reader is None:
            self._reader = pod5.DatasetReader(self._pod5_paths, recursive=True, max_cached_readers=8)

        try:
            out_dict = {}
            selection = [UUID(read_id) for read_id in read_ids]
            for read_record in self._reader.reads(selection=selection):
                fetched_signal = read_record.signal
                fetched_read_id = str(read_record.read_id)
                if fetched_signal is None:
                    raise ValueError(f"Signal for read ID {fetched_read_id} is empty or missing.")
                out_dict[fetched_read_id] = fetched_signal
            return out_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load signal for the provided read_ids")

    def _prepare_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = signal.astype(np.float32).reshape(-1, 1)
        scaler = SequenceScaler(scale=self.signal_scale, offset=0)
        return scaler.fit_transform([signal])[0].reshape(-1)

    def _create_figure(
        self,
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
        fig, ax = plt.subplots(figsize=figure_size)
        pred_y = 1100
        pred_smooth_y = 1250
        prob_y = 1400

        if plot_signal:
            ax.plot(signal, label="Signal", color='black', alpha=1, linewidth=0.5)

        self._create_prediction_bar(ax, predictions, chunk_size, pred_y)
        ax.text(0, pred_y + 50, "Predictions", fontsize=12, fontweight='bold')

        if predictions_smooth is not None:
            self._create_prediction_bar(ax, predictions_smooth, chunk_size, pred_smooth_y)
            ax.text(0, pred_smooth_y + 50, "Predictions (CRF Smoothed)", fontsize=12, fontweight='bold')

        if plot_probabilities:
            self._plot_probabilities(ax, probabilities, chunk_size, prob_y)

        ax.set_title(f"Read: {read_id}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (samples)", fontsize=14)
        ax.set_xlim(0, len(signal))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

        self._add_class_legend(ax)
        plt.tight_layout()
        plt.close()
        return fig

    def _create_prediction_bar(self, ax, predictions, chunk_size, y_position, width=100):
        x = 0
        for pred in predictions:
            color = self.color_map.get(pred, 'gray')
            rect = Rectangle((x, y_position), chunk_size, width, edgecolor=color, facecolor=color)
            ax.add_patch(rect)
            x += chunk_size

    def _plot_probabilities(self, ax, probabilities, chunk_size, y_base, y_scale=200):
        x_positions = np.arange(len(probabilities)) * chunk_size + chunk_size // 2
        for class_idx, color in self.color_map.items():
            y_values = probabilities[:, class_idx] * y_scale + y_base
            ax.plot(x_positions, y_values, color=color, alpha=0.7, linewidth=2, label=self.class_labels[class_idx])
        ax.hlines([y_base, y_base + y_scale * 0.5, y_base + y_scale], 0, x_positions[-1], color='black', linestyle='--', alpha=0.3)
        ax.text(-len(x_positions) * chunk_size * 0.02, y_base, "0.0", fontsize=10)
        ax.text(-len(x_positions) * chunk_size * 0.02, y_base + y_scale * 0.5, "0.5", fontsize=10)
        ax.text(-len(x_positions) * chunk_size * 0.02, y_base + y_scale, "1.0", fontsize=10)
        ax.text(0, y_base + y_scale + 20, "Class Probabilities", fontsize=12, fontweight='bold')

    def _add_class_legend(self, ax):
        handles = [Rectangle((0, 0), 1, 1, fc=color, label=self.class_labels[idx]) for idx, color in self.color_map.items()]
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

    def close(self):
        """Close the internal POD5 reader if it is open."""
        if hasattr(self, "_reader") and self._reader is not None:
            self._reader.clear_index()
            self._reader.clear_readers()
            self._reader = None

    
    def __enter__(self) -> "ResultsVisualizer":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
