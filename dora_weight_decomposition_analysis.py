"""
Utilities for reproducing the DoRA weight decomposition analysis.

This module is designed to fit the custom `DoRALinear` implementation in
`DoRA_implementation.ipynb` and can be imported directly from the notebook.

What it covers:
1. Saving intermediate snapshots for FT / LoRA / DoRA.
2. Loading merged checkpoints for FT / LoRA / DoRA.
3. Computing the paper's Delta M / Delta D metrics.
4. Plotting the three-panel scatter figure for a target projection matrix
   such as `q_proj` or `v_proj`.

The paper's Figure 2 uses query weights and Appendix A.1 uses value weights.
This module supports both through `target_suffix`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM


def snapshot_update_steps(total_update_steps: int, num_intermediate: int = 3) -> List[int]:
    """Return the training update steps used for the paper-style trajectory plot."""
    if total_update_steps < 1:
        raise ValueError("total_update_steps must be >= 1")
    if num_intermediate < 1:
        return [total_update_steps]

    steps = []
    for i in range(1, num_intermediate + 1):
        step = max(1, round(total_update_steps * i / (num_intermediate + 1)))
        steps.append(step)
    steps.append(total_update_steps)
    return sorted(set(steps))


def save_ft_or_lora_snapshot(model: nn.Module, output_dir: str | Path) -> Path:
    """Save a Hugging Face or PEFT checkpoint directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    return output_dir


class DoRALinear(nn.Module):
    """Standalone DoRA linear wrapper matching the notebook implementation."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            if base.bias is not None
            else None
        )

        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=self.weight.device, dtype=self.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # PyTorch Linear stores weights as [out_features, in_features].
        # To match the notebook, we decompose along dim=1 (row-wise vectors).
        magnitude = torch.linalg.vector_norm(self.weight.float(), ord=2, dim=1).to(self.weight.dtype)
        self.magnitude = nn.Parameter(magnitude)

    def merged_weight(self) -> torch.Tensor:
        directional = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        norm = (
            torch.linalg.vector_norm(directional.float(), ord=2, dim=1, keepdim=True)
            .to(directional.dtype)
            .clamp_min(1e-6)
        )
        return self.magnitude[:, None] * directional / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            directional = self.weight + self.scaling * (self.lora_B @ self.lora_A)
            norm = (
                torch.linalg.vector_norm(directional.float(), ord=2, dim=1)
                .to(directional.dtype)
                .clamp_min(1e-6)
            )
            norm_scale = self.magnitude / norm.detach()

            base_out = F.linear(x, self.weight, self.bias)
            lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
            return norm_scale * (base_out + lora_out)

        return F.linear(self.dropout(x), self.merged_weight(), self.bias)


def replace_target_linears_with_dora(
    model: nn.Module,
    target_modules: Iterable[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> int:
    modules = dict(model.named_modules())
    replacements = []
    target_modules = tuple(target_modules)

    for name, module in modules.items():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if any(target == leaf or target in name for target in target_modules):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = modules[parent_name] if parent_name else model
            replacements.append((parent, child_name))

    for parent, child_name in replacements:
        old = getattr(parent, child_name)
        setattr(parent, child_name, DoRALinear(old, rank=rank, alpha=alpha, dropout=dropout))

    return len(replacements)


def load_dora_adapter(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["state_dict"]

    for name, module in model.named_modules():
        if not isinstance(module, DoRALinear):
            continue
        for attr in ("lora_A", "lora_B", "magnitude"):
            key = f"{name}.{attr}"
            tensor = state[key].to(device=getattr(module, attr).device, dtype=getattr(module, attr).dtype)
            getattr(module, attr).data.copy_(tensor)
    return model


def save_dora_snapshot(
    model: nn.Module,
    output_path: str | Path,
    rank: int,
    target_modules: Sequence[str],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {}

    for name, module in model.named_modules():
        if isinstance(module, DoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            state[f"{name}.magnitude"] = module.magnitude.detach().cpu()

    torch.save(
        {
            "rank": rank,
            "alpha": 2 * rank,
            "target_modules": list(target_modules),
            "state_dict": state,
        },
        output_path,
    )
    return output_path


@dataclass
class AnalysisRunSpec:
    name: str
    kind: str
    checkpoints: Sequence[str | Path]
    rank: Optional[int] = None
    alpha: Optional[int] = None
    target_modules: Optional[Sequence[str]] = None
    dropout: float = 0.0
    adapter_name: Optional[str] = None


def _load_base_model(base_model: str, torch_dtype: torch.dtype) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _load_ft_checkpoint(base_model: str, checkpoint_dir: str | Path, torch_dtype: torch.dtype) -> nn.Module:
    # The FT checkpoint is assumed to be a full `save_pretrained` directory.
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _load_lora_checkpoint(base_model: str, checkpoint_dir: str | Path, torch_dtype: torch.dtype) -> nn.Module:
    base = _load_base_model(base_model, torch_dtype=torch_dtype)
    peft_model = PeftModel.from_pretrained(base, str(checkpoint_dir), is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged.eval()
    return merged


def _load_dora_checkpoint(
    base_model: str,
    checkpoint_path: str | Path,
    torch_dtype: torch.dtype,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    dropout: float,
) -> nn.Module:
    model = _load_base_model(base_model, torch_dtype=torch_dtype)
    replace_target_linears_with_dora(
        model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    load_dora_adapter(model, checkpoint_path)
    model.eval()
    return model


def load_merged_model(
    base_model: str,
    spec: AnalysisRunSpec,
    checkpoint: str | Path,
    torch_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    kind = spec.kind.lower()
    if kind == "ft":
        return _load_ft_checkpoint(base_model, checkpoint, torch_dtype=torch_dtype)
    if kind == "lora":
        return _load_lora_checkpoint(base_model, checkpoint, torch_dtype=torch_dtype)
    if kind == "dora":
        if spec.rank is None:
            raise ValueError("DoRA analysis requires `rank` in AnalysisRunSpec.")
        alpha = spec.alpha if spec.alpha is not None else 2 * spec.rank
        target_modules = spec.target_modules or ("q_proj", "v_proj")
        return _load_dora_checkpoint(
            base_model=base_model,
            checkpoint_path=checkpoint,
            torch_dtype=torch_dtype,
            rank=spec.rank,
            alpha=alpha,
            target_modules=target_modules,
            dropout=spec.dropout,
        )
    raise ValueError(f"Unsupported run kind: {spec.kind}")


def extract_projection_weights(
    model: nn.Module,
    target_suffix: str,
) -> Dict[str, torch.Tensor]:
    weights: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, DoRALinear) and name.endswith(target_suffix):
            weights[name] = module.merged_weight().detach().float().cpu()
        elif isinstance(module, nn.Linear) and name.endswith(target_suffix):
            weights[name] = module.weight.detach().float().cpu()

    if not weights:
        raise ValueError(f"No modules ending with `{target_suffix}` were found.")
    return weights


def decompose_weight(weight: torch.Tensor, vector_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    magnitude = torch.linalg.vector_norm(weight, ord=2, dim=vector_dim)
    norm = magnitude.unsqueeze(vector_dim).clamp_min(1e-12)
    direction = weight / norm
    return magnitude, direction


def compute_delta_m_delta_d(
    base_weight: torch.Tensor,
    adapted_weight: torch.Tensor,
    vector_dim: int = 1,
) -> Tuple[float, float]:
    base_m, _ = decompose_weight(base_weight, vector_dim=vector_dim)
    adapted_m, _ = decompose_weight(adapted_weight, vector_dim=vector_dim)

    delta_m = (adapted_m - base_m).abs().mean().item()

    # The paper compares the directional vectors of the adapted matrix with the
    # corresponding vectors in the pre-trained weight. For our stored [out, in]
    # linear weights, `vector_dim=1` means one point per row vector.
    cos = F.cosine_similarity(adapted_weight, base_weight, dim=vector_dim, eps=1e-12)
    delta_d = (1.0 - cos).mean().item()
    return delta_m, delta_d


_LAYER_PATTERNS = (
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.encoder\.layers\.(\d+)\."),
    re.compile(r"\.decoder\.layers\.(\d+)\."),
)


def infer_layer_index(module_name: str) -> int:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(module_name)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer a layer index from module name: {module_name}")


def collect_weight_decomposition_points(
    base_model: str,
    spec: AnalysisRunSpec,
    target_suffix: str,
    torch_dtype: torch.dtype = torch.float32,
    vector_dim: int = 1,
) -> List[Dict[str, float]]:
    base = _load_base_model(base_model, torch_dtype=torch_dtype)
    base_weights = extract_projection_weights(base, target_suffix=target_suffix)

    points: List[Dict[str, float]] = []
    step_labels = [f"Inter step {i + 1}" for i in range(max(0, len(spec.checkpoints) - 1))] + ["Final step"]

    for ckpt_idx, checkpoint in enumerate(spec.checkpoints):
        merged = load_merged_model(base_model, spec, checkpoint, torch_dtype=torch_dtype)
        merged_weights = extract_projection_weights(merged, target_suffix=target_suffix)

        common_names = sorted(set(base_weights) & set(merged_weights))
        if not common_names:
            raise ValueError(
                f"No shared `{target_suffix}` modules between the base model and {spec.name} checkpoint {checkpoint}."
            )

        for name in common_names:
            layer_idx = infer_layer_index(name)
            delta_m, delta_d = compute_delta_m_delta_d(
                base_weights[name],
                merged_weights[name],
                vector_dim=vector_dim,
            )
            points.append(
                {
                    "method": spec.name,
                    "step_index": ckpt_idx,
                    "step_label": step_labels[ckpt_idx],
                    "layer": layer_idx,
                    "delta_m": delta_m,
                    "delta_d": delta_d,
                }
            )

        del merged

    del base
    return points


def plot_weight_decomposition_triptych(
    base_model: str,
    specs: Sequence[AnalysisRunSpec],
    target_suffix: str = "v_proj",
    torch_dtype: torch.dtype = torch.float32,
    vector_dim: int = 1,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (16, 5),
    show_layer_legend: bool = True,
    y_padding_ratio: float = 0.15,
) -> Tuple[plt.Figure, List[Dict[str, float]]]:
    if len(specs) not in (2, 3):
        raise ValueError("Expected two or three specs, typically LoRA / DoRA or FT / LoRA / DoRA.")

    all_points: List[Dict[str, float]] = []
    for spec in specs:
        all_points.extend(
            collect_weight_decomposition_points(
                base_model=base_model,
                spec=spec,
                target_suffix=target_suffix,
                torch_dtype=torch_dtype,
                vector_dim=vector_dim,
            )
        )

    layers = sorted({int(p["layer"]) for p in all_points})
    step_labels = sorted({str(p["step_label"]) for p in all_points}, key=_step_sort_key)
    layer_colors = plt.cm.tab20(torch.linspace(0, 1, max(len(layers), 2)).tolist())
    layer_to_color = {layer: layer_colors[i] for i, layer in enumerate(layers)}
    markers = ["o", "s", "^", "D", "P", "X"]
    step_to_marker = {step: markers[i % len(markers)] for i, step in enumerate(step_labels)}

    fig, axes = plt.subplots(1, len(specs), figsize=figsize, constrained_layout=True)
    if len(specs) == 1:
        axes = [axes]
    for ax, spec in zip(axes, specs):
        points = [p for p in all_points if p["method"] == spec.name]
        for point in points:
            ax.scatter(
                point["delta_d"],
                point["delta_m"],
                c=[layer_to_color[int(point["layer"])]],
                marker=step_to_marker[str(point["step_label"])],
                s=80,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.4,
            )

        if len(points) >= 2:
            x = torch.tensor([float(p["delta_d"]) for p in points], dtype=torch.float64)
            y = torch.tensor([float(p["delta_m"]) for p in points], dtype=torch.float64)
            x_centered = x - x.mean()
            denom = torch.sum(x_centered * x_centered)
            if denom.item() > 0:
                slope = torch.sum(x_centered * (y - y.mean())) / denom
                intercept = y.mean() - slope * x.mean()
                x_line = torch.linspace(float(x.min().item()), float(x.max().item()), steps=100)
                y_line = intercept + slope * x_line
                ax.plot(
                    x_line.tolist(),
                    y_line.tolist(),
                    color="blue",
                    linewidth=2.0,
                    alpha=0.9,
                )

        if points:
            y_values = [float(p["delta_m"]) for p in points]
            y_min = min(y_values)
            y_max = max(y_values)
            y_span = y_max - y_min
            y_pad = max(y_span * y_padding_ratio, 1e-5)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_title(spec.name)
        ax.set_xlabel("ΔD")
        ax.set_ylabel("ΔM")
        ax.grid(True, alpha=0.25)

    legend_handles = []
    if show_layer_legend:
        for layer in layers:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"layer {layer}",
                    markerfacecolor=layer_to_color[layer],
                    markeredgecolor="black",
                    markersize=8,
                )
            )
    for step in step_labels:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=step_to_marker[step],
                color="black",
                linestyle="None",
                label=step,
                markersize=8,
            )
        )

    axes[-1].legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )

    plot_title = title or f"Weight decomposition analysis on `{target_suffix}`"
    fig.suptitle(plot_title, fontsize=14)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, all_points


def plot_weight_decomposition_stacked(
    base_model: str,
    specs: Sequence[AnalysisRunSpec],
    target_suffixes: Sequence[str] = ("q_proj", "v_proj"),
    torch_dtype: torch.dtype = torch.float32,
    vector_dim: int = 1,
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (16, 10),
    y_padding_ratio: float = 0.15,
) -> Tuple[plt.Figure, Dict[str, List[Dict[str, float]]]]:
    if len(specs) not in (2, 3):
        raise ValueError("Expected two or three specs, typically LoRA / DoRA or FT / LoRA / DoRA.")
    if not target_suffixes:
        raise ValueError("Expected at least one target suffix.")

    points_by_suffix: Dict[str, List[Dict[str, float]]] = {}
    step_labels = set()
    for target_suffix in target_suffixes:
        suffix_points: List[Dict[str, float]] = []
        for spec in specs:
            suffix_points.extend(
                collect_weight_decomposition_points(
                    base_model=base_model,
                    spec=spec,
                    target_suffix=target_suffix,
                    torch_dtype=torch_dtype,
                    vector_dim=vector_dim,
                )
            )
        points_by_suffix[target_suffix] = suffix_points
        step_labels.update(str(p["step_label"]) for p in suffix_points)

    all_layers = sorted(
        {
            int(point["layer"])
            for suffix_points in points_by_suffix.values()
            for point in suffix_points
        }
    )
    layer_colors = plt.cm.tab20(torch.linspace(0, 1, max(len(all_layers), 2)).tolist())
    layer_to_color = {layer: layer_colors[i] for i, layer in enumerate(all_layers)}
    sorted_step_labels = sorted(step_labels, key=_step_sort_key)
    markers = ["o", "s", "^", "D", "P", "X"]
    step_to_marker = {step: markers[i % len(markers)] for i, step in enumerate(sorted_step_labels)}

    fig, axes = plt.subplots(
        len(target_suffixes),
        len(specs),
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, target_suffix in enumerate(target_suffixes):
        suffix_points = points_by_suffix[target_suffix]
        for col_idx, spec in enumerate(specs):
            ax = axes[row_idx][col_idx]
            points = [p for p in suffix_points if p["method"] == spec.name]
            for point in points:
                ax.scatter(
                    point["delta_d"],
                    point["delta_m"],
                    c=[layer_to_color[int(point["layer"])]],
                    marker=step_to_marker[str(point["step_label"])],
                    s=80,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.4,
                )

            if len(points) >= 2:
                x = torch.tensor([float(p["delta_d"]) for p in points], dtype=torch.float64)
                y = torch.tensor([float(p["delta_m"]) for p in points], dtype=torch.float64)
                x_centered = x - x.mean()
                denom = torch.sum(x_centered * x_centered)
                if denom.item() > 0:
                    slope = torch.sum(x_centered * (y - y.mean())) / denom
                    intercept = y.mean() - slope * x.mean()
                    x_line = torch.linspace(float(x.min().item()), float(x.max().item()), steps=100)
                    y_line = intercept + slope * x_line
                    ax.plot(
                        x_line.tolist(),
                        y_line.tolist(),
                        color="blue",
                        linewidth=2.0,
                        alpha=0.9,
                    )

            if points:
                y_values = [float(p["delta_m"]) for p in points]
                y_min = min(y_values)
                y_max = max(y_values)
                y_span = y_max - y_min
                y_pad = max(y_span * y_padding_ratio, 1e-5)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

            if row_idx == 0:
                ax.set_title(spec.name)
            ax.set_xlabel("ΔD")
            ax.set_ylabel(f"{target_suffix} ΔM")
            ax.grid(True, alpha=0.25)

    legend_handles = []
    for step in sorted_step_labels:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=step_to_marker[step],
                color="black",
                linestyle="None",
                label=step,
                markersize=8,
            )
        )

    axes[0][-1].legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )

    plot_title = title or "Weight decomposition analysis"
    fig.suptitle(plot_title, fontsize=14)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, points_by_suffix


def _step_sort_key(step_label: str) -> Tuple[int, str]:
    if step_label == "Final step":
        return (10**9, step_label)
    match = re.search(r"(\d+)", step_label)
    return (int(match.group(1)) if match else 10**8, step_label)


def pretty_print_snapshot_schedule(total_update_steps: int, num_intermediate: int = 3) -> None:
    steps = snapshot_update_steps(total_update_steps, num_intermediate=num_intermediate)
    print("Suggested analysis checkpoints:")
    for idx, step in enumerate(steps, start=1):
        label = f"Inter step {idx}" if idx < len(steps) else "Final step"
        print(f"  {label}: update step {step}")
