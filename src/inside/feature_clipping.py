from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .utils import ensure_dir


def resolve_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    named_modules = dict(model.named_modules())

    if module_name in named_modules:
        module = named_modules[module_name]
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"resolved object for '{module_name}' is not a torch.nn.Module")
        return module

    current = model
    for part in module_name.split("."):
        if not hasattr(current, part):
            available = [name for name in named_modules.keys() if name.endswith(part)]
            raise ValueError(
                f"module path '{module_name}' not found at '{part}'. "
                f"Some available module names ending with '{part}': {available[:20]}"
            )
        current = getattr(current, part)

    if not isinstance(current, torch.nn.Module):
        raise ValueError(f"resolved object for '{module_name}' is not a torch.nn.Module")

    return current


def extract_main_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("hook output is not a tensor or tuple starting with a tensor")


def replace_main_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return new_tensor
    if isinstance(output, tuple):
        if len(output) == 0:
            raise TypeError("cannot replace tensor in empty tuple output")
        return (new_tensor, *output[1:])
    raise TypeError("hook output is not a tensor or tuple")


@dataclass
class FeatureClippingThresholds:
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    module_name: str
    lower_percentile: float
    upper_percentile: float
    n_tokens_total: int
    hidden_dim: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bounds": self.lower_bounds.cpu(),
            "upper_bounds": self.upper_bounds.cpu(),
            "module_name": self.module_name,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "n_tokens_total": self.n_tokens_total,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureClippingThresholds":
        return cls(
            lower_bounds=data["lower_bounds"],
            upper_bounds=data["upper_bounds"],
            module_name=data["module_name"],
            lower_percentile=float(data["lower_percentile"]),
            upper_percentile=float(data["upper_percentile"]),
            n_tokens_total=int(data["n_tokens_total"]),
            hidden_dim=int(data["hidden_dim"]),
        )


class ActivationCollector:
    def __init__(self, store_dtype: torch.dtype = torch.float32) -> None:
        self.store_dtype = store_dtype
        self.chunks: List[torch.Tensor] = []
        self.n_tokens_total = 0
        self.hidden_dim: Optional[int] = None
        self.handle = None

    def hook_fn(self, module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        tensor = extract_main_tensor(output)

        if tensor.ndim < 2:
            raise ValueError(f"expected tensor with at least 2 dims, got shape {tuple(tensor.shape)}")

        hidden_dim = tensor.shape[-1]
        flat = tensor.detach().reshape(-1, hidden_dim).to("cpu", dtype=self.store_dtype)

        self.chunks.append(flat)
        self.n_tokens_total += flat.shape[0]
        self.hidden_dim = hidden_dim

        return output

    def attach(self, module: torch.nn.Module) -> None:
        if self.handle is not None:
            raise RuntimeError("collector hook is already attached")
        self.handle = module.register_forward_hook(self.hook_fn)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_activations(self) -> torch.Tensor:
        if not self.chunks:
            raise ValueError("no activations were collected")
        return torch.cat(self.chunks, dim=0)


class FeatureClipper:
    def __init__(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> None:
        if lower_bounds.ndim != 1 or upper_bounds.ndim != 1:
            raise ValueError("lower_bounds and upper_bounds must be 1D tensors")

        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError("lower_bounds and upper_bounds must have the same shape")

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.handle = None
        self.total_values = 0
        self.total_clipped = 0

    def hook_fn(self, module: torch.nn.Module, inputs: Any, output: Any) -> Any:
        tensor = extract_main_tensor(output)

        if tensor.shape[-1] != self.lower_bounds.shape[0]:
            raise ValueError(
                f"hidden dim mismatch in clipping hook: tensor has {tensor.shape[-1]}, "
                f"thresholds have {self.lower_bounds.shape[0]}"
            )

        lower = self.lower_bounds.to(device=tensor.device, dtype=tensor.dtype)
        upper = self.upper_bounds.to(device=tensor.device, dtype=tensor.dtype)

        clipped_tensor = torch.maximum(torch.minimum(tensor, upper), lower)

        clipped_mask = (clipped_tensor != tensor)
        self.total_values += clipped_mask.numel()
        self.total_clipped += int(clipped_mask.sum().item())

        return replace_main_tensor(output, clipped_tensor)

    def attach(self, module: torch.nn.Module) -> None:
        if self.handle is not None:
            raise RuntimeError("clipping hook is already attached")
        self.handle = module.register_forward_hook(self.hook_fn)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset_stats(self) -> None:
        self.total_values = 0
        self.total_clipped = 0

    @property
    def clip_fraction(self) -> Optional[float]:
        if self.total_values == 0:
            return None
        return float(self.total_clipped / self.total_values)


def compute_thresholds_from_activations(
    activations: torch.Tensor,
    module_name: str,
    lower_percentile: float,
    upper_percentile: float,
) -> FeatureClippingThresholds:
    if activations.ndim != 2:
        raise ValueError(f"expected activations with shape [n_tokens, hidden_dim], got {tuple(activations.shape)}")

    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0.0 <= lower < upper <= 100.0")

    lower_q = lower_percentile / 100.0
    upper_q = upper_percentile / 100.0

    lower_bounds = torch.quantile(activations, q=lower_q, dim=0)
    upper_bounds = torch.quantile(activations, q=upper_q, dim=0)

    return FeatureClippingThresholds(
        lower_bounds=lower_bounds.cpu(),
        upper_bounds=upper_bounds.cpu(),
        module_name=module_name,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        n_tokens_total=int(activations.shape[0]),
        hidden_dim=int(activations.shape[1]),
    )


def save_thresholds(thresholds: FeatureClippingThresholds, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(thresholds.to_dict(), path)


def load_thresholds(path: Path | str) -> FeatureClippingThresholds:
    path = Path(path)
    data = torch.load(path, map_location="cpu")
    return FeatureClippingThresholds.from_dict(data)


class TemporaryFeatureClipping:
    def __init__(self, model: torch.nn.Module, thresholds: FeatureClippingThresholds) -> None:
        self.model = model
        self.thresholds = thresholds
        self.module = resolve_module_by_name(model, thresholds.module_name)
        self.clipper = FeatureClipper(
            lower_bounds=thresholds.lower_bounds,
            upper_bounds=thresholds.upper_bounds,
        )

    def __enter__(self) -> "TemporaryFeatureClipping":
        self.clipper.reset_stats()
        self.clipper.attach(self.module)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.clipper.detach()

    @property
    def clip_fraction(self) -> Optional[float]:
        return self.clipper.clip_fraction
