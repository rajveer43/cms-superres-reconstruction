from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn


class ActivationGradientTracker:
    """Forward/backward hooks that record per-module activation and grad L2 norms.

    Usage:
        tracker = ActivationGradientTracker(model, target_names=["blocks.0", "blocks.1"])
        ... train one step ...
        stats = tracker.flush()  # dict[name -> {"act_norm", "grad_norm"}]
    """

    def __init__(self, model: nn.Module, target_substrings: Iterable[str]) -> None:
        self.targets = tuple(target_substrings)
        self._act: dict[str, float] = {}
        self._grad: dict[str, float] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register(model)

    def _matches(self, name: str) -> bool:
        return any(t in name for t in self.targets)

    def _register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if not self._matches(name):
                continue
            self._handles.append(module.register_forward_hook(self._make_fwd(name)))
            self._handles.append(module.register_full_backward_hook(self._make_bwd(name)))

    def _make_fwd(self, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                self._act[name] = float(tensor.detach().float().norm().item())
        return hook

    def _make_bwd(self, name: str):
        def hook(_module, _grad_input, grad_output):
            g = grad_output[0]
            if isinstance(g, torch.Tensor):
                self._grad[name] = float(g.detach().float().norm().item())
        return hook

    def flush(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(dict)
        for k, v in self._act.items():
            out[k]["act_norm"] = v
        for k, v in self._grad.items():
            out[k]["grad_norm"] = v
        self._act.clear()
        self._grad.clear()
        return dict(out)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
