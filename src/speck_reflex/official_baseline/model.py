"""Models taken from pinned Sinabs v3.1.3 examples."""

from __future__ import annotations

import random
from pathlib import Path

import nir
import numpy as np
import sinabs
import sinabs.layers as sl
import torch
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from torch import nn

SEED = 17
INPUT_SHAPE = (2, 34, 34)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_random_quick_start_model(batch_size: int = 1) -> nn.Sequential:
    """Official quick-start topology with deterministic random Xavier weights."""
    model = nn.Sequential(
        nn.Conv2d(2, 8, 3, padding=1, bias=False),
        sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(8, 16, 3, padding=1, bias=False),
        sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(16, 16, 3, padding=1, stride=2, bias=False),
        sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10, bias=False),
        sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    )
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(layer.weight.data)
    return model


def load_official_nir_model(checkpoint: Path):
    """Load the NIR model distributed with Sinabs v3.1.3."""
    return sinabs.from_nir(nir.read(checkpoint), batch_size=1)


def nir_output_tensor(output: object) -> torch.Tensor:
    # COMPATIBILITY_PATCH: nirtorch in the pinned environment returns
    # (output_tensor, GraphExecutorState), whereas the upstream notebook uses a tensor.
    if isinstance(output, tuple):
        return output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"unexpected NIR output type: {type(output)!r}")
    return output


def sequentialize_nir_model(graph_model) -> nn.Sequential:
    """Follow the official nir_to_speck notebook conversion exactly."""
    model = nn.Sequential(*(node.elem for node in graph_model.execution_order)).cpu()
    for module in model:
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.bias is not None:
            if torch.count_nonzero(module.bias).item() != 0:
                raise ValueError("official NIR model has a non-zero bias unsupported by Specksim")
            # COMPATIBILITY_PATCH: the official NIR stores explicit all-zero bias
            # parameters; Specksim 3.1.3 rejects bias objects even when they are zero.
            module.bias = None
        if isinstance(module, sl.IAF):
            module.spike_threshold = nn.Parameter(torch.tensor([module.spike_threshold.flatten()[0]]))
            module.min_v_mem = nn.Parameter(torch.tensor([module.min_v_mem.flatten()[0]]))
    return model


def reset_states(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(layer, sl.StatefulLayer):
            # COMPATIBILITY_PATCH: Sinabs 3.1.3 reset_states() calls detach_() on a
            # view, rejected by Torch 2.10. Re-register detached zero buffers.
            for name, buffer in list(layer.named_buffers(recurse=False)):
                layer.register_buffer(name, torch.zeros_like(buffer.detach()))
