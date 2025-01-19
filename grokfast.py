from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Literal


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach()) # .cpu())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads

def gradfilter_with_depth_scaling(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb_max: float = 5.0,
    lamb_min: float = 1.0,
    d_max: int = 12,  # Total number of transformer layers
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False,
    embedding_layer_name: str = "embedding",
    final_and_output_layer_names: List[str] = ["ln_f", "head"],  # Default final and output layer names
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            # Append current gradient to deque
            grads[n].append(p.grad.data.detach())  # Store gradients for filtering

            # Determine depth or position
            if embedding_layer_name in n:
                position = "embedding"
                depth = 0  # Embedding layers are assigned depth 0
            elif "layers" in n:
                # Extract depth information from name, e.g., "layers.0", "layers.1"
                depth = int(n.split(".")[1]) + 1  # Increment depth for transformer layers
                position = f"layer_{depth}"
            elif any(layer_name in n for layer_name in final_and_output_layer_names):
                position = "final_or_output"
                depth = d_max + 1  # Final and output layers are d_max + 1
            else:
                position = "other"
                depth = d_max  # Default for unclassified layers

            # Adjust lambda based on updated depths
            lambda_d = lamb_max - (depth / (d_max + 1)) * (lamb_max - lamb_min)

            # Apply gradient filtering if warmup condition is met
            if not warmup or (len(grads[n]) == window_size and not trigger):
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")

                # Modify gradient based on depth and position
                if position == "embedding":
                    p.grad.data = p.grad.data + avg * lamb_max  # Embedding gets max lambda
                elif "layer" in position:
                    p.grad.data = p.grad.data + avg * lambda_d  # Scale by depth
                elif position == "final_or_output":
                    p.grad.data = p.grad.data + avg * (lamb_min / 2)  # Final/output get min lambda
                else:
                    p.grad.data = p.grad.data + avg * lamb_min  # Default behavior for other params

            # Optionally store metadata for later inspection
            grads[n] = {
                "queue": grads[n],
                "depth": depth,
                "lambda": lambda_d,
            }

    return grads

    
    
def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads