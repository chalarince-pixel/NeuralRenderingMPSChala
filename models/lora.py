"""
LoRA (Low-Rank Adaptation) Module from Scratch
Implements: h = W_0 * x + (alpha / r) * (B * A) * x
Base weights W_0 are completely frozen; only A and B receive gradients.
Supports direct export/import to zero-copy .safetensors format.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer or replaces it with a low-rank adapter.
    W_0 is frozen.
    A is initialized with Gaussian noise (scaled by 1/r).
    B is initialized to 0, ensuring Delta W = 0 at step 0 (exact identity preservation).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 1. Freeze base weight and bias
        self.base_weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.base_bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.base_bias = None

        # 2. Trainable low-rank adapter matrices
        # A: (rank, in_features)
        # B: (out_features, rank)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base projection: W_0 * x
        base_out = torch.nn.functional.linear(x, self.base_weight, self.base_bias)

        # LoRA branch: (alpha / r) * (B * A * x)
        # (batch, ..., in_features) -> (batch, ..., rank) -> (batch, ..., out_features)
        lora_in = self.dropout(x)
        lora_mid = torch.matmul(lora_in, self.lora_A.t())
        lora_out = torch.matmul(lora_mid, self.lora_B.t()) * self.scaling

        return base_out + lora_out


def inject_lora(model: nn.Module, target_names: List[str], rank: int = 4, alpha: float = 1.0) -> Dict[str, LoRALinear]:
    """
    Searches the model for linear layers matching target_names, replaces them with LoRALinear,
    and returns a dictionary of injected adapters.
    """
    injected_adapters = {}

    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name
            if any(target in full_child_name for target in target_names) and isinstance(child_module, nn.Linear):
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
                injected_adapters[full_child_name] = lora_layer

    return injected_adapters


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts only the trainable LoRA parameters (lora_A and lora_B)."""
    lora_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_dict[name] = param.data.contiguous()
    return lora_dict


def save_lora_safetensors(model: nn.Module, filepath: str, metadata: Optional[Dict[str, str]] = None):
    """Saves only the low-rank adapter weights into an efficient .safetensors file."""
    state_dict = extract_lora_state_dict(model)
    meta = metadata or {"format": "lora_adapter", "framework": "pytorch_mps"}
    save_file(state_dict, filepath, metadata=meta)
    print(f"[SafeTensors] Saved {len(state_dict)} LoRA tensors to -> {filepath}")


def load_lora_safetensors(model: nn.Module, filepath: str):
    """Loads LoRA weights from a .safetensors file into the model."""
    loaded = load_file(filepath)
    model_state = model.state_dict()
    for k, v in loaded.items():
        if k in model_state:
            model_state[k].copy_(v)
    print(f"[SafeTensors] Loaded {len(loaded)} LoRA tensors from -> {filepath}")
