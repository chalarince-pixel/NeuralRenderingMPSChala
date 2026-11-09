"""
Torch Graph & MPS Kernel Debugger
Profiles Metal Performance Shader execution, inspects memory allocations,
verifies gradient flow, and detects silent CPU synchronization fallbacks.
"""

import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class MPSGraphDebugger:
    """
    Diagnostic tool for tracking PyTorch computational graph execution on Apple Silicon MPS.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.is_mps = device.type == "mps"

    def get_memory_info(self) -> Dict[str, float]:
        """Returns current allocated and driver memory in Megabytes."""
        if self.is_mps and hasattr(torch.mps, "current_allocated_memory"):
            allocated_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            driver_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
            return {"allocated_mb": allocated_mb, "driver_mb": driver_mb}
        return {"allocated_mb": 0.0, "driver_mb": 0.0}

    def verify_tensor_devices(self, tensor_dict: Dict[str, torch.Tensor]) -> bool:
        """Verifies that all tensors reside on the expected MPS device."""
        all_on_device = True
        for name, tensor in tensor_dict.items():
            if tensor is not None and tensor.device.type != self.device.type:
                print(f"[WARNING] Tensor '{name}' is on {tensor.device}, expected {self.device}!")
                all_on_device = False
        return all_on_device

    def inspect_lora_gradients(self, model: nn.Module) -> Dict[str, bool]:
        """
        Validates the fundamental contract of LoRA:
        1. Base parameters must have requires_grad == False or grad is None (frozen).
        2. LoRA parameters (lora_A, lora_B) must have requires_grad == True and grad is not None.
           (Note: at step 1, if B=0, dL/dA is mathematically 0, but grad is attached in the graph).
        """
        status = {"base_frozen": True, "lora_active": True}
        active_lora_count = 0
        frozen_base_count = 0

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                if param.grad is None:
                    print(f"[LoRA ERROR] LoRA parameter '{name}' not connected to autograd graph!")
                    status["lora_active"] = False
                else:
                    active_lora_count += 1
            elif "base_weight" in name or "base_bias" in name:
                if param.requires_grad or (param.grad is not None and not torch.all(param.grad == 0)):
                    print(f"[LoRA ERROR] Base parameter '{name}' received gradient, but should be frozen!")
                    status["base_frozen"] = False
                else:
                    frozen_base_count += 1

        print(f"[Autograd Check] Verified {frozen_base_count} frozen base params, {active_lora_count} active LoRA params connected to autograd.")
        return status

    def benchmark_step(
        self,
        forward_fn,
        backward_fn=None,
        num_warmup: int = 3,
        num_iters: int = 10,
    ) -> Dict[str, float]:
        """
        Precise timing benchmark for MPS kernel execution using torch.mps.synchronize().
        """
        # Warmup
        for _ in range(num_warmup):
            out = forward_fn()
            if backward_fn and out is not None:
                backward_fn(out)
        if self.is_mps:
            torch.mps.synchronize()

        # Timed iterations
        start_time = time.perf_counter()
        for _ in range(num_iters):
            out = forward_fn()
            if backward_fn and out is not None:
                backward_fn(out)
        if self.is_mps:
            torch.mps.synchronize()
        end_time = time.perf_counter()

        avg_latency_ms = ((end_time - start_time) / num_iters) * 1000.0
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        mem_info = self.get_memory_info()
        print(f"[MPS Benchmark] Step Time: {avg_latency_ms:.2f} ms | Throughput: {fps:.1f} steps/sec | Memory: {mem_info['allocated_mb']:.1f} MB")
        return {"avg_latency_ms": avg_latency_ms, "fps": fps, **mem_info}
