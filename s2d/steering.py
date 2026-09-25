from typing import List, Tuple, Dict, Any, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def find_transformer_blocks(model) -> List[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    raise ValueError("Unknown model structure. Extend find_transformer_blocks().")

class SVSteering(nn.Module):
    def __init__(self, hidden_size: int, init_scale: float = 0.0):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        if init_scale and init_scale > 0:
            nn.init.normal_(self.v, mean=0.0, std=float(init_scale))

class SVHookManager:
    def __init__(self, blocks: List[nn.Module], sv_module: SVSteering,
                 lam: float, steer_layer: int, steer_pos: str = "residual"):
        self.blocks = blocks
        self.sv_module = sv_module
        self.lam = float(lam)
        self.steer_layer = int(steer_layer)
        self.steer_pos = str(steer_pos)
        self.handles =[]

    def _inject(self, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
            v = self.sv_module.v.to(device=h.device, dtype=h.dtype)
            h = h + self.lam * v
            return (h,) + outputs[1:]
        else:
            h = outputs
            v = self.sv_module.v.to(device=h.device, dtype=h.dtype)
            return h + self.lam * v

    def _make_hook(self):
        def fn(module, inputs, outputs):
            return self._inject(outputs)
        return fn

    def _get_target_module(self, block: nn.Module) -> nn.Module:
        if self.steer_pos == "residual":
            return block
        if self.steer_pos == "attention":
            if not hasattr(block, "self_attn"):
                raise AttributeError("Block has no attribute self_attn.")
            return block.self_attn
        if self.steer_pos == "mlp":
            if not hasattr(block, "mlp"):
                raise AttributeError("Block has no attribute mlp.")
            return block.mlp
        raise ValueError(f"Unknown steer_pos={self.steer_pos}")

    def register(self):
        self.remove()
        if not (0 <= self.steer_layer < len(self.blocks)):
            raise ValueError(f"--steer_layer={self.steer_layer} out of range.")
        block = self.blocks[self.steer_layer]
        target = self._get_target_module(block)
        self.handles.append(target.register_forward_hook(self._make_hook()))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles =[]

class EvaluationSVHookManager:
    """
    Inject SV at ONE chosen layer and ONE chosen component:
      - residual: hook on the whole block output
      - attention: hook on block.self_attn output
      - mlp: hook on block.mlp output
    """
    def __init__(
        self,
        blocks: List[nn.Module],
        sv_module: SVSteering,
        lam: float,
        steer_layer: int,
        steer_pos: str = "residual",
    ):
        self.blocks = blocks
        self.sv_module = sv_module
        self.lam = float(lam)
        self.steer_layer = int(steer_layer)
        self.steer_pos = str(steer_pos)
        self.handles = []

    def _inject(self, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
            if torch.is_tensor(h) and h.dim() == 3:
                v = self.sv_module.v.to(device=h.device, dtype=h.dtype)
                h = h + self.lam * v
                return (h,) + outputs[1:]
            return outputs
        else:
            h = outputs
            if torch.is_tensor(h) and h.dim() == 3:
                v = self.sv_module.v.to(device=h.device, dtype=h.dtype)
                return h + self.lam * v
            return outputs

    def _make_hook(self):
        def fn(module, inputs, outputs):
            return self._inject(outputs)
        return fn

    def _get_target_module(self, block: nn.Module) -> nn.Module:
        if self.steer_pos == "residual":
            return block
        if self.steer_pos == "attention":
            if not hasattr(block, "self_attn"):
                raise AttributeError("Block has no attribute self_attn; cannot use --steer_pos attention.")
            return block.self_attn
        if self.steer_pos == "mlp":
            if not hasattr(block, "mlp"):
                raise AttributeError("Block has no attribute mlp; cannot use --steer_pos mlp.")
            return block.mlp
        raise ValueError(f"Unknown steer_pos={self.steer_pos}")

    def register(self):
        self.remove()
        if not (0 <= self.steer_layer < len(self.blocks)):
            raise ValueError(f"steer_layer out of range: {self.steer_layer}, num_layers={len(self.blocks)}")
        block = self.blocks[self.steer_layer]
        target = self._get_target_module(block)
        self.handles.append(target.register_forward_hook(self._make_hook()))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
