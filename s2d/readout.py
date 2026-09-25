from typing import List, Tuple, Dict, Any, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class ReadoutHookManager:
    def __init__(self, blocks: List[nn.Module], readout_layers: List[int]):
        self.blocks = blocks
        self.readout_layers = [int(x) for x in readout_layers]
        self.handles = []
        self.cache: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def fn(module, inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            self.cache[layer_idx] = h
            return outputs
        return fn

    def register(self):
        self.remove()
        self.cache = {}
        for l in self.readout_layers:
            self.handles.append(self.blocks[l].register_forward_hook(self._make_hook(l)))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles =[]
        self.cache = {}

    def get_ordered(self) -> List[torch.Tensor]:
        hs =[]
        for l in self.readout_layers:
            hs.append(self.cache[l])
        return hs

def last_ratio_tokens_mean_pool(
    h: torch.Tensor,
    attn: torch.Tensor,
    ratio: float
) -> torch.Tensor:
    if not (0.0 < ratio <= 1.0):
        raise ValueError("last_token_ratio must be in (0, 1].")

    B, T, D = h.shape
    lengths = attn.long().sum(dim=1).clamp(min=1)
    reps =[]

    for i in range(B):
        L = int(lengths[i].item())
        kk = max(1, int(np.ceil(float(ratio) * L)))

        valid_idx = attn[i].nonzero(as_tuple=False).squeeze(-1)
        take_idx = valid_idx[-kk:]
        seg = h[i].index_select(dim=0, index=take_idx)
        reps.append(seg.mean(dim=0))

    return torch.stack(reps, dim=0)

def aggregate_poststeer_lastlayers_lastratio_tokens(
    h_list: List[torch.Tensor],
    attn: torch.Tensor,
    last_token_ratio: float
) -> torch.Tensor:
    per_layer =[]
    for h in h_list:
        r = last_ratio_tokens_mean_pool(h, attn, ratio=last_token_ratio)
        per_layer.append(r)
    rep = torch.stack(per_layer, dim=0).mean(dim=0)
    return rep.float()

class EvaluationReadoutHookManager:
    """
    Capture outputs of selected transformer blocks.
    """
    def __init__(self, blocks: List[nn.Module], readout_layers: List[int]):
        self.blocks = blocks
        self.readout_layers = [int(x) for x in readout_layers]
        self.handles = []
        self.cache: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.cache = {}

    def _make_hook(self, layer_idx: int):
        def fn(module, inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            if torch.is_tensor(h) and h.dim() == 3:
                self.cache[layer_idx] = h
            return outputs
        return fn

    def register(self):
        self.remove()
        self.cache = {}
        for l in self.readout_layers:
            if not (0 <= l < len(self.blocks)):
                raise ValueError(f"readout layer {l} out of range [0, {len(self.blocks)-1}]")
            self.handles.append(self.blocks[l].register_forward_hook(self._make_hook(l)))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self.cache = {}

    def get_ordered(self) -> List[torch.Tensor]:
        hs = []
        for l in self.readout_layers:
            if l not in self.cache:
                raise RuntimeError(f"Layer {l} not captured. Check hooks/model forward.")
            hs.append(self.cache[l])
        return hs

def evaluation_last_ratio_tokens_mean_pool(
    h: torch.Tensor,         # [B,T,D]
    attn: torch.Tensor,      # [B,T]
    ratio: float,
    padding_side: str = "left",
) -> torch.Tensor:
    """
    For sample i:
      L_i = number of valid tokens
      k_i = max(1, ceil(ratio * L_i))
    Then mean-pool the last k_i valid tokens.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError("last_token_ratio must be in (0, 1].")

    B, T, D = h.shape
    lengths = attn.long().sum(dim=1).clamp(min=1)  # [B]
    kk = torch.ceil(lengths.float() * float(ratio)).long().clamp(min=1)  # [B]
    pos = torch.arange(T, device=h.device).view(1, T)

    if padding_side == "left":
        valid_start = (T - lengths).view(B, 1)
        start = (T - kk).view(B, 1)
        mask = (pos >= start) & (pos >= valid_start)
    else:
        valid_end = lengths.view(B, 1)
        start = (lengths - kk).view(B, 1)
        mask = (pos >= start) & (pos < valid_end)

    mask = mask.to(dtype=h.dtype)
    denom = mask.sum(dim=1).clamp(min=1).view(B, 1)
    rep = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
    return rep

def evaluation_aggregate_poststeer_lastlayers_lastratio_tokens(
    h_list: List[torch.Tensor],
    attn: torch.Tensor,
    last_token_ratio: float,
    padding_side: str,
) -> torch.Tensor:
    per_layer = []
    for h in h_list:
        r = evaluation_last_ratio_tokens_mean_pool(
            h,
            attn,
            ratio=last_token_ratio,
            padding_side=padding_side,
        )
        per_layer.append(r)
    rep = torch.stack(per_layer, dim=0).mean(dim=0)
    return rep.float()
