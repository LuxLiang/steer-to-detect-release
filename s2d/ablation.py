#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_sv_eval_no_steer_centroid_train_valid_thresh.py

目标：
- 维持与原脚本相同的输出结构（AUC/阈值/summary.json/每样本 scores.json）
- 不注入 steering vector（不做 h <- h + lam * v）
- centroid 不再从 ckpt 读取，而是用“TRAIN 数据集均值”来构造：
    centroid[c] = mean_{i: y_i=c} rep(x_i)
  其中 rep(x) 是当前 readout+pooling 得到的 FP32 表示
- 阈值严格在 VALID 数据集上选择
- readout 仍沿用 “post-steer layers, last N” 的规则，以保证评估流程一致
- 仍保留 cos_temp（从 ckpt 读取或参数覆盖），用于 LLR 或 softmax score

实现约定：
- centroid 默认从 TRAIN 数据集计算（可用 --centroid_data_path 指定其他数据集）
- threshold 默认从 VALID 数据集计算
- 若启用 --centroid_normalize_per_sample，则先对每个样本 rep 做 L2 normalize，再求类别均值，
  最后对 centroid 再做一次 L2 normalize
"""

import os
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)


# -------------------------
# Data utilities
# -------------------------
def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data)}.")
    return data


def expand_pairs_to_samples(
    pairs: List[Dict[str, Any]],
    llm_field: str,
    human_field: str,
    max_pairs: Optional[int] = None,
) -> Tuple[List[str], List[int], List[str], Dict[str, int]]:
    """
    Returns:
      texts  : List[str]
      labels : List[int]   (1=llm, 0=human)
      ids    : List[str]   (e.g., "42_llm", "42_human")
      stats  : Dict
    """
    texts, labels, ids = [], [], []
    skipped = 0
    used = 0

    subset = pairs[:max_pairs] if max_pairs is not None else pairs

    for pair_id, item in enumerate(subset):
        llm_text = item.get(llm_field, None)
        human_text = item.get(human_field, None)

        if not (
            isinstance(llm_text, str) and llm_text.strip() and
            isinstance(human_text, str) and human_text.strip()
        ):
            skipped += 1
            continue

        texts.append(llm_text)
        labels.append(1)
        ids.append(f"{pair_id}_llm")

        texts.append(human_text)
        labels.append(0)
        ids.append(f"{pair_id}_human")

        used += 1

    stats = {
        "used_pairs": used,
        "skipped_pairs": skipped,
        "total_pairs_seen": len(subset),
        "total_samples": len(texts)
    }

    return texts, labels, ids, stats


# -------------------------
# Model block discovery
# -------------------------
def find_transformer_blocks(model) -> List[nn.Module]:
    # LLaMA/Qwen-like
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT2-like
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # OPT-like
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    raise ValueError("Unknown model structure. Extend find_transformer_blocks().")


# -------------------------
# Readout hooks
# -------------------------
class ReadoutHookManager:
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
                self.cache[layer_idx] = h  # [B,T,D]
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


# -------------------------
# Pooling: last K tokens masked mean (vectorized), then layer mean
# -------------------------
def last_k_tokens_mean_pool(
    h: torch.Tensor,         # [B,T,D]
    attn: torch.Tensor,      # [B,T] 1=valid 0=pad
    k: int,
    padding_side: str = "left",
) -> torch.Tensor:
    """
    Vectorized last-K mean pooling over valid tokens.
    """
    if k <= 0:
        raise ValueError("last_k_tokens must be > 0")

    B, T, D = h.shape
    lengths = attn.long().sum(dim=1).clamp(min=1)  # [B]
    kk = torch.minimum(lengths, torch.full_like(lengths, int(k)))  # [B]
    pos = torch.arange(T, device=h.device).view(1, T)  # [1,T]

    if padding_side == "left":
        valid_start = (T - lengths).view(B, 1)
        start = (T - kk).view(B, 1)
        mask = (pos >= start) & (pos >= valid_start)
    else:
        valid_end = lengths.view(B, 1)
        start = (lengths - kk).view(B, 1)
        mask = (pos >= start) & (pos < valid_end)

    mask = mask.to(dtype=h.dtype)                  # [B,T]
    denom = mask.sum(dim=1).clamp(min=1).view(B, 1)
    rep = (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]
    return rep


def aggregate_poststeer_lastlayers_lastk_tokens(
    h_list: List[torch.Tensor],  # each [B,T,D]
    attn: torch.Tensor,          # [B,T]
    last_k_tokens: int,
    padding_side: str,
) -> torch.Tensor:
    per_layer = []
    for h in h_list:
        r = last_k_tokens_mean_pool(h, attn, k=last_k_tokens, padding_side=padding_side)
        per_layer.append(r)
    rep = torch.stack(per_layer, dim=0).mean(dim=0)  # [B,D]
    return rep.float()  # FP32


# -------------------------
# Scoring + metrics
# -------------------------
@torch.no_grad()
def score_texts_llr(
    model, tokenizer, device,
    readout_mgr,
    texts: List[str],
    labels: List[int],
    ids: Sequence,
    centroids: torch.Tensor,   # shape [2, D], fp32
    batch_size: int,
    cos_temp: float,           # temperature T
    max_length: int,
    last_k_tokens: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LLR score:
      rep = pooled representation (FP32, L2 normalized)
      z_c = (rep · mu_c) / T
      llr = z_1 - z_0
    Returns:
      llr_scores: (N,)
      labels:     (N,)
      ids:        (N,)
    """
    assert len(texts) == len(labels) == len(ids), "texts, labels, and ids must have equal lengths"

    model.eval()
    centroids = centroids.to(device=device, dtype=torch.float32)
    centroids = F.normalize(centroids, p=2, dim=-1, eps=1e-6)  # [2,D]

    T = float(max(cos_temp, 1e-6))

    all_scores, all_y, all_ids = [], [], []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_y = labels[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        readout_mgr.clear()
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)

        h_list = readout_mgr.get_ordered()
        rep = aggregate_poststeer_lastlayers_lastk_tokens(
            h_list=h_list,
            attn=attn,
            last_k_tokens=last_k_tokens,
            padding_side=tokenizer.padding_side,
        )  # [B,D] fp32
        rep = F.normalize(rep, p=2, dim=-1, eps=1e-6)

        logits = (rep @ centroids.T) / T        # [B,2]
        llr = logits[:, 1] - logits[:, 0]       # [B]

        all_scores.append(llr.detach().cpu().numpy().astype(np.float32))
        all_y.append(np.asarray(batch_y, dtype=np.int32))
        all_ids.append(np.asarray(batch_ids))

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_ids, axis=0),
    )


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, (fp + tn))
    tpr = tp / max(1, (tp + fn))

    return {
        "conf_matrix": cm.tolist(),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "fpr": float(fpr),
        "tpr": float(tpr),
    }


def select_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
    j = tpr - fpr
    i = int(np.argmax(j))
    return {
        "threshold": float(thr[i]),
        "fpr": float(fpr[i]),
        "tpr": float(tpr[i]),
        "youden_j": float(j[i]),
    }


def select_threshold_np_quantile(y_true, y_score, alpha):
    neg = y_score[y_true == 0]
    tau = float(np.quantile(neg, 1.0 - alpha, method="higher"))
    y_pred = (y_score >= tau).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / max(1, fp + tn)
    tpr = tp / max(1, tp + fn)
    return {"threshold": tau, "fpr": float(fpr), "tpr": float(tpr), "n_neg": int(len(neg))}


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) > 0:
        tpr_step = float(np.max(tpr[valid]))
    else:
        tpr_step = float(tpr[int(np.argmin(fpr))])
    tpr_interp = float(np.interp(target_fpr, fpr, tpr))
    return {"tpr_step": tpr_step, "tpr_interp": tpr_interp}


# -------------------------
# Build centroids from dataset mean
# -------------------------
@torch.no_grad()
def compute_centroids_from_dataset_mean(
    model, tokenizer, device,
    readout_mgr,
    texts: List[str],
    labels: List[int],
    batch_size: int,
    max_length: int,
    last_k_tokens: int,
    normalize_per_sample: bool = True,
) -> torch.Tensor:
    """
    centroid[c] = mean rep(x_i) over class c

    建议做法：
    - 先对每个样本 rep 做 L2 normalize（normalize_per_sample=True）
    - 分类别累加求均值
    - 最后对 centroid 再做一次 L2 normalize
    """
    assert len(texts) == len(labels), "texts and labels must have equal lengths"
    model.eval()

    sum_vec = [None, None]
    cnt = [0, 0]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_y = labels[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        readout_mgr.clear()
        _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        h_list = readout_mgr.get_ordered()

        rep = aggregate_poststeer_lastlayers_lastk_tokens(
            h_list=h_list,
            attn=attn,
            last_k_tokens=last_k_tokens,
            padding_side=tokenizer.padding_side,
        )  # [B,D] fp32

        if normalize_per_sample:
            rep = F.normalize(rep, p=2, dim=-1, eps=1e-6)

        rep_cpu = rep.detach().cpu()  # fp32

        for i, y in enumerate(batch_y):
            y = int(y)
            if y not in (0, 1):
                raise ValueError(f"Only binary labels 0/1 are supported, got {y}")
            if sum_vec[y] is None:
                sum_vec[y] = rep_cpu[i].clone()
            else:
                sum_vec[y] += rep_cpu[i]
            cnt[y] += 1

    if cnt[0] == 0 or cnt[1] == 0:
        raise RuntimeError(f"Cannot compute centroids: cnt0={cnt[0]}, cnt1={cnt[1]}")

    c0 = (sum_vec[0] / float(cnt[0])).float()
    c1 = (sum_vec[1] / float(cnt[1])).float()
    centroids = torch.stack([c0, c1], dim=0)  # [2,D]
    centroids = F.normalize(centroids, p=2, dim=-1, eps=1e-6)
    return centroids


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True
    )
    parser.add_argument( # read paras
        "--ckpt_path",
        type=str,
        required=True
    )

    # 关键修改：显式区分 train / valid / test
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="TRAIN dataset path used to compute centroids by default."
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="VALID dataset path used to select thresholds."
    )
    parser.add_argument(
        "--test_data_paths",
        type=str,
        required=True,
        help="Comma-separated list of test json paths."
    )

    # overrides (optional)
    parser.add_argument("--llm_field", type=str, default=None)
    parser.add_argument("--human_field", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    # multi-readout controls (override; default uses ckpt meta when present)
    parser.add_argument("--readout_last_n_layers", type=int, default=None)
    parser.add_argument("--last_k_tokens", type=int, default=None)

    # eval controls
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_fpr", type=float, default=1e-2)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out_dir", type=str, default="results_sv/diff_steer_or_not/no_steer_centroid_train_valid_thresh")

    # no steering
    parser.add_argument("--no_steer", action="store_true", help="Disable steering-vector injection (always off here)")

    # centroid from dataset mean
    parser.add_argument(
        "--centroid_data_path",
        type=str,
        default=None,
        help="Dataset used to compute centroids. Default: train_data_path"
    )
    parser.add_argument(
        "--centroid_normalize_per_sample",
        action="store_true",
        help="If set, L2-normalize each sample rep before averaging (recommended)."
    )
    parser.add_argument(
        "--cos_temp",
        type=float,
        default=None,
        help="Override cos_temp. If None, try reading from ckpt, else default 0.4"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # ---- load ckpt (only for meta like steer_layer/readout/cos_temp/fields) ----
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    steer_layer = int(ckpt.get("steer_layer", 0))

    # cos_temp
    if args.cos_temp is not None:
        cos_temp = float(args.cos_temp)
    else:
        cos_temp = float(ckpt.get("cos_temp", 0.4))

    # fields / max_length
    llm_field = args.llm_field if args.llm_field is not None else str(ckpt.get("llm_field", "direct_prompt"))
    human_field = args.human_field if args.human_field is not None else str(ckpt.get("human_field", "human_text"))
    max_length = int(args.max_length) if args.max_length is not None else int(ckpt.get("max_length", 2048))

    # last_k_tokens
    last_k_tokens = args.last_k_tokens
    if last_k_tokens is None:
        meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}
        last_k_tokens = int(ckpt.get("last_k_tokens", meta.get("last_k_tokens", 64)))
    last_k_tokens = int(last_k_tokens)

    # ---- load model/tokenizer ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- blocks / readout layers ----
    blocks = find_transformer_blocks(model)
    num_layers = len(blocks)

    if not (0 <= steer_layer < num_layers):
        raise ValueError(f"steer_layer={steer_layer} out of range; num_layers={num_layers}")

    # readout_layers: prefer ckpt['readout_layers'] if present
    if "readout_layers" in ckpt and isinstance(ckpt["readout_layers"], (list, tuple)) and len(ckpt["readout_layers"]) > 0:
        readout_layers = [int(x) for x in ckpt["readout_layers"]]
    else:
        post_layers = list(range(steer_layer, num_layers))
        n = args.readout_last_n_layers
        if n is None:
            n = int(ckpt.get("readout_last_n_layers", 8))
        n = int(max(1, n))
        readout_layers = sorted(post_layers[-n:])

    # allow override for readout_last_n_layers even if ckpt has readout_layers
    if args.readout_last_n_layers is not None:
        post_layers = list(range(steer_layer, num_layers))
        n = int(max(1, args.readout_last_n_layers))
        readout_layers = sorted(post_layers[-n:])

    # ---- hooks: NO STEERING, only readout ----
    readout_mgr = ReadoutHookManager(blocks=blocks, readout_layers=readout_layers)
    readout_mgr.register()

    # -------------------------
    # Build centroids from TRAIN average
    # -------------------------
    centroid_path = args.centroid_data_path if args.centroid_data_path is not None else args.train_data_path
    centroid_pairs = load_json_list(centroid_path)
    centroid_texts, centroid_labels, centroid_ids, cstats = expand_pairs_to_samples(
        centroid_pairs, llm_field, human_field
    )
    print(f"[centroid data / TRAIN by default] path={centroid_path} {cstats} -> num_samples={len(centroid_texts)}")

    centroids = compute_centroids_from_dataset_mean(
        model=model, tokenizer=tokenizer, device=device,
        readout_mgr=readout_mgr,
        texts=centroid_texts,
        labels=centroid_labels,
        batch_size=args.batch_size,
        max_length=max_length,
        last_k_tokens=last_k_tokens,
        normalize_per_sample=bool(args.centroid_normalize_per_sample),
    )
    # centroids: torch.Tensor [2,D] fp32 on CPU

    # -------------------------
    # VALID: choose thresholds here
    # -------------------------
    valid_pairs = load_json_list(args.valid_data_path)
    valid_texts, valid_labels, valid_ids, vstats = expand_pairs_to_samples(valid_pairs, llm_field, human_field)
    print(f"[valid data / threshold selection] {vstats} -> num_samples={len(valid_texts)}")

    valid_scores, valid_y, valid_id = score_texts_llr(
        model=model, tokenizer=tokenizer, device=device,
        readout_mgr=readout_mgr,
        texts=valid_texts, labels=valid_labels, ids=valid_ids,
        centroids=centroids,
        batch_size=args.batch_size,
        cos_temp=cos_temp,
        max_length=max_length,
        last_k_tokens=last_k_tokens,
    )
    valid_auc = roc_auc_score(valid_y, valid_scores)

    low_sel = select_threshold_np_quantile(valid_y, valid_scores, args.target_fpr)
    you_sel = select_threshold_youden(valid_y, valid_scores)
    tprlow = tpr_at_fpr(valid_y, valid_scores, args.target_fpr)

    thr_low = float(low_sel["threshold"])
    thr_you = float(you_sel["threshold"])

    valid_result = {
        "roc_auc": float(valid_auc),
        "target_fpr": float(args.target_fpr),

        "lowfpr_select": {
            "threshold": thr_low,
            "achieved_fpr": float(low_sel["fpr"]),
            "tpr_at_selected": float(low_sel["tpr"]),
            "n_neg": int(low_sel["n_neg"]),
        },
        "youden_select": {
            "threshold": thr_you,
            "fpr": float(you_sel["fpr"]),
            "tpr": float(you_sel["tpr"]),
            "youden_j": float(you_sel["youden_j"]),
        },
        "tpr_at_target_fpr": {
            "tpr_step": float(tprlow["tpr_step"]),
            "tpr_interp": float(tprlow["tpr_interp"]),
        },

        "metrics_at_lowfpr_threshold": metrics_at_threshold(valid_y, valid_scores, thr_low),
        "metrics_at_youden_threshold": metrics_at_threshold(valid_y, valid_scores, thr_you),

        "meta": {
            "steer_layer": int(steer_layer),
            "readout_layers": [int(x) for x in readout_layers],
            "cos_temp": float(cos_temp),
            "max_length": int(max_length),
            "last_k_tokens": int(last_k_tokens),
            "llm_field": llm_field,
            "human_field": human_field,
            "dtype": str(args.dtype),

            "steering_enabled": False,
            "train_data_path": str(args.train_data_path),
            "valid_data_path": str(args.valid_data_path),
            "centroid_source_path": str(centroid_path),
            "threshold_source_path": str(args.valid_data_path),
            "centroid_normalize_per_sample": bool(args.centroid_normalize_per_sample),
        },
    }

    print("VALID result:", json.dumps(valid_result, indent=2, ensure_ascii=False))

    # ---- TEST ----
    os.makedirs(args.out_dir, exist_ok=True)
    test_paths = [p.strip() for p in args.test_data_paths.split(",") if p.strip()]
    summary = {"valid_result": valid_result, "test_results": {}}

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    out_dir_llm = os.path.join(args.out_dir, ckpt_name)
    os.makedirs(out_dir_llm, exist_ok=True)

    for tp in test_paths:
        test_pairs = load_json_list(tp)
        test_texts, test_labels, test_ids, tstats = expand_pairs_to_samples(test_pairs, llm_field, human_field)
        print(f"[test data] {os.path.basename(tp)} {tstats} -> num_samples={len(test_texts)}")

        test_scores, test_y, test_id = score_texts_llr(
            model=model, tokenizer=tokenizer, device=device,
            readout_mgr=readout_mgr,
            texts=test_texts, labels=test_labels, ids=test_ids,
            centroids=centroids,
            batch_size=args.batch_size,
            cos_temp=cos_temp,
            max_length=max_length,
            last_k_tokens=last_k_tokens,
        )
        test_auc = roc_auc_score(test_y, test_scores)

        test_result = {
            "roc_auc": float(test_auc),

            "lowfpr_threshold": thr_low,
            "metrics_at_lowfpr_threshold": metrics_at_threshold(test_y, test_scores, thr_low),

            "youden_threshold": thr_you,
            "metrics_at_youden_threshold": metrics_at_threshold(test_y, test_scores, thr_you),

            "meta": {
                "steer_layer": int(steer_layer),
                "readout_layers": [int(x) for x in readout_layers],
                "cos_temp": float(cos_temp),
                "max_length": int(max_length),
                "last_k_tokens": int(last_k_tokens),
                "dtype": str(args.dtype),

                "steering_enabled": False,
                "train_data_path": str(args.train_data_path),
                "valid_data_path": str(args.valid_data_path),
                "centroid_source_path": str(centroid_path),
                "threshold_source_path": str(args.valid_data_path),
                "centroid_normalize_per_sample": bool(args.centroid_normalize_per_sample),
            },
        }

        summary["test_results"][os.path.basename(tp)] = test_result
        print(f"TEST {tp} result:", json.dumps(test_result, indent=2, ensure_ascii=False))

        score_dump = [
            {"score": float(s), "label": int(y), "id": str(iid)}
            for s, y, iid in zip(test_scores, test_y, test_id)
        ]
        out_path = os.path.join(out_dir_llm, f"{os.path.basename(tp)}_scores.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(score_dump, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir_llm, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    readout_mgr.remove()


if __name__ == "__main__":
    main()
