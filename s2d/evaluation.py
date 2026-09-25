from .data import load_json_list, expand_pairs_for_evaluation as expand_pairs_to_samples
from .steering import find_transformer_blocks, SVSteering, EvaluationSVHookManager as SVHookManager
from .readout import EvaluationReadoutHookManager as ReadoutHookManager, evaluation_aggregate_poststeer_lastlayers_lastratio_tokens as aggregate_poststeer_lastlayers_lastratio_tokens
from .metrics import metrics_at_threshold, select_threshold_youden, select_threshold_np_quantile, tpr_at_fpr, multi_tpr_at_fprs, format_fpr_key
#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
# Scoring
# -------------------------
@torch.no_grad()
def score_texts_llr(
    model,
    tokenizer,
    device,
    readout_mgr,
    texts: List[str],
    labels: List[int],
    ids: Sequence,
    centroids: torch.Tensor,
    batch_size: int,
    cos_temp: float,
    max_length: int,
    last_token_ratio: float,
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
    centroids = F.normalize(centroids, p=2, dim=-1, eps=1e-6)
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
        rep = aggregate_poststeer_lastlayers_lastratio_tokens(
            h_list=h_list,
            attn=attn,
            last_token_ratio=last_token_ratio,
            padding_side=tokenizer.padding_side,
        )
        rep = F.normalize(rep, p=2, dim=-1, eps=1e-6)

        logits = (rep @ centroids.T) / T
        llr = logits[:, 1] - logits[:, 0]

        all_scores.append(llr.detach().cpu().numpy().astype(np.float32))
        all_y.append(np.asarray(batch_y, dtype=np.int32))
        all_ids.append(np.asarray(batch_ids))

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_ids, axis=0),
    )


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--valid_llm_field", default="direct_prompt")
    parser.add_argument("--valid_human_field", default="human_text")
    parser.add_argument(
        "--test_data_paths",
        type=str,
        required=True,
        help="Comma-separated list of test json paths",
    )

    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--llm_field", type=str, default=None)
    parser.add_argument("--human_field", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--steer_pos",
        type=str,
        default=None,
        choices=["residual", "attention", "mlp"],
        help="Override injection position. If not set, use ckpt['steer_pos'] if present, else residual.",
    )

    parser.add_argument("--readout_last_n_layers", type=int, default=None)
    parser.add_argument("--last_token_ratio", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--target_fprs",
        type=float,
        nargs="+",
        default=[1e-2, 5e-3, 0.0001],
        help="List of target FPRs, e.g. --target_fprs 0.01 0.005",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out_dir", type=str, default="results_s2d/diff_test/")
    args = parser.parse_args()

    target_fprs = sorted(set(float(x) for x in args.target_fprs), reverse=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    steer_layer = int(ckpt.get("steer_layer", 0))
    cos_temp = float(ckpt.get("cos_temp", 1.0))
    steer_pos = args.steer_pos if args.steer_pos is not None else str(ckpt.get("steer_pos", "residual"))

    centroids_raw = ckpt.get("centroids", None)
    if centroids_raw is None:
        raise ValueError("ckpt missing 'centroids'")
    centroids = torch.tensor(centroids_raw, dtype=torch.float32)

    lam = float(args.lam) if args.lam is not None else float(ckpt.get("lam", 2.0))
    llm_field = args.llm_field if args.llm_field is not None else str(ckpt.get("llm_field", "direct_prompt"))
    human_field = args.human_field if args.human_field is not None else str(ckpt.get("human_field", "human_text"))
    max_length = int(args.max_length) if args.max_length is not None else int(ckpt.get("max_length", 2048))

    last_token_ratio = args.last_token_ratio
    if last_token_ratio is None:
        _meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}
        last_token_ratio = ckpt.get("last_token_ratio", _meta.get("last_token_ratio", 0.25))
    last_token_ratio = float(last_token_ratio)
    if not (0.0 < last_token_ratio <= 1.0):
        raise ValueError(f"last_token_ratio must be in (0,1], got {last_token_ratio}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = find_transformer_blocks(model)
    num_layers = len(blocks)

    if not (0 <= steer_layer < num_layers):
        raise ValueError(f"steer_layer={steer_layer} out of range; num_layers={num_layers}")

    if "readout_layers" in ckpt and isinstance(ckpt["readout_layers"], (list, tuple)) and len(ckpt["readout_layers"]) > 0:
        readout_layers = [int(x) for x in ckpt["readout_layers"]]
    else:
        post_layers = list(range(steer_layer, num_layers))
        n = args.readout_last_n_layers
        if n is None:
            n = int(ckpt.get("readout_last_n_layers", 8))
        n = int(max(1, n))
        readout_layers = sorted(post_layers[-n:])

    if args.readout_last_n_layers is not None:
        post_layers = list(range(steer_layer, num_layers))
        n = int(max(1, args.readout_last_n_layers))
        readout_layers = sorted(post_layers[-n:])

    hidden_size = int(getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd"))
    sv = SVSteering(hidden_size=hidden_size).to(device)
    sv.load_state_dict(ckpt["sv_state_dict"])

    sv_hook = SVHookManager(
        blocks=blocks,
        sv_module=sv,
        lam=lam,
        steer_layer=steer_layer,
        steer_pos=steer_pos,
    )
    sv_hook.register()

    readout_mgr = ReadoutHookManager(blocks=blocks, readout_layers=readout_layers)
    readout_mgr.register()

    # ---- VALID ----
    valid_pairs = load_json_list(args.valid_data_path)
    valid_texts, valid_labels, valid_ids, vstats = expand_pairs_to_samples(
        valid_pairs,
        args.valid_llm_field,
        args.valid_human_field,
    )
    print(f"[valid data] {vstats} -> num_samples={len(valid_texts)}")

    valid_scores, valid_y, valid_id = score_texts_llr(
        model=model,
        tokenizer=tokenizer,
        device=device,
        readout_mgr=readout_mgr,
        texts=valid_texts,
        labels=valid_labels,
        ids=valid_ids,
        centroids=centroids,
        batch_size=args.batch_size,
        cos_temp=cos_temp,
        max_length=max_length,
        last_token_ratio=last_token_ratio,
    )
    valid_auc = roc_auc_score(valid_y, valid_scores)

    low_sel_dict = {}
    thr_low_dict = {}
    for alpha in target_fprs:
        sel = select_threshold_np_quantile(valid_y, valid_scores, alpha)
        low_sel_dict[alpha] = sel
        thr_low_dict[alpha] = float(sel["threshold"])

    you_sel = select_threshold_youden(valid_y, valid_scores)
    thr_you = float(you_sel["threshold"])

    valid_tpr_interp_dict = multi_tpr_at_fprs(valid_y, valid_scores, target_fprs)

    valid_result = {
        "roc_auc": float(valid_auc),
        "target_fprs": [float(x) for x in target_fprs],

        "tpr_at_target_fpr": {
            key: {
                "tpr_interp": float(val["tpr_interp"]),
            }
            for key, val in valid_tpr_interp_dict.items()
        },

        "lowfpr_select": {
            format_fpr_key(alpha): {
                "threshold": float(low_sel_dict[alpha]["threshold"]),
                "achieved_fpr": float(low_sel_dict[alpha]["fpr"]),
                "tpr_at_selected": float(low_sel_dict[alpha]["tpr"]),
                "n_neg": int(low_sel_dict[alpha]["n_neg"]),
            }
            for alpha in target_fprs
        },

        "metrics_at_lowfpr_threshold": {
            format_fpr_key(alpha): metrics_at_threshold(valid_y, valid_scores, thr_low_dict[alpha])
            for alpha in target_fprs
        },

        "youden_select": {
            "threshold": thr_you,
            "fpr": float(you_sel["fpr"]),
            "tpr": float(you_sel["tpr"]),
            "youden_j": float(you_sel["youden_j"]),
        },

        "metrics_at_youden_threshold": metrics_at_threshold(valid_y, valid_scores, thr_you),

        "meta": {
            "steer_layer": int(steer_layer),
            "steer_pos": str(steer_pos),
            "readout_layers": [int(x) for x in readout_layers],
            "lam": float(lam),
            "cos_temp": float(cos_temp),
            "max_length": int(max_length),
            "last_token_ratio": float(last_token_ratio),
            "llm_field": llm_field,
            "human_field": human_field,
            "dtype": str(args.dtype),
        },
    }

    print("VALID result:", json.dumps(valid_result, indent=2))

    # ---- TEST ----
    os.makedirs(args.out_dir, exist_ok=True)
    test_paths = [p.strip() for p in args.test_data_paths.split(",") if p.strip()]
    summary = {"valid_result": valid_result, "test_results": {}}

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    out_dir_llm = os.path.join(args.out_dir, ckpt_name)
    os.makedirs(out_dir_llm, exist_ok=True)

    for tp in test_paths:
        test_pairs = load_json_list(tp)
        test_texts, test_labels, test_ids, tstats = expand_pairs_to_samples(
            test_pairs,
            llm_field,
            human_field,
        )
        print(f"[test data] {os.path.basename(tp)} {tstats} -> num_samples={len(test_texts)}")

        test_scores, test_y, test_id = score_texts_llr(
            model=model,
            tokenizer=tokenizer,
            device=device,
            readout_mgr=readout_mgr,
            texts=test_texts,
            labels=test_labels,
            ids=test_ids,
            centroids=centroids,
            batch_size=args.batch_size,
            cos_temp=cos_temp,
            max_length=max_length,
            last_token_ratio=last_token_ratio,
        )
        test_auc = roc_auc_score(test_y, test_scores)
        test_tpr_interp_dict = multi_tpr_at_fprs(test_y, test_scores, target_fprs)

        test_result = {
            "roc_auc": float(test_auc),

            "tpr_at_target_fpr": {
                key: {
                    "tpr_interp": float(val["tpr_interp"]),
                }
                for key, val in test_tpr_interp_dict.items()
            },

            "lowfpr_threshold": {
                format_fpr_key(alpha): float(thr_low_dict[alpha])
                for alpha in target_fprs
            },

            "metrics_at_lowfpr_threshold": {
                format_fpr_key(alpha): metrics_at_threshold(test_y, test_scores, thr_low_dict[alpha])
                for alpha in target_fprs
            },

            "youden_threshold": thr_you,
            "metrics_at_youden_threshold": metrics_at_threshold(test_y, test_scores, thr_you),

            "meta": {
                "steer_layer": int(steer_layer),
                "steer_pos": str(steer_pos),
                "readout_layers": [int(x) for x in readout_layers],
                "lam": float(lam),
                "cos_temp": float(cos_temp),
                "max_length": int(max_length),
                "last_token_ratio": float(last_token_ratio),
                "dtype": str(args.dtype),
            },
        }

        summary["test_results"][os.path.basename(tp)] = test_result
        print(f"TEST {tp} result:", json.dumps(test_result, indent=2))

        score_dump = [
            {"score": float(s), "label": int(y), "id": str(sample_id)}
            for s, y, sample_id in zip(test_scores.tolist(), test_y.tolist(), test_id.tolist())
        ]
        out_path = os.path.join(out_dir_llm, f"{os.path.basename(tp)}_scores.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(score_dump, f, indent=2)

    with open(os.path.join(out_dir_llm, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    readout_mgr.remove()
    sv_hook.remove()


if __name__ == "__main__":
    main()
