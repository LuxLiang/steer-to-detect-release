from .data import load_json_list, expand_pairs_to_samples
from .steering import find_transformer_blocks, SVSteering, SVHookManager
from .readout import ReadoutHookManager, last_ratio_tokens_mean_pool, aggregate_poststeer_lastlayers_lastratio_tokens
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Centroids
# -------------------------
@torch.no_grad()
def init_centroids_from_exemplars(
    model, tokenizer, device,
    readout_mgr: ReadoutHookManager,
    texts: List[str], labels: List[int],
    batch_size: int,
    max_length: int,
    last_token_ratio: float,
) -> torch.Tensor:
    model.eval()
    h0, h1 = [],[]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]

        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        _ = model(input_ids=input_ids, attention_mask=attn)

        h_list = readout_mgr.get_ordered()
        rep = aggregate_poststeer_lastlayers_lastratio_tokens(h_list, attn, last_token_ratio=last_token_ratio)
        rep = F.normalize(rep, p=2, dim=-1, eps=1e-6)

        for r, y in zip(rep, batch_labels):
            if int(y) == 1:
                h1.append(r.detach().cpu())
            else:
                h0.append(r.detach().cpu())

    c0 = torch.stack(h0, dim=0).mean(dim=0)
    c1 = torch.stack(h1, dim=0).mean(dim=0)
    centroids = torch.stack([c0, c1], dim=0).to(device=device, dtype=torch.float32)
    centroids = F.normalize(centroids, p=2, dim=-1, eps=1e-6)
    return centroids


@torch.no_grad()
def update_centroids_ema(
    centroids: torch.Tensor, reps: torch.Tensor, labels: torch.Tensor, ema: float
) -> torch.Tensor:
    for cls in (0, 1):
        mask = (labels == cls)
        if mask.any():
            mean_rep = reps[mask].mean(dim=0)
            centroids[cls] = ema * centroids[cls] + (1.0 - ema) * mean_rep
    centroids[:] = F.normalize(centroids, p=2, dim=-1, eps=1e-6)
    return centroids


# -------------------------
# Training
# -------------------------
def train_sv_detection(
    model, tokenizer, device,
    readout_mgr: ReadoutHookManager,
    train_texts: List[str], train_labels: List[int],
    sv: SVSteering,
    lr: float, epochs: int, batch_size: int,
    cos_temp: float, ema_decay: float,
    num_exemplars: int,
    out_dir: str,
    max_length: int,
    last_token_ratio: float,
    l2_reg: float = 0.0,
    grad_clip: float = 1.0,
    args=None,
    run_idx: Optional[int] = None  # Optional suffix for repeated sampling runs
):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    sv.train()

    idx0 =[i for i, y in enumerate(train_labels) if int(y) == 0]
    idx1 =[i for i, y in enumerate(train_labels) if int(y) == 1]
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Training samples missing a class; cannot train.")

    half = max(1, num_exemplars // 2)
    ex0 = random.sample(idx0, k=min(half, len(idx0)))
    ex1 = random.sample(idx1, k=min(half, len(idx1)))
    exemplars = ex0 + ex1

    ex_texts = [train_texts[i] for i in exemplars]
    ex_labels = [train_labels[i] for i in exemplars]

    centroids = init_centroids_from_exemplars(
        model=model, tokenizer=tokenizer, device=device,
        readout_mgr=readout_mgr, texts=ex_texts, labels=ex_labels,
        batch_size=batch_size, max_length=max_length,
        last_token_ratio=last_token_ratio,
    ).float()

    optimizer = torch.optim.AdamW(sv.parameters(), lr=lr)
    cos_temp = float(max(cos_temp, 1e-4))
    ema_decay = float(min(max(ema_decay, 0.0), 0.9999))

    os.makedirs(out_dir, exist_ok=True)
    indices = list(range(len(train_texts)))
    num_batches = (len(indices) + batch_size - 1) // batch_size

    for ep in range(1, epochs + 1):
        random.shuffle(indices)
        running, total = 0.0, 0

        pbar = tqdm(
            range(0, len(indices), batch_size),
            total=num_batches,
            desc=f"Epoch {ep}/{epochs}",
            dynamic_ncols=True,
        )

        for start in pbar:
            batch_idx = indices[start:start + batch_size]
            batch_texts = [train_texts[i] for i in batch_idx]
            batch_y = torch.tensor(
                [int(train_labels[i]) for i in batch_idx],
                device=device, dtype=torch.long,
            )

            enc = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _ = model(input_ids=input_ids, attention_mask=attn)

            h_list = readout_mgr.get_ordered()
            rep = aggregate_poststeer_lastlayers_lastratio_tokens(
                h_list=h_list, attn=attn, last_token_ratio=last_token_ratio
            )
            rep = F.normalize(rep, p=2, dim=-1, eps=1e-6)

            c = F.normalize(centroids, p=2, dim=-1, eps=1e-6)
            logits = (rep @ c.T) / cos_temp
            loss = F.cross_entropy(logits, batch_y)

            if l2_reg and l2_reg > 0:
                loss = loss + float(l2_reg) * sv.v.pow(2).sum()

            if not torch.isfinite(loss):
                pbar.set_postfix({"loss": "NaN/Inf (skip)"})
                continue

            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sv.parameters(), max_norm=float(grad_clip))

            grad_ok = True
            for p in sv.parameters():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    grad_ok = False
                    break
            if not grad_ok:
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({"loss": "grad NaN/Inf (skip)"})
                continue

            optimizer.step()

            with torch.no_grad():
                centroids = update_centroids_ema(centroids, rep.detach(), batch_y, ema=ema_decay)

            running += float(loss.item()) * len(batch_idx)
            total += len(batch_idx)
            avg = running / max(1, total)
            pbar.set_postfix({"loss": f"{avg:.4f}"})

        print(f"[train] epoch={ep}/{epochs} avg_loss={running / max(1, total):.6f}")

    ckpt = {
        "sv_state_dict": sv.state_dict(),
        "centroids": centroids.detach().cpu().numpy(),
        "cos_temp": float(cos_temp),
        "ema_decay": float(ema_decay),
        "meta": {
            "num_train_samples": int(len(train_texts)),
            "max_length": int(max_length),
            "l2_reg": float(l2_reg),
            "grad_clip": float(grad_clip),
            "pooling": "post_steer_last_n_layers + last_ratio_tokens_mean + layer_mean",
            "last_token_ratio": float(last_token_ratio),
        },
    }

    # Derive checkpoint filename from the input dataset
    if "_llm_type_" in args.train_data_path:
        base_name = args.train_data_path.rsplit("_llm_type_", 1)[1].rsplit(".json", 1)[0]
    else:
        base_name = os.path.basename(args.train_data_path).replace(".json", "")

    if run_idx is not None:
        out_path = os.path.join(args.out_dir, f"s2d_ckpt_{base_name}_run_{run_idx}.pt")
    else:
        out_path = os.path.join(args.out_dir, f"s2d_ckpt_{base_name}.pt")

    torch.save(ckpt, out_path)
    print(f"Saved checkpoint: {out_path}")
    return out_path


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # ======== Bootstrap (optional) ========
    parser.add_argument("--bootstrap_samples", type=int, default=-1)
    parser.add_argument("--bootstrap_runs", type=int, default=5)
    # =======================================

    # Dataset fields
    parser.add_argument("--max_train_pairs", type=int, default=512)
    parser.add_argument("--llm_field", type=str, default="direct_prompt")
    parser.add_argument("--human_field", type=str, default="human_text")

    # SV intervention controls
    parser.add_argument("--steer_layer", type=int, default=11)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--steer_pos", type=str, default="residual", choices=["residual", "attention", "mlp"])

    # Readout controls
    parser.add_argument("--readout_last_n_layers", type=int, default=8)
    parser.add_argument("--last_token_ratio", type=float, default=0.25)
    parser.add_argument(
        "--fixed_final_readout",
        action="store_true",
        help=(
            "Read the model's final N blocks regardless of steer_layer. "
            "Default behavior remains the original post-steer-only readout."
        ),
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cos_temp", type=float, default=0.4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--num_exemplars", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--l2_reg", type=float, default=0.0)

    # Practical controls
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--init_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2025)

    # Precision control
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = getattr(torch, args.model_dtype)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=model_dtype,
    device_map="auto",
    )
    #model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype).to(device)
    device = model.device
    print("model_name_or_path", args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = find_transformer_blocks(model)
    num_layers = len(blocks)

    if not (0 <= args.steer_layer < num_layers):
        raise ValueError(f"--steer_layer must be within [0, {num_layers-1}], got {args.steer_layer}")

    n = int(max(1, args.readout_last_n_layers))
    if args.fixed_final_readout:
        readout_layers = list(range(max(0, num_layers - n), num_layers))
    else:
        post_layers = list(range(args.steer_layer, num_layers))
        if len(post_layers) == 0:
            raise ValueError("No post-steer layers found; check steer_layer.")
        readout_layers = sorted(post_layers[-n:])

    hidden_size = int(getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd"))

    # Load whole dataset
    all_pairs = load_json_list(args.train_data_path)
    runs = args.bootstrap_runs if args.bootstrap_samples > 0 else 1

    for run_idx in range(runs):
        if args.bootstrap_samples > 0:
            print(f"\n=======================================================")
            print(f"=== Bootstrap Run {run_idx + 1}/{runs} (Samples: {args.bootstrap_samples}) ===")
            print(f"=======================================================")
            current_pairs = random.sample(all_pairs, min(args.bootstrap_samples, len(all_pairs)))
            current_run_idx = run_idx
            current_max_pairs = len(current_pairs)
        else:
            current_pairs = all_pairs
            current_run_idx = None
            current_max_pairs = args.max_train_pairs

        train_texts, train_labels, stats = expand_pairs_to_samples(
            current_pairs,
            llm_field=args.llm_field,
            human_field=args.human_field,
            max_pairs=current_max_pairs,
        )
        print(f"[data] {stats} -> num_samples={len(train_texts)} (LGT/HWT)")

        sv = SVSteering(hidden_size=hidden_size, init_scale=args.init_scale).to(device)

        sv_hook = SVHookManager(
            blocks=blocks,
            sv_module=sv,
            lam=args.lam,
            steer_layer=args.steer_layer,
            steer_pos=args.steer_pos,
        )
        sv_hook.register()

        readout_mgr = ReadoutHookManager(blocks=blocks, readout_layers=readout_layers)
        readout_mgr.register()

        out_path = train_sv_detection(
            model=model,
            tokenizer=tokenizer,
            device=device,
            readout_mgr=readout_mgr,
            train_texts=train_texts,
            train_labels=train_labels,
            sv=sv,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cos_temp=args.cos_temp,
            ema_decay=args.ema_decay,
            num_exemplars=args.num_exemplars,
            out_dir=args.out_dir,
            max_length=args.max_length,
            last_token_ratio=args.last_token_ratio,
            l2_reg=args.l2_reg,
            grad_clip=args.grad_clip,
            args=args,
            run_idx=current_run_idx,
        )

        # Update checkpoint meta for reproducibility
        ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
        ckpt["steer_layer"] = int(args.steer_layer)
        ckpt["steer_pos"] = str(args.steer_pos)
        ckpt["lam"] = float(args.lam)
        ckpt["readout_layers"] = [int(x) for x in readout_layers]
        ckpt["readout_last_n_layers"] = int(n)
        ckpt["fixed_final_readout"] = bool(args.fixed_final_readout)
        ckpt["last_token_ratio"] = float(args.last_token_ratio)
        ckpt["llm_field"] = str(args.llm_field)
        ckpt["human_field"] = str(args.human_field)
        ckpt["max_length"] = int(args.max_length)
        ckpt["model_dtype"] = str(args.model_dtype)
        ckpt["model_name_or_path"] = str(args.model_name_or_path)
        ckpt["training_args"] = vars(args).copy()
        if args.bootstrap_samples > 0:
            ckpt["bootstrap_run_id"] = current_run_idx
        torch.save(ckpt, out_path)
        print(f"Updated checkpoint meta: {out_path}\n")

        readout_mgr.remove()
        sv_hook.remove()

if __name__ == "__main__":
    main()
