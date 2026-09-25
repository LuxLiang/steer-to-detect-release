import json
from typing import List, Tuple, Dict, Any, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Expand N paired items into 2N samples:
      label=1: item[llm_field]   (LGT)
      label=0: item[human_field] (HWT)
    """
    texts, labels = [], []
    skipped = 0
    used = 0

    subset = pairs[:max_pairs] if max_pairs is not None else pairs
    for item in subset:
        llm_text = item.get(llm_field, None)
        human_text = item.get(human_field, None)
        if not (isinstance(llm_text, str) and llm_text.strip() and isinstance(human_text, str) and human_text.strip()):
            skipped += 1
            continue
        texts.append(llm_text)
        labels.append(1)
        texts.append(human_text)
        labels.append(0)
        used += 1

    stats = {"used_pairs": used, "skipped_pairs": skipped, "total_pairs_seen": len(subset)}
    return texts, labels, stats

def expand_pairs_for_evaluation(
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
        "total_samples": len(texts),
    }
    return texts, labels, ids, stats
