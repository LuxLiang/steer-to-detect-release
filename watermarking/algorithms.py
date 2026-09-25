"""Research watermark processors and token-ID detectors.

Gumbel sampling follows the existing research variant (noise + multinomial
sampling), not an exact Gumbel argmax sampler. SynthID uses the local uniform-g
MarkLLM adaptation. RNG device type must match between generation and detection.
"""
import hashlib
from types import SimpleNamespace

import numpy as np
import torch
from transformers import LogitsProcessor

from .vendor.synthid_core import SynthIDUtils, SynthIDLogitsProcessor


def gumbel_uniform(previous_token, key, vocab_size, device):
    seed = int(hashlib.sha256(f'{previous_token}_{key}'.encode()).hexdigest(), 16) % (2**32)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(vocab_size, generator=generator, device=device)


class GumbelProcessor(LogitsProcessor):
    def __init__(self, key):
        self.key = key

    def __call__(self, input_ids, scores):
        # Each row is independently keyed; the original driver used batch size 1.
        result = scores.clone()
        for row in range(len(input_ids)):
            u = gumbel_uniform(int(input_ids[row, -1]), self.key, scores.shape[-1], scores.device)
            result[row] += -torch.log(-torch.log(u + 1e-10) + 1e-10)
        return result


def greenlist(previous_token, vocab_size, gamma, key, device):
    seed = (key * previous_token) % (2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randperm(vocab_size, generator=generator, device=device)[:int(gamma * vocab_size)]


class KGWProcessor(LogitsProcessor):
    def __init__(self, gamma, delta, key):
        self.gamma, self.delta, self.key = gamma, delta, key

    def __call__(self, input_ids, scores):
        result = scores.clone()
        for row in range(len(input_ids)):
            ids = greenlist(int(input_ids[row, -1]), scores.shape[-1], self.gamma, self.key, scores.device)
            result[row, ids] += self.delta
        return result


def synthid_processor(config, device, temperature=1.0, top_k=50):
    # Fresh processor/state per independent sequence. No model/tokenizer needed.
    config = dict(config, device=torch.device(device), temperature=temperature, top_k=top_k)
    obj = SimpleNamespace(**config)
    return SynthIDLogitsProcessor(obj, SynthIDUtils(obj))


def make_processor(metadata, device):
    name = metadata['algorithm']
    if name == 'gumbel':
        return GumbelProcessor(metadata['key'])
    if name == 'kgw':
        return KGWProcessor(metadata['gamma'], metadata['delta'], metadata['key'])
    if name == 'synthid':
        return synthid_processor(metadata['synthid_config'], device, metadata['temperature'], metadata['top_k'])
    raise ValueError(f'Unknown watermark algorithm: {name}')


class TokenDetector:
    def __init__(self, metadata, device):
        self.meta, self.device = metadata, torch.device(device)
        if self.device.type != metadata['rng_device_type']:
            raise ValueError('Detection must use the generation RNG device type (CPU/CUDA), which is recorded in the dataset.')
        self.processor = make_processor(metadata, self.device)

    @torch.no_grad()
    def score(self, full_ids, prompt_len, length):
        """Score exactly length continuation tokens; exclude prompt from statistic."""
        if prompt_len < 1 or len(full_ids) < prompt_len + length:
            raise ValueError('Insufficient prompt/continuation tokens.')
        if any(t < 0 or t >= self.meta['vocab_size'] for t in full_ids):
            raise ValueError('Token ID outside the recorded model vocabulary.')
        name, size = self.meta['algorithm'], self.meta['vocab_size']
        if name == 'synthid':
            # Source processor starts with zero context; omit the first n-1
            # continuation tokens so every scored ngram uses generated context.
            ids = torch.tensor([full_ids[prompt_len:prompt_len + length]], device=self.device)
            if length < self.meta['synthid_config']['ngram_len']:
                raise ValueError('SynthID window must contain at least ngram_len continuation tokens.')
            return float(self.processor.compute_g_values(ids).mean().item())
        values = []
        for i in range(prompt_len, prompt_len + length):
            previous, token = full_ids[i-1], full_ids[i]
            if name == 'gumbel':
                values.append(float(gumbel_uniform(previous, self.meta['key'], size, self.device)[token]))
            else:
                ids = greenlist(previous, size, self.meta['gamma'], self.meta['key'], self.device)
                values.append(int(bool((ids == token).any())))
        if name == 'gumbel':
            return float(np.sum(-np.log(1 - np.array(values) + 1e-10)))
        gamma = self.meta['gamma']
        return float((sum(values) - gamma * length) / np.sqrt(length * gamma * (1-gamma)))
