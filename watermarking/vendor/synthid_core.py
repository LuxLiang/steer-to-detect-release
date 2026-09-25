# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Extracted from the local research adaptation of MarkLLM on 2026-09-25.
# Only the scoring utilities, state, and logits processor are retained.
# See THIRD_PARTY_NOTICES.md for provenance and changes.
from __future__ import annotations
import torch
import numpy as np
from transformers import LogitsProcessor

class SynthIDUtils:
    """Utility class for SynthID algorithm, contains helper functions."""

    def __init__(self, config: SynthIDConfig, *args, **kwargs) -> None:
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.sampling_table_seed)

    def accumulate_hash(
        self,
        current_hash: torch.LongTensor,
        data: torch.LongTensor,
        multiplier: int = 6364136223846793005,
        increment: int = 1,
    ) -> torch.LongTensor:
        """Accumulate hash of data on current hash.

        Method uses adapted linear congruential generator with newlib/musl parameters.

        This function has following property -
        f(x, data[T]) = f(f(x, data[:T - 1]), data[T])

        This function expects current_hash.shape and data.shape[:-1] to
        match/broadcastable.

        Args:
            current_hash: (shape,)
            data: (shape, tensor_len)
            multiplier: (int) multiplier of linear congruential generator
            increment: (int) increment of linear congruential generator

        Returns:
            updated hash (shape,)
        """
        for i in range(data.shape[-1]):
            current_hash = torch.add(current_hash, data[..., i])
            current_hash = torch.mul(current_hash, multiplier)
            current_hash = torch.add(current_hash, increment)
        return current_hash

    def sum_probs_for_lower_g_values(self, g_values_at_depth, probs):
        """
        Vectorized implementation to compute sum of probabilities for tokens with g-values less than 
        each token's g-value.
        """
        batch_size, vocab_size = g_values_at_depth.shape
        result = torch.zeros_like(probs)

        for b in range(batch_size):
            # Expand g-values to compare each token with every other token
            # Shape: (vocab_size, 1) and (1, vocab_size)
            g_values_expanded = g_values_at_depth[b].unsqueeze(1)
            g_values_transposed = g_values_at_depth[b].unsqueeze(0)

            # Comparison mask: g_B < g_A, shape (vocab_size, vocab_size)
            # Each row a corresponds to token A, and contains True for each token B where g_B < g_A
            comparison_mask = g_values_transposed < g_values_expanded

            # Multiply the mask by the probabilities and sum along dimension 1
            # This gives us the sum of p(B) where g_B < g_A for each token A
            result[b] = torch.sum(comparison_mask * probs[b].unsqueeze(0), dim=1)

        return result

    def update_scores(
        self,
        g_distribution: str,
        scores: torch.FloatTensor,
        g_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Updates scores using the g values.

        We assume that the scores are in the log space.
        Args:
            scores: Scores (batch_size, vocab_size).
            g_values: G values (batch_size, vocab_size, depth).

        Returns:
            Updated scores (batch_size, vocab_size).
        """
        _, _, depth = g_values.shape
        device = scores.device

        probs = torch.softmax(scores, dim=1)

        if g_distribution == 'uniform':
            for i in range(depth):
                g_values_at_depth = g_values[:, :, i]
                prob_lower_g_values = self.sum_probs_for_lower_g_values(g_values_at_depth, probs)
                probs = probs * (probs + 2*prob_lower_g_values)
                probs = probs / torch.sum(probs, dim=1, keepdim=True) # normalize it in case of numerical instability
                # print('largest prob:', torch.max(probs), i)
        else:
            for i in range(depth):
                g_values_at_depth = g_values[:, :, i]
                g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
                probs = probs * (1 + g_values_at_depth - g_mass_at_depth)
                # print('largest prob:', torch.max(probs), i)

        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
        )
        return log_probs

    def update_scores_distortionary(
        self,
        scores: torch.FloatTensor,
        g_values: torch.FloatTensor,
        num_leaves: int,
    ) -> torch.FloatTensor:
        """Update scores using the g values for distortionary tournament watermarking.

        We assume that the scores are in the log space.
        Args:
            scores: Scores (batch_size, vocab_size).
            g_values: G values (batch_size, vocab_size, depth).
            num_leaves: Number of leaves per node in the tournament tree.

        Returns:
            Updated scores (batch_size, vocab_size).
        """
        _, _, depth = g_values.shape
        device = scores.device

        probs = torch.softmax(scores, dim=1)

        for i in range(depth):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
            coeff_not_in_g = (1 - g_mass_at_depth)**(num_leaves - 1)
            coeff_in_g = (1 - (1 - g_mass_at_depth)**(num_leaves)) / g_mass_at_depth
            coeffs = torch.where(
                torch.logical_and(g_values_at_depth == 1, probs > 0),
                coeff_in_g, coeff_not_in_g)
            probs = probs * coeffs

        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
        )
        return log_probs

    def mean_score_numpy(self, g_values, mask):
        """
        Args:
            g_values: shape [batch_size, seq_len, watermarking_depth]
            mask: shape [batch_size, seq_len]
        Returns:
            scores: shape [batch_size]
        """
        watermarking_depth = g_values.shape[-1]
        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
                watermarking_depth * num_unmasked
        )

    def weighted_mean_score_numpy(
        self,
        g_values: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray = None,
    ) -> np.ndarray:
        """Computes the Weighted Mean score.

        Args:
            g_values: g-values of shape [batch_size, seq_len, watermarking_depth]
            mask: A binary array shape [batch_size, seq_len] indicating which g-values
                should be used. g-values with mask value 0 are discarded
            weights: array of non-negative floats, shape [watermarking_depth]. The
                weights to be applied to the g-values. If not supplied, defaults to
                linearly decreasing weights from 10 to 1

        Returns:
            Weighted Mean scores, of shape [batch_size]. This is the mean of the
            unmasked g-values, re-weighted using weights.
        """
        watermarking_depth = g_values.shape[-1]

        if weights is None:
            weights = np.linspace(start=10, stop=1, num=watermarking_depth)

        # Normalise weights so they sum to watermarking_depth
        weights *= watermarking_depth / np.sum(weights)

        # Apply weights to g-values
        g_values = g_values * np.expand_dims(weights, axis=(0, 1))

        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
            watermarking_depth * num_unmasked
        )

class SynthIDState:
  """SynthID watermarking state."""

  def __init__(
      self,
      batch_size: int,
      ngram_len: int,
      context_history_size: int,
      device: torch.device,
  ):
    """Initializes the state.

    Args:
      batch_size: Batch size.
      ngram_len: Ngram length.
      context_history_size: Size of the tensor to keep track of seen contexts.
      device: Device to use.
    """
    self.context = torch.zeros(
        (batch_size, ngram_len - 1),
        dtype=torch.int64,
        device=device,
    )
    self.context_history = torch.zeros(
        (batch_size, context_history_size),
        dtype=torch.int64,
        device=device,
    )
    self.num_calls = 0

class SynthIDLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for SynthID algorithm, process logits to add watermark."""

    def __init__(self, config: SynthIDConfig, utils: SynthIDUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils
        self.state = None

        # Initialize parameters from config
        self.ngram_len = config.ngram_len
        self.keys = torch.tensor(config.keys, device=config.device)
        self.sampling_table_size = config.sampling_table_size
        self.context_history_size = config.context_history_size
        self.device = config.device

        # Initialize sampling table
        if self.config.g_distribution == 'uniform':

            self.sampling_table = torch.rand(
        size=(self.sampling_table_size,),
        generator=self.utils.rng,
        device=self.device,
            )
        else:
            self.sampling_table = torch.randint(
                low=0,
                high=2,
                size=(self.sampling_table_size,),
                generator=self.utils.rng,
                device=self.device,
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        scores_processed = scores / self.config.temperature
        batch_size, vocab_size = scores.shape
        final_scores = scores.clone()

        if self.config.top_k > 0:
            top_k_result = torch.topk(scores_processed, k=self.config.top_k, dim=1)
            scores_top_k = top_k_result.values
            # scores_top_k shape [batch_size, top_k]
            top_k_indices = top_k_result.indices
            # top_k_indices shape [batch_size, top_k]
        else:
            scores_top_k = scores_processed
            top_k_indices = torch.stack([
                torch.arange(vocab_size, device=self.device)
                for _ in range(batch_size)
            ])

        # Initialize state if needed
        if self.state is None:
            self.state = {
                "context": torch.zeros((batch_size, self.ngram_len - 1), dtype=torch.int64, device=self.device),
                "context_history": torch.zeros((batch_size, self.context_history_size), dtype=torch.int64, device=self.device),
                "num_calls": 0
            }

        # Update context with last input token
        if self.state["num_calls"] > 0:
            self.state["context"] = torch.cat((self.state["context"], input_ids[:, -1:]), dim=1)
            self.state["context"] = self.state["context"][:, 1:]

        self.state["num_calls"] += 1

        # Generate ngram keys and sample g values
        ngram_keys, hash_context = self._compute_keys(self.state["context"], top_k_indices)
        g_values = self.sample_g_values(ngram_keys)

        if self.config.watermark_mode == "non-distortionary":
            # Update scores based on g values
            updated_scores = self.utils.update_scores(self.config.g_distribution, scores_top_k, g_values)

            if self.config.repeat_mask == 1:
                # Check for repeated context
                hash_context = hash_context[:, None]
                is_repeated = (self.state["context_history"] == hash_context).any(dim=1, keepdim=True)
                # Update context history
                self.state["context_history"] = torch.cat((hash_context, self.state["context_history"]), dim=1)[:, :-1]
            else:
                is_repeated = torch.tensor([False] * batch_size, device=self.device).unsqueeze(1)

        elif self.config.watermark_mode == "distortionary":
            # Update scores based on g values
            updated_scores = self.utils.update_scores_distortionary(scores_top_k, g_values, self.config.num_leaves)

            # Disable context repetition check
            is_repeated = torch.tensor([False] * batch_size, device=self.device).unsqueeze(1)
        # Return original scores if context is repeated, otherwise return updated scores
        # return torch.where(is_repeated, scores, updated_scores)
        for b in range(batch_size):
          if not is_repeated[b]:
              if self.config.top_k > 0: # here final score is all zero except top k
                  # Update the top-k positions with watermarked scores
                  final_scores[b, top_k_indices[b]] = updated_scores[b]
                  # Create a mask for non-top-k tokens
                  mask = torch.ones(vocab_size, device=self.device, dtype=torch.bool)
                  mask[top_k_indices[b]] = False
                  # Set non-top-k tokens to -inf, which will become 0 after softmax
                  final_scores[b, mask] = -1e12
              else:
                  final_scores[b] = updated_scores[b]
        # print(torch.max(final_scores))
        return final_scores

    def _compute_keys(self, context: torch.LongTensor, top_k_indices: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Compute ngram keys for given context and possible next tokens."""
        batch_size = context.shape[0]

        # Initial hash of context
        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        hash_context = self.utils.accumulate_hash(hash_result, context)

        # Compute hash for each possible continuation
        hash_result = torch.vmap(self.utils.accumulate_hash, in_dims=(None, 1), out_dims=1)(
            hash_context, top_k_indices[:, :, None]
        )

        # Add watermarking keys
        keys = self.keys[None, None, :, None]
        hash_result = torch.vmap(self.utils.accumulate_hash, in_dims=(None, 2), out_dims=2)(
            hash_result, keys
        )

        return hash_result, hash_context

    def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor:
        """Sample g values from pre-computed sampling table."""
        ngram_keys = ngram_keys % self.sampling_table_size
        sampling_table = self.sampling_table.reshape((1, 1, self.sampling_table_size))
        return torch.take_along_dim(sampling_table, indices=ngram_keys, dim=2)

    def compute_g_values(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes g values for each ngram from the given sequence of tokens.

        Args:
            input_ids: Input token ids (batch_size, input_len).

        Returns:
            G values (batch_size, input_len - (ngram_len - 1), depth).
        """
        ngrams = input_ids.unfold(dimension=1, size=self.ngram_len, step=1)
        ngram_keys = self.compute_ngram_keys(ngrams)
        return self.sample_g_values(ngram_keys)

    def compute_ngram_keys(
      self,
        ngrams: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes random keys for each ngram and depth.

        Args:
            ngrams: Ngrams (batch_size, num_ngrams, ngram_len).

        Returns:
            ngram keys (batch_size, num_ngrams, depth).
        """
        if len(ngrams.shape) != 3:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len), but"
                f" is {ngrams.shape}"
        )
        if ngrams.shape[2] != self.ngram_len:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len),"
                f" where ngram_len is {self.ngram_len}, but is {ngrams.shape}"
            )
        batch_size, _, _ = ngrams.shape

        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        # hash_result shape [batch_size,]
        # ngrams shape [batch_size, num_ngrams, ngram_len]
        hash_result = torch.vmap(
            self.utils.accumulate_hash, in_dims=(None, 1), out_dims=1
        )(hash_result, ngrams)
        # hash_result shape [batch_size, num_ngrams]

        keys = self.keys[None, None, :, None]
        # hash_result shape [batch_size, num_ngrams]
        # keys shape [1, 1, depth, 1]
        hash_result = torch.vmap(
            self.utils.accumulate_hash, in_dims=(None, 2), out_dims=2
        )(hash_result, keys)
        # hash_result shape [batch_size, num_ngrams, depth]

        return hash_result

    def compute_eos_token_mask(
        self,
        input_ids: torch.LongTensor,
        eos_token_id: int,
    ) -> torch.LongTensor:
        """Computes repetitions mask.

        1 stands for ngrams that don't contain EOS tokens and vice versa.

        Args:
            input_ids: Input token ids (batch_size, input_len).
            eos_token_id: EOS token ID.

        Returns:
            EOS token mask (batch_size, input_len).
        """
        noneos_masks = []
        all_eos_equated = input_ids == eos_token_id
        for eos_equated in all_eos_equated:
            nonzero_idx = torch.nonzero(eos_equated)
            noneos_mask = torch.ones_like(eos_equated)
            if nonzero_idx.shape[0] != 0:
                noneos_mask[nonzero_idx[0][0]:] = 0
            noneos_masks.append(noneos_mask)
        return torch.stack(noneos_masks, dim=0)

    def compute_context_repetition_mask(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes repetition mask.

        0 and 1 stand for repeated and not repeated context n-1 grams respectively.

        Args:
            input_ids: Input token ids (batch_size, input_len).

        Returns:
            Repetitions mask (batch_size, input_len - (ngram_len - 1)).
        """
        batch_size, _ = input_ids.shape
        state = SynthIDState(
            batch_size=batch_size,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            device=self.device,
        )
        contexts = input_ids[:, :-1].unfold(
            dimension=1,
            size=self.ngram_len - 1,
            step=1,
        )
        _, num_contexts, _ = contexts.shape

        are_repeated_contexts = []
        for i in range(num_contexts):
            context = contexts[:, i, :]
            hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
            context_hash = self.utils.accumulate_hash(hash_result, context)[
                :, None
            ]
            is_repeated_context = (state.context_history == context_hash).any(
                dim=1,
                keepdim=True,
            )
            are_repeated_contexts.append(is_repeated_context)
            state.context_history = torch.concat(
                (context_hash, state.context_history),
                dim=1,
            )[:, :-1]
        are_repeated_contexts = torch.concat(are_repeated_contexts, dim=1)

        return torch.logical_not(are_repeated_contexts)
