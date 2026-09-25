"""Create a random tiny Llama and tokenizer locally for an offline smoke test."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def create_model(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2025)
    words = '[PAD] [UNK] [EOS] I was thinking about the today It reminded me of a small surprise This example discusses Several relevant observations can be considered garden train bread rain river music forest coffee ocean books mountain window .'.split()
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]', padding_side='left')
    tokenizer.save_pretrained(output)
    config = LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, pad_token_id=0, eos_token_id=2)
    LlamaForCausalLM(config).save_pretrained(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    create_model(parser.parse_args().output)
