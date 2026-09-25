"""Generate matched human, watermarked, and unwatermarked continuations."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, set_seed
from tqdm import tqdm

from .algorithms import make_processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', required=True, choices=['gumbel', 'kgw', 'synthid'])
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--human_field', default='human_text')
    parser.add_argument('--prompt_len', type=int, default=50)
    parser.add_argument('--gen_len', type=int, default=256)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--key', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=.25)
    parser.add_argument('--delta', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=['float32','bfloat16','float16'], default=None)
    parser.add_argument('--synthid_config', type=Path, default=Path(__file__).parent/'configs/synthid.json')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if min(args.prompt_len, args.gen_len, args.max_samples) < 1 or args.temperature <= 0 or args.top_k < 0:
        parser.error('Lengths, sample count, and temperature must be positive; top_k must be nonnegative.')
    if not 0 < args.gamma < 1:
        parser.error('gamma must be in (0,1).')
    if args.output.exists() and not args.overwrite:
        parser.error('Output exists. Choose another path or explicitly use --overwrite.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    dtype = args.dtype or ('bfloat16' if device.type == 'cuda' else 'float32')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        parser.error('Tokenizer requires a pad or EOS token.')
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=getattr(torch, dtype)).to(device).eval()
    size = model.get_output_embeddings().weight.shape[0]
    if args.top_k > size:
        parser.error(f'top_k exceeds model vocabulary ({size}); set --top_k <= {size}.')
    if args.algorithm == 'kgw' and not 0 < int(args.gamma * size) < size:
        parser.error('gamma produces an empty or full greenlist for this vocabulary.')
    metadata = dict(algorithm=args.algorithm, model_name_or_path=args.model_name_or_path,
                    vocab_size=size, rng_device_type=device.type, dtype=dtype,
                    temperature=args.temperature, top_k=args.top_k, gamma=args.gamma,
                    delta=args.delta, key=args.key if args.key is not None else (15485863 if args.algorithm=='kgw' else 42),
                    seed=args.seed, prompt_len=args.prompt_len, gen_len=args.gen_len,
                    human_field=args.human_field, torch_version=str(torch.__version__),
                    protocol='continuation-only-v1')
    if args.algorithm == 'synthid':
        metadata['synthid_config'] = json.loads(args.synthid_config.read_text())
    data = json.loads(args.data_path.read_text())
    if not isinstance(data,list):
        parser.error('Input must be a JSON array of human-text records.')
    records, skipped = [], 0
    for index, entry in enumerate(tqdm(data, desc=args.algorithm)):
        if len(records) >= args.max_samples:
            break
        text = entry.get(args.human_field) if isinstance(entry,dict) else None
        if not isinstance(text,str) or not text.strip():
            skipped += 1
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < args.prompt_len + args.gen_len:
            skipped += 1
            continue
        prompt = ids[:args.prompt_len]
        inputs = torch.tensor([prompt], device=device)
        common = dict(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                      max_new_tokens=args.gen_len, min_new_tokens=args.gen_len,
                      do_sample=True, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id, top_p=1.0)
        processor = make_processor(metadata, device)
        with torch.no_grad():
            # SynthID applies temperature/top-k internally; disable second warping.
            watermarked = model.generate(**common, temperature=1.0 if args.algorithm=='synthid' else args.temperature,
                        top_k=0 if args.algorithm=='synthid' else args.top_k,
                        logits_processor=LogitsProcessorList([processor]))[0].tolist()
            unwatermarked = model.generate(**common, temperature=args.temperature, top_k=args.top_k)[0].tolist()
        human = ids[:args.prompt_len+args.gen_len]
        decode = lambda seq: tokenizer.decode(seq[args.prompt_len:], skip_special_tokens=True)
        records.append(dict(id=entry.get('id', index), prompt_len=args.prompt_len,
                    prompt=tokenizer.decode(prompt, skip_special_tokens=True),
                    full_ids_human=human, full_ids_watermarked=watermarked,
                    full_ids_unwatermarked=unwatermarked, human_text_remaining=decode(human),
                    watermarked_llm_text=decode(watermarked), unwatermarked_llm_text=decode(unwatermarked)))
    if not records:
        parser.error('No records have enough tokens for prompt_len + gen_len.')
    metadata.update(num_records=len(records), skipped_records=skipped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema_version=1, metadata=metadata, records=records),indent=2,ensure_ascii=False)+'\n')
    print(f'Saved {len(records)} records to {args.output}')


if __name__ == '__main__':
    main()
