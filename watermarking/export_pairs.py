"""Export watermark continuations to the paired JSON format accepted by S2D."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--llm_source', choices=['watermarked','unwatermarked'], default='watermarked')
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    if not isinstance(dataset,dict) or dataset.get('schema_version') != 1:
        parser.error('Expected a version-1 watermark dataset.')
    pairs=[]
    for record in dataset['records']:
        h, m = record['human_text_remaining'], record[args.llm_source+'_llm_text']
        if isinstance(h,str) and h.strip() and isinstance(m,str) and m.strip():
            pairs.append(dict(id=record['id'],human_text=h,direct_prompt=m))
    if not pairs:
        parser.error('No nonempty continuation pairs to export.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(pairs,indent=2,ensure_ascii=False)+'\n')
    print(f'Exported {len(pairs)} pairs; skipped {len(dataset["records"])-len(pairs)} empty-text records.')


if __name__=='__main__':
    main()
