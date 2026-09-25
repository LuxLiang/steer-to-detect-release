"""Evaluate native watermark detectors using exact stored continuation token IDs."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .algorithms import TokenDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[64,128,256])
    parser.add_argument('--target_fprs', type=float, nargs='+', default=[.01,.0001])
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if any(n < 1 for n in args.lengths) or any(not 0 < f < 1 for f in args.target_fprs):
        parser.error('Lengths must be positive and target FPRs must be in (0,1).')
    dataset = json.loads(args.dataset.read_text())
    if not isinstance(dataset,dict) or dataset.get('schema_version') != 1:
        parser.error('Expected a version-1 dataset produced by watermarking.generate.')
    meta = dataset['metadata']
    if meta['algorithm']=='synthid' and min(args.lengths)<meta['synthid_config']['ngram_len']:
        parser.error('SynthID length must be at least ngram_len.')
    detector = TokenDetector(meta, args.device or meta['rng_device_type'])
    results = {}
    for length in sorted(set(args.lengths)):
        comparisons = {}
        for reference in ['human','unwatermarked']:
            pos, neg, details = [], [], []
            for record in dataset['records']:
                start = record['prompt_len']
                wm, other = record['full_ids_watermarked'], record['full_ids_'+reference]
                if min(len(wm),len(other)) < start+length:
                    continue
                p, n = detector.score(wm,start,length), detector.score(other,start,length)
                if not np.isfinite([p,n]).all():
                    raise ValueError('Non-finite watermark score.')
                pos.append(p); neg.append(n)
                details.append(dict(id=record['id'],watermarked_score=p,reference_score=n))
            if not pos:
                comparisons[reference] = dict(num_pairs=0, skipped_pairs=len(dataset['records']),roc_auc=None,tpr_interp=None,scores=[])
                continue
            labels, scores = [1]*len(pos)+[0]*len(neg), pos+neg
            fpr,tpr,_ = roc_curve(labels,scores,drop_intermediate=False)
            comparisons[reference] = dict(num_pairs=len(pos),skipped_pairs=len(dataset['records'])-len(pos),
                        roc_auc=float(roc_auc_score(labels,scores)),
                        tpr_interp={str(f):float(np.interp(f,fpr,tpr)) for f in args.target_fprs},scores=details)
        results[str(length)] = comparisons
    output = dict(metadata=meta,protocol='continuation-only-v1',
                  metric_note='Empirical test ROC interpolation; no validation-calibrated FPR guarantee.',results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(f'Saved native detector scores and metrics to {args.output}')


if __name__ == '__main__':
    main()
