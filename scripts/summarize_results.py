"""Export held-out results and achieved operating points from S2D summaries."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.rglob('summary.json')):
        summary = json.loads(path.read_text())
        for dataset, result in summary.get('test_results', {}).items():
            for target, operating in result.get('metrics_at_lowfpr_threshold', {}).items():
                rows.append(dict(summary=str(path), dataset=dataset, roc_auc=result['roc_auc'], target_fpr=target, achieved_fpr=operating['fpr'], tpr=operating['tpr']))
    if not rows:
        parser.error('No main-S2D test operating metrics found under the supplied directory.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
