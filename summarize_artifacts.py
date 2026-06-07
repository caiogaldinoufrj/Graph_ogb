import argparse
import csv
from datetime import datetime
from pathlib import Path


def collect_artifacts(root):
    artifacts = []
    root_path = Path(root)
    for path in sorted(root_path.rglob('*')):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root_path)
        suffix = path.suffix.lower()
        kind = None
        if path.name.startswith('log_') or suffix == '.log' or suffix == '.txt':
            kind = 'log'
        elif suffix in {'.pt', '.pth'}:
            kind = 'checkpoint'
        elif suffix in {'.npy', '.npz', '.pkl'}:
            kind = 'data'
        else:
            continue

        info = ''
        if kind == 'log':
            try:
                with path.open('r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                info = first_line
            except Exception:
                info = ''

        artifacts.append({
            'root': str(root_path),
            'relative_path': str(rel_path),
            'kind': kind,
            'size_bytes': path.stat().st_size,
            'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=' '),
            'info': info,
        })
    return artifacts


def write_csv(rows, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['root', 'relative_path', 'kind', 'size_bytes', 'modified_time', 'info'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Summarize logs, checkpoints and data artifacts in a project tree.')
    parser.add_argument('--root', type=str, default='.', help='Root directory to scan')
    parser.add_argument('--output_csv', type=str, default='artifact_summary.csv', help='CSV file to write summary')
    args = parser.parse_args()

    artifacts = collect_artifacts(args.root)
    write_csv(artifacts, args.output_csv)
    print(f'Saved {len(artifacts)} artifacts to {args.output_csv}')


if __name__ == '__main__':
    main()
