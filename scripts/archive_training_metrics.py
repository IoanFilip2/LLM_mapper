"""Preserve validation checkpoint metadata from an already-running training job.

This read-only observer never loads weights or changes the training process.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def archive_once(output):
    output = Path(output)
    archive = output / 'metrics_history'
    archive.mkdir(exist_ok=True)
    for checkpoint in sorted(output.glob('checkpoint-*')):
        if not checkpoint.name.removeprefix('checkpoint-').isdigit():
            continue
        source = checkpoint / 'checkpoint_manifest.json'
        target = archive / (checkpoint.name + '.json')
        if target.exists():
            continue
        try:
            record = json.loads(source.read_text())
        except FileNotFoundError:
            continue
        if record.get('status') != 'complete':
            continue
        record['archived_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(target)
    manifest = output / 'training_manifest.json'
    return json.loads(manifest.read_text()).get('status') if manifest.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        status = archive_once(args.output)
        if not args.watch or status in {'complete', 'budget_exhausted', 'failed'}:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
