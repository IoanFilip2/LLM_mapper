"""Watch an authorized experiment and resume budget stops until completion.

Never restart a failed run. Training's writer lock prevents concurrent writers.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def action_for(manifest):
    status = manifest.get('status')
    if status == 'complete':
        return 'complete'
    if status == 'failed':
        return 'failed'
    if status == 'budget_exhausted':
        if not manifest.get('last_checkpoint'):
            raise ValueError('Budget stop has no checkpoint')
        return 'resume'
    return 'wait'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    training = root / 'training'
    def record(status, **fields):
        payload = dict(status=status, updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = root / 'budget_resume_status.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(root / 'budget_resume_status.json')
    with (root / 'budget_resume.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record('watching')
        last_resumed_step = -1
        while True:
            manifest = json.loads((training / 'training_manifest.json').read_text())
            action = action_for(manifest)
            if action in {'complete', 'failed'}:
                record(action, last_checkpoint=manifest.get('last_checkpoint'))
                return
            if action == 'wait':
                time.sleep(15)
                continue
            # The manifest can be updated just before the old writer exits.
            with (training / 'writer.lock').open('a') as writer:
                fcntl.flock(writer, fcntl.LOCK_EX)
            checkpoint = training / manifest['last_checkpoint']
            state = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())['state']
            step = state['optimizer_step']
            if step <= last_resumed_step:
                raise RuntimeError('Refusing repeated resume without checkpoint progress')
            last_resumed_step = step
            command = [sys.executable, '-u', '-m', 'scripts.finetune_contrastive', 'train',
                       str(root / 'data'), str(training), '--config', str(args.config),
                       '--device', manifest['signature']['device'], '--resume', str(checkpoint)]
            log_name = f'training_auto_resume_from_{step:06d}.log'
            record('resuming', checkpoint=str(checkpoint), command=command, log=log_name)
            with (root / 'budget_resume_events.jsonl').open('a') as events:
                events.write(json.dumps(dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                                            step=step, command=command, log=log_name)) + '\n')
            env = os.environ.copy()
            env.update(HF_HOME='data/hf-cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            with (root / log_name).open('ab') as log:
                result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            if result.returncode:
                record('failed', returncode=result.returncode, log=log_name)
                return


if __name__ == '__main__':
    main()
