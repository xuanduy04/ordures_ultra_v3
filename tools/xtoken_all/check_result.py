"""Judge a two-update smoke from its metrics and successful process exits."""
import argparse
import json
import math
from pathlib import Path

import yaml


def inspect_checkpoint(root, step):
    """Check finalized file closure without loading tensors or pickle data."""
    path = root / ('step_' + str(step))
    reasons = []
    info = path / 'training_info.json'
    try:
        valid_info = info.is_file() and json.loads(info.read_text())['total_steps'] == step
    except (OSError, ValueError, KeyError, TypeError):
        valid_info = False
    if not valid_info:
        reasons.append('missing/wrong training_info for checkpoint step ' + str(step))
    for name in ('config.yaml', 'train_dataloader.pt'):
        item = path / name
        if not item.is_file() or item.stat().st_size == 0:
            reasons.append('missing checkpoint file: ' + str(item))
    for name in ('weights', 'optimizer', 'tokenizer'):
        directory = path / 'policy' / name
        files = [item for item in directory.rglob('*') if item.is_file() and item.stat().st_size > 0]
        if name != 'tokenizer':
            files = [item for item in files if item.suffix in ('.safetensors', '.distcp', '.pt', '.pth', '.bin')]
        if not files:
            reasons.append('missing checkpoint payload: ' + str(directory))
    return {'step': step, 'path': str(path), 'success': not reasons, 'reasons': reasons,
            'scope': 'finalized metadata/file closure; no load or resume-parity claim'}


def assess(run, mode):
    lines = (run / 'driver.log').read_text(errors='replace').splitlines()
    def events(prefix):
        return [json.loads(line[len(prefix):]) for line in lines if line.startswith(prefix)]
    steps = events('XTOKEN_TRAIN_STEP=')
    rollouts = events('XTOKEN_ROLLOUT=')
    scores = events('XTOKEN_TEACHER_SCORE=')
    exits = json.loads((run / 'process-exit.json').read_text())
    reasons = []
    if exits['driver_exit_code'] != 0 or exits['tee_exit_code'] != 0:
        reasons.append('process/log capture failed')
    if [item['step'] for item in steps] != [1, 2]:
        reasons.append('expected exactly two updates')
    if not all(math.isfinite(item['loss']) and math.isfinite(item['grad_norm']) and item['grad_norm'] > 0 for item in steps):
        reasons.append('non-finite loss/gradient or zero gradient')
    if not scores or not all(item['representation'] == 'topk_logprobs' and item['top_k'] == 20 for item in scores):
        reasons.append('missing or mismatched teacher-score representation')
    if mode == 'on' and [item['policy_version'] for item in rollouts if item['training']] != [0, 1]:
        reasons.append('rollout versions must be exactly 0 then 1')
    if mode == 'on' and not all(item['response_tokens'] > 0 for item in rollouts if item['training']):
        reasons.append('empty rollout')
    if mode == 'on' and not all(item['backend'] == 'nemo_gym' and item['reward_used_in_loss'] is False
                              for item in rollouts if item['training']):
        reasons.append('wrong rollout backend or reward mixed into KD')
    if not any('Max steps reached, stopping training.' in line for line in lines):
        reasons.append('missing normal completion marker')
    checkpoint_reports = []
    cfg = yaml.safe_load((run / 'effective.yaml').read_text())
    if cfg['checkpointing']['enabled']:
        root = Path(cfg['checkpointing']['checkpoint_dir']).resolve()
        root.relative_to(run.resolve())
        for step in (1, 2):
            report = inspect_checkpoint(root, step)
            checkpoint_reports.append(report)
            reasons.extend(report['reasons'])
    return {'success': not reasons, 'reasons': reasons, 'mode': mode, 'steps': steps,
            'rollouts': rollouts, 'teacher_score_receipts': len(scores),
            'checkpoints': checkpoint_reports,
            'scope': 'mechanical smoke; no quality, convergence, full tokenizer coverage or resume-parity claim'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    mode = json.loads((args.run_dir / 'provenance.json').read_text())['mode']
    result = assess(args.run_dir, mode)
    (args.run_dir / 'smoke-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
