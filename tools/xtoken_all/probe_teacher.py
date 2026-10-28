"""Read-only metadata/tokenizer probes plus exactly one tiny teacher-score request."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from integrity import sha256
from teacher_probe_contract import mode_basis, score_contract, validate_teacher_probe


def main():
    started = time.monotonic()
    def phase(name):
        print('XTK_TEACHER_PHASE=' + json.dumps({'phase': name, 'elapsed_s': time.monotonic() - started}), flush=True)

    phase('starting')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--server-mode-evidence', type=Path)
    args = parser.parse_args()
    path = Path(args.config).resolve().parent / 'teacher-probe.json'
    path.write_text(json.dumps({'success': False, 'state': 'probe_pending',
                               'effective_config_sha256': sha256(args.config)}) + '\n')
    source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source))
    import torch
    from omegaconf import OmegaConf
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.algorithms.x_token.remote_teacher import XTokenFullLogprobClient, XTokenTopKLogprobClient

    phase('framework_imports_complete')

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    policy_path = Path(__file__).with_name('teacher_probe_policy.yaml')
    policy = OmegaConf.to_container(OmegaConf.load(policy_path), resolve=True)
    remote = dict(cfg['teachers'][0]['remote'])
    evidence = json.loads(args.server_mode_evidence.read_text()) if args.server_mode_evidence else None
    score_mode, expected_k, _ = score_contract(cfg)
    assert remote['score_mode'] == score_mode
    if score_mode == 'topk':
        assert remote['top_k'] == expected_k
    key_name = remote['api_key_env']
    headers = {'Authorization': 'Bearer ' + (os.environ[key_name] if key_name else 'EMPTY')}
    base_url = remote['url'].rstrip('/')

    def request(route, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base_url + route, data=body,
                                     headers={**headers, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = response.read(1048577)
                if len(result) > 1048576:
                    raise ValueError('Metadata/tokenize response exceeds 1 MiB')
                return json.loads(result)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code} at {route}; inspect route/auth/capability, server not changed') from None

    version = request('/version')['version']
    models = [item['id'] for item in request('/v1/models')['data']]
    phase('metadata_received')
    if remote['model'] not in models:
        raise ValueError('Configured served name is absent from /v1/models')
    basis = mode_basis(remote, version, evidence, policy)
    if evidence is not None and not basis['accepted']:
        raise ValueError('Explicit deployment configuration reports an incompatible logprobs mode')
    tokenizer = get_tokenizer(cfg['teachers'][0]['tokenizer'])
    phase('tokenizer_loaded')
    for text in ('Hello world.', 'Xin chào, 201 + 37 = ?',
                 tokenizer.apply_chat_template([{'role': 'user', 'content': 'Say hello.'}],
                     tokenize=False, add_generation_prompt=True)):
        expected = tokenizer.encode(text, add_special_tokens=False)
        actual = request('/tokenize', {'model': remote['model'], 'prompt': text,
                                     'add_special_tokens': False})['tokens']
        if actual != expected:
            raise ValueError('Teacher tokenizer IDs differ from staged tokenizer; do not train')
    phase('tokenizer_parity_complete')
    ids = tokenizer.encode('Hello world. Another sentence.', add_special_tokens=False)[:2]
    assert len(ids) == 2
    assert max(ids) < remote['model_vocab_size']
    remote['max_attempts'] = 1
    client = XTokenTopKLogprobClient(remote) if score_mode == 'topk' else XTokenFullLogprobClient(remote)
    try:
        scores = asyncio.run(client.score_group(
            {'url': remote['url'], 'model': remote['model']},
            torch.tensor([ids], dtype=torch.long), torch.tensor([2], dtype=torch.long)))
        phase('scores_received')
        if score_mode == 'topk':
            valid = scores['topk_valid'][0, 0]
            retained = float(scores['topk_logprobs'][0, 0][valid].exp().sum())
            count, k, representation = int(valid.sum()), remote['top_k'], 'topk_logprobs'
            assert k <= count <= k + 1
        else:
            count, k, representation = scores.shape[-1], None, 'full_logprobs'
            retained = float(scores[0, 0].exp().sum())
        result = {'success': basis['accepted'], 'teacher_score_contract_satisfied': basis['accepted'],
                  'score_request_completed': True, 'representation': representation,
                  'vllm_version_observed': version, 'served_model': remote['model'],
                  'top_k': k, 'returned_count': count,
                  'returned_probability_mass': retained, 'tokenizer_parity_probes': 3,
                  'normalization_atol': remote['normalization_atol'],
                  **{key: value for key, value in basis.items() if key != 'accepted'},
                  'server_mode_evidence_sha256': sha256(args.server_mode_evidence) if args.server_mode_evidence else None,
                  'teacher_probe_policy_sha256': sha256(policy_path),
                  'effective_config_sha256': sha256(args.config),
                  'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                  'note': 'Scores/parity measured. Upstream defaults, when used, are an accepted assumption, not observed server configuration.'}
        if result['success']:
            validate_teacher_probe(result, policy, cfg)
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('XTK_TEACHER_PROBE=' + json.dumps(result))
        if not result['success']:
            raise SystemExit('Scoring observed, but mode unconfirmed for this version; supply deployment evidence, not a teacher restart by default')
    finally:
        asyncio.run(client.close())


if __name__ == '__main__':
    main()
