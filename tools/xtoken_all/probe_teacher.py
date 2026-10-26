"""Read-only metadata/tokenizer probes plus exactly one tiny teacher-score request."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

from integrity import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--server-mode-evidence', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source))
    import torch
    from omegaconf import OmegaConf
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.algorithms.x_token.remote_teacher import XTokenTopKLogprobClient

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    remote = dict(cfg['teachers'][0]['remote'])
    evidence = json.loads(args.server_mode_evidence.read_text())
    assert evidence['logprobs_mode'] == 'raw_logprobs'
    assert evidence['teacher_url'].rstrip('/') == remote['url'].rstrip('/')
    assert evidence['served_model'] == remote['model']
    assert evidence['verification_source'] in ('operator_checked_server_args', 'server_reported_config')
    assert evidence['observed_at_utc'] and evidence['evidence_description']
    assert remote['score_mode'] == 'topk' and remote['top_k'] == 20
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
    if remote['model'] not in models:
        raise ValueError('Configured served name is absent from /v1/models')
    tokenizer = get_tokenizer(cfg['teachers'][0]['tokenizer'])
    for text in ('Hello world.', 'Xin chào, 201 + 37 = ?',
                 tokenizer.apply_chat_template([{'role': 'user', 'content': 'Say hello.'}],
                     tokenize=False, add_generation_prompt=True)):
        expected = tokenizer.encode(text, add_special_tokens=False)
        actual = request('/tokenize', {'model': remote['model'], 'prompt': text,
                                     'add_special_tokens': False})['tokens']
        if actual != expected:
            raise ValueError('Teacher tokenizer IDs differ from staged tokenizer; do not train')
    ids = tokenizer.encode('Hello world. Another sentence.', add_special_tokens=False)[:2]
    assert len(ids) == 2
    assert max(ids) < remote['model_vocab_size']
    remote['max_attempts'] = 1
    client = XTokenTopKLogprobClient(remote)
    try:
        scores = asyncio.run(client.score_group(
            {'url': remote['url'], 'model': remote['model']},
            torch.tensor([ids], dtype=torch.long), torch.tensor([2], dtype=torch.long)))
        valid = scores['topk_valid'][0, 0]
        retained = float(scores['topk_logprobs'][0, 0][valid].exp().sum())
        assert remote['top_k'] <= int(valid.sum()) <= remote['top_k'] + 1
        result = {'success': True, 'vllm_version_observed': version, 'served_model': remote['model'],
                  'top_k': remote['top_k'], 'returned_count': int(valid.sum()),
                  'returned_probability_mass': retained, 'tokenizer_parity_probes': 3,
                  'server_logprobs_mode_verified': True,
                  'server_mode_verification_source': evidence['verification_source'],
                  'server_mode_evidence_sha256': sha256(args.server_mode_evidence),
                  'effective_config_sha256': sha256(args.config),
                  'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                  'note': 'Operator must independently confirm raw_logprobs mode; echo/mass alone cannot prove it.'}
        path = Path(args.config).resolve().parent / 'teacher-probe.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('XTK_TEACHER_PROBE=' + json.dumps(result))
    finally:
        asyncio.run(client.close())


if __name__ == '__main__':
    main()
