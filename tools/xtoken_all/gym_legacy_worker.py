"""Render and assess the unchanged company GRPO/OPD entrypoint on real GPUs."""

from copy import deepcopy
import json
import math
from pathlib import Path
import hashlib
import urllib.request

import yaml


def render_legacy(config, common):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from nemo_rl.utils.config import load_config

    recipe = yaml.safe_load(Path('/work/source/tools/xtoken_all/gym_legacy.yaml').read_text())
    base = load_config('/work/source/examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml')
    for key in ('grpo', 'loss_fn', 'on_policy_distillation', 'penalize_duplicated_reasoning',
                'penalize_empty_final_answer', 'penalize_eos_token', 'penalize_malformed_think_tag'):
        base[key] = deepcopy(recipe[key])
    for key in ('policy', 'cluster', 'data', 'logger', 'checkpointing'):
        base[key] = deepcopy(common[key])
    knobs = config['canary']
    base.grpo.num_prompts_per_step = knobs['global_batch_size']
    base.grpo.num_generations_per_prompt = 1
    base.grpo.max_num_steps = knobs['max_num_steps']
    base.grpo.val_at_start = False
    base.grpo.val_at_end = False
    base.grpo.val_period = 0
    base.grpo.max_val_samples = None
    base.grpo.seq_logprob_error_threshold = None
    base.grpo.skip_reference_policy_logprobs_calculation = True
    base.grpo.async_grpo.max_trajectory_age_steps = recipe['canary_overrides']['max_trajectory_age_steps']
    base.logger.tensorboard_enabled = True
    base.logger.swanlab_enabled = False
    base.checkpointing.metric_name = 'train:loss'
    base.env = {'should_use_nemo_gym': True, 'should_log_nemo_gym_responses': False,
                'nemo_gym': deepcopy(common['on_policy']['nemo_gym']['global_config'])}
    base.env.nemo_gym.general_qa.resources_servers.general_qa.judge_server_url = '${oc.env:XTOKEN_LEGACY_TEACHER_URL}'
    base.env.nemo_gym.policy_model.responses_api_models.vllm_model.num_groups_nemo_rl = 2
    tokenizer = AutoTokenizer.from_pretrained(common['policy']['tokenizer']['name'], local_files_only=True)
    ids = {'eos': tokenizer.eos_token_id, 'think_open': tokenizer.convert_tokens_to_ids('<think>'),
           'think_close': tokenizer.convert_tokens_to_ids('</think>')}
    if any(type(value) is not int or value < 0 or value >= len(tokenizer) for value in ids.values()):
        raise ValueError('Legacy format-penalty token IDs must belong to the actual Qwen tokenizer')
    base.token_ids = ids
    return OmegaConf.to_container(base, resolve=True), recipe


def assess_legacy(run, relay_url):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tools.xtoken_all.check_result import inspect_checkpoint

    exits = json.loads((run / 'process-exit.json').read_text())
    lines = (run / 'driver.log').read_text(errors='replace')
    reasons, scalars, event_hashes = [], {}, {}
    for event in run.rglob('events.out.tfevents.*'):
        event_hashes[event.relative_to(run).as_posix()] = hashlib.sha256(event.read_bytes()).hexdigest()
        accumulator = EventAccumulator(str(event)).Reload()
        for tag in accumulator.Tags()['scalars']:
            scalars.setdefault(tag, {}).update({str(item.step): item.value for item in accumulator.Scalars(tag)})
    loss = scalars['train/loss'] if 'train/loss' in scalars else {}
    grad_tags = [tag for tag in scalars if tag.startswith('train/') and 'grad_norm' in tag]
    if exits['driver_exit_code'] != 0 or exits['tee_exit_code'] != 0:
        reasons.append('Legacy entrypoint process failed')
    if set(loss) != {'1', '2'} or not all(math.isfinite(value) for value in loss.values()):
        reasons.append('Expected two finite legacy loss measurements')
    if not grad_tags or not all({'1', '2'}.issubset(scalars[tag]) and all(
            math.isfinite(scalars[tag][step]) and scalars[tag][step] > 0 for step in ('1', '2')) for tag in grad_tags):
        reasons.append('Missing/nonfinite/zero legacy gradient measurements')
    if 'Max number of steps has been reached, stopping training early' not in lines:
        reasons.append('Legacy normal completion marker missing')
    with urllib.request.urlopen(relay_url + '/_receipts', timeout=10) as response:
        receipts = json.load(response)
    if len(receipts) < 2 or not all(item['status'] == 200 for item in receipts):
        reasons.append('Insufficient successful real legacy teacher scoring calls')
    checkpoints = [inspect_checkpoint(run / 'checkpoints', step) for step in (1, 2)]
    for report in checkpoints:
        reasons.extend(report['reasons'])
        for name in ('replay_buffer.pt', 'rollouts.pt'):
            path = Path(report['path']) / name
            if not path.is_file() or not path.stat().st_size:
                reasons.append('Legacy checkpoint missing ' + name)
    return {'success': not reasons, 'reasons': reasons, 'mode': 'legacy_opd',
            'loss': loss, 'gradient_metrics': {tag: scalars[tag] for tag in grad_tags},
            'teacher_score_receipts': receipts, 'checkpoints': checkpoints,
            'tensorboard_sha256': event_hashes,
            'scope': 'Two-update mechanical legacy OPD/Gym check; no quality or resume-parity claim'}
