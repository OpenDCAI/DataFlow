import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from filelock import FileLock, Timeout

from dataflow.operators.core_vision import ChineseVisualCoTGenerator
from dataflow.utils.visual_cot.model_protocol import encode_image
from dataflow.prompts.chinese_visual_cot import ChineseVisualCoTPrompt, INPUT_MARKER
from dataflow.utils.visual_cot.preprocessing import BestEffortImageFilter, EstimatedImagePolicy
from dataflow.utils.visual_cot.preprocessing import projected_clip_output
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import FileStorage, LazyFileStorage
from visual_cot_fixtures import DemoCLIP, DemoModel, make_demo
from dataflow.utils.visual_cot.serving import VisualCoTAPIServing, VisualCoTServingBundle
pytestmark = pytest.mark.cpu


class MemoryStorage:
    def __init__(self, frame):
        self.frame = frame.copy()

    def read(self, kind):
        assert kind == 'dataframe'
        return self.frame.copy()

    def write(self, frame):
        self.frame = frame.copy()
        return 'memory'


def make_case(tmp_path, indices=(0, 1), journal=False, policy=None):
    path = make_demo(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    storage = MemoryStorage(pd.DataFrame([rows[i] for i in indices]))
    small = DemoModel('Qwen3.5-4B')
    teachers = [DemoModel(n) for n in ['Qwen3.5-9B', 'Qwen3.5-27B', 'Qwen3.5-122B-A10B']]
    verifier = DemoModel('Qwen3.5-27B', 'verifier')
    cleaner = DemoModel('Qwen3.5-122B-A10B', 'cleaner')
    image_filter = BestEffortImageFilter(DemoCLIP(), policy, use_faiss=False)
    op = ChineseVisualCoTGenerator(VisualCoTServingBundle(small, teachers, verifier, cleaner), image_filter, tmp_path,
                                    tmp_path / 'journal' if journal else None)
    return op, storage


def results(op, storage):
    key = op.run(storage)
    return list(storage.frame[key])


def test_registered_native_operator():
    assert OPERATOR_REGISTRY.get('ChineseVisualCoTGenerator') is ChineseVisualCoTGenerator


def test_lazy_registration_preserves_existing_vision_operator():
    code = """
import sys
from dataflow.operators import core_vision
from dataflow.utils.registry import OPERATOR_REGISTRY

module = 'dataflow.utils.visual_cot.operator'
assert module not in sys.modules
original = core_vision.PromptedVQAGenerator
assert module not in sys.modules
from dataflow.operators.core_vision.generate.chinese_visual_cot_generator import ChineseVisualCoTGenerator
assert core_vision.ChineseVisualCoTGenerator is ChineseVisualCoTGenerator
assert OPERATOR_REGISTRY.get('ChineseVisualCoTGenerator') is ChineseVisualCoTGenerator
assert core_vision.PromptedVQAGenerator is original
assert OPERATOR_REGISTRY.get('PromptedVQAGenerator') is original
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('storage_class', [FileStorage, LazyFileStorage])
def test_real_storage_keeps_input_and_reference_string(tmp_path, storage_class):
    operator, _ = make_case(tmp_path)
    source = tmp_path / 'input.jsonl'
    row = {'id': '0008', 'images': ['images/three.png'], 'question': '图中红色方块有几个？', 'answer': '003'}
    source.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    options = {'save_on_exit': False} if storage_class is LazyFileStorage else {}
    storage = storage_class(str(source), cache_path=str(tmp_path / 'cache'), **options)
    assert operator.run(storage.step()) == 'visual_cot_result'
    output_step = storage.step()
    if isinstance(output_step, LazyFileStorage):
        output_step.flush_all()
    # Load the persisted JSON as text so pandas' default inference does not coerce "003" to 3.
    path = Path(output_step._get_cache_file_path(output_step.operator_step))
    output = json.loads(path.read_text(encoding='utf-8'))
    assert {key: output[key] for key in row} == row
    assert output['visual_cot_result']['reference_answer'] == '003'


def test_direct_run_protects_shared_journal_before_generation(tmp_path):
    operator, storage = make_case(tmp_path, (0,), journal=True)
    with FileLock(tmp_path / 'journal.lock', timeout=0):
        with pytest.raises(Timeout):
            operator.run(storage)
    assert not operator.cleaner.calls and not operator.journal_dir.exists()
    assert results(operator, storage)[0]['status'] == 'accepted'


def test_full_cascade_selects_likelihood_not_first_verified(tmp_path):
    op, storage = make_case(tmp_path)
    a, b = results(op, storage)
    assert [a['status'], b['status']] == ['accepted', 'accepted']
    assert a['teacher'] == 'Qwen3.5-9B' and a['selected_rollout_id'] == 2
    assert b['teacher'] == 'Qwen3.5-27B' and b['selected_rollout_id'] == 0
    assert len(op.teachers[0].calls) == 8  # all four candidates for both questions
    assert len(op.teachers[0].score_calls) == 2
    assert len(op.teachers[1].calls) == 4
    assert not op.teachers[1].score_calls  # direct selection of a sole valid answer
    assert not op.teachers[2].calls
    assert all(not call['images'] for call in op.verifier.calls)
    assert all(len(call['images']) == 1 for call in op.small.calls + op.cleaner.calls)
    assert a['mocked'] and a['protocol'] == 'visual-cot-zh-v2'


def test_any_of_four_small_model_passes_excludes_sample(tmp_path):
    op, storage = make_case(tmp_path, (2,))
    result, = results(op, storage)
    assert result['reason'] == 'solved_by_4B_filter'
    assert result['difficulty'] == {'correct_count': 1, 'rollout_count': 4}
    assert len(op.small.calls) == 4
    assert all(not t.calls for t in op.teachers)


@pytest.mark.parametrize('keep_one', [True, False])
def test_teacher_language_checked_before_likelihood(tmp_path, keep_one):
    op, storage = make_case(tmp_path, (0,))
    original = op.teachers[0].complete
    def generate(**kw):
        value = original(**kw)
        if keep_one and kw['phase'].endswith('_1'):
            value['text'] = '<think>根据图形逐一计数，共有三个。</think>\nTherefore, the final answer is <answer>3</answer>'
        else:
            value['text'] = re.sub(r'<think>.*?</think>', '<think>中。The count is three.</think>', value['text'])
        return value
    op.teachers[0].complete = generate
    record, = results(op, storage)
    assert not op.teachers[0].score_calls and not op.teachers[1].calls
    if keep_one:
        assert record['selected_rollout_id'] == 1
        assert record['messages'][-1]['content'] == '<think>根据图形逐一计数，共有三个。</think>\n<answer>3</answer>'
    else:
        assert record['reason'] == 'reasoning_not_chinese' and 'messages' not in record


def test_reasoning_language_does_not_change_literal_answer(tmp_path):
    op, storage = make_case(tmp_path, (0,))
    storage.frame.loc[0, 'answer'] = 'STOP'
    op.teachers[0].complete = lambda **kw: {'text': (
        '<think>画面中的标牌写着 STOP。</think><answer>STOP</answer>'
        if kw['phase'].endswith('_0') else '<think>Count three.</think><answer>STOP</answer>')}
    record, = results(op, storage)
    assert record['answer'] == 'STOP' and record['selected_rollout_id'] == 0


def test_three_unsolved_teachers_drop_sample(tmp_path):
    op, storage = make_case(tmp_path, (0,))
    storage.frame.loc[0, 'question'] = '请确定整幅图片的图形总数。'
    result, = results(op, storage)
    assert result['reason'] == 'unsolved_by_all_three_teachers'
    assert [len(t.calls) for t in op.teachers] == [4, 4, 4]


def test_first_solving_teacher_is_not_replaced_for_format(tmp_path):
    op, storage = make_case(tmp_path, (0,))
    original_verifier = op.verifier.complete
    def verify(**kw):
        if kw['phase'].startswith('teacher_0'):
            return {'text': '{"verdict":"correct","reasoning_consistent":true,"reason":"test"}'}
        return original_verifier(**kw)
    op.verifier.complete = verify
    op.teachers[0].complete = lambda **kw: {'text': '没有最终答案标签的候选解答'}
    result, = results(op, storage)
    assert result['reason'] == 'first_solving_teacher_has_no_format_valid_candidate'
    assert not op.teachers[1].calls


def test_normalized_score_and_stable_tie_break(tmp_path):
    op, storage = make_case(tmp_path, (0,))
    def score(**kw):
        i = int(re.search(r'候选(\d+)', kw['response']).group(1))
        return {'answer_logprob_mean': -1 if i == 1 else -3,
                'answer_logprob_sum': -100 if i == 1 else -3}
    op.teachers[0].score_answer = score
    result, = results(op, storage)
    assert result['selected_rollout_id'] == 1
    op2, storage2 = make_case(tmp_path, (0,))
    op2.teachers[0].score_answer = lambda **kw: {'answer_logprob_mean': -1.0}
    result2, = results(op2, storage2)
    assert result2['selected_rollout_id'] == 1


def test_unparseable_judge_is_not_zero_of_four(tmp_path):
    op, storage = make_case(tmp_path, (0,))
    op.verifier.complete = lambda **kw: {'text': 'not JSON'}
    with pytest.raises(RuntimeError, match='not a verified incorrect'):
        op.run(storage)
    assert not op.teachers[0].calls


@pytest.mark.parametrize(('role', 'incomplete_phase', 'finish_reason'), [
    ('small', 'difficulty_rollout_0', 'length'),
    ('teacher', 'teacher_0_rollout_2', 'length'),
    ('verifier', 'teacher_0_verify_2', 'content_filter'),
])
def test_incomplete_service_reply_cannot_enter_training_output(tmp_path, role, incomplete_phase, finish_reason):
    op, storage = make_case(tmp_path, (0,))
    provider = {'small': op.small, 'teacher': op.teachers[0], 'verifier': op.verifier}[role]
    original = provider.complete

    def incomplete(**kwargs):
        reply = original(**kwargs)
        if kwargs['phase'] == incomplete_phase:
            reply['finish_reason'] = finish_reason
        return reply

    provider.complete = incomplete
    with pytest.raises(RuntimeError, match='finish_reason'):
        op.run(storage)
    assert 'visual_cot_result' not in storage.frame


def test_reference_never_leaks_to_generation_via_template_tokens(tmp_path):
    prompt = ChineseVisualCoTPrompt().build_prompt('generation', '中文原文 {answer}', 'GOLD_SECRET')
    assert json.loads(prompt.split(INPUT_MARKER, 1)[1]) == {'question': '中文原文 {answer}'}
    op, storage = make_case(tmp_path, (0,))
    storage.frame.loc[0, 'question'] = '请数出红色图形，原文带有{answer}占位符。'
    storage.frame.loc[0, 'answer'] = 'GOLD_SECRET'
    results(op, storage)
    assert all('GOLD_SECRET' not in c['prompt'] for c in op.small.calls + op.teachers[0].calls)
    assert any('GOLD_SECRET' in c['prompt'] for c in op.verifier.calls)


def test_invalid_and_duplicate_inputs_do_not_generate(tmp_path):
    op, storage = make_case(tmp_path, (0, 3, 4))
    out = results(op, storage)
    assert [r['status'] for r in out] == ['accepted', 'rejected', 'invalid_input']
    assert out[1]['reason'] == 'exact_image_QA_duplicate'
    assert len(op.small.calls) == 4
    op2, storage2 = make_case(tmp_path, (0,))
    storage2.frame.loc[0, 'answer'] = float('nan')
    assert results(op2, storage2)[0]['status'] == 'invalid_input'


def test_same_image_different_questions_are_retained(tmp_path):
    make_demo(tmp_path)
    filtering = BestEffortImageFilter(DemoCLIP(), use_faiss=False)
    image = tmp_path / 'images/three.png'
    assert filtering.evaluate(image, '红色方块的数量是多少？', '3')['keep']
    assert filtering.evaluate(image, '这些形状在画面中的颜色是什么？', '红色')['keep']


@pytest.mark.parametrize('missing', [pd.NA, pd.NaT])
def test_nullable_reference_is_invalid_before_any_model_call(tmp_path, missing):
    op, storage = make_case(tmp_path, (0,))
    storage.frame['answer'] = pd.Series([missing], dtype=object)
    assert results(op, storage)[0]['status'] == 'invalid_input'
    assert not op.cleaner.calls


def test_resume_reuses_results_and_rejects_changed_policy(tmp_path):
    op, storage = make_case(tmp_path, (0, 1), journal=True)
    expected = results(op, storage)
    op2, storage2 = make_case(tmp_path, (0, 1), journal=True)
    assert results(op2, storage2) == expected
    assert not op2.small.calls and not op2.cleaner.calls
    op3, storage3 = make_case(tmp_path, (0, 1), journal=True,
                          policy=EstimatedImagePolicy(min_short_side=33))
    with pytest.raises(ValueError, match='Resume manifest differs'):
        op3.run(storage3)


@pytest.mark.parametrize('field', ['user_turn', 'selected_rollout_id'])
def test_resume_detects_changed_training_question_or_rollout(tmp_path, field):
    op, storage = make_case(tmp_path, (0,), journal=True)
    assert results(op, storage)[0]['status'] == 'accepted'
    checkpoint = op.journal_dir / '000000000.json'
    saved = json.loads(checkpoint.read_text(encoding='utf8'))
    if field == 'user_turn':
        saved['messages'][0]['content'] = '<image>\n这是一道不同的题。'
    else:
        saved['selected_rollout_id'] = 0
    checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding='utf8')
    resumed, resumed_storage = make_case(tmp_path, (0,), journal=True)
    with pytest.raises(ValueError, match='checksum'):
        resumed.run(resumed_storage)
    assert 'visual_cot_result' not in resumed_storage.frame
    assert not resumed.cleaner.calls


def test_optional_nullable_source_labels_and_metadata_resume_identity(tmp_path):
    op, storage = make_case(tmp_path, (0,), journal=True)
    storage.frame['type'] = pd.Series([pd.NA], dtype=object)
    storage.frame['subtype'] = pd.Series([float('nan')], dtype=object)
    result, = results(op, storage)
    assert result['status'] == 'accepted' and result['type'] == 'numeric'
    assert result['subtype'] is None
    changed, changed_storage = make_case(tmp_path, (0,), journal=True)
    changed_storage.frame['type'] = 'multiple_choice'
    with pytest.raises(ValueError, match='Resume manifest differs'):
        changed.run(changed_storage)


@pytest.mark.parametrize('invalid_score', [True, float('nan'), None, 1.0])
def test_invalid_likelihood_values_abort_instead_of_selecting(tmp_path, invalid_score):
    op, storage = make_case(tmp_path, (0,))
    op.teachers[0].score_answer = lambda **kw: {'answer_logprob_mean': invalid_score}
    with pytest.raises(RuntimeError, match='Actual finite answer-token'):
        op.run(storage)


@pytest.mark.parametrize('size,expected', [((16, 100), (32, 200)), ((4096, 512), (2048, 256)),
                                          ((512, 512), (512, 512)), ((3, 10000), (1, 2048))])
def test_image_resizing_and_jpeg_encoding(tmp_path, size, expected):
    path = tmp_path / 'input.png'
    Image.new('RGB', size, 'white').save(path)
    uri = encode_image(str(path))
    assert uri.startswith('data:image/jpeg;base64,')
    with Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1]))) as image:
        assert image.size == expected


def test_real_likelihood_uses_token_ids_and_handles_boundary_merges():
    calls = []
    def transport(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {'prompt_token_ids': [10, 20, 30]}
        return {'prompt_token_ids': [10, 20, 40, 50],
                'prompt_logprobs': [None, {'20': {'logprob': -.2}},
                                    {'40': {'logprob': -1}}, {'50': {'logprob': -3}}]}
    serving = VisualCoTAPIServing('teacher', 'Qwen3.5-9B', 'http://localhost/v1', transport=transport)
    value = serving.score_answer(prompt='问题', images=[], response='<think>推理</think><answer>3</answer>')
    assert value['target_start_token'] == 2
    assert value['answer_logprob_mean'] == -2
    assert all(c['prompt_logprobs'] == 1 and c['return_token_ids'] is True for c in calls)
    assert all(c['continue_final_message'] and not c['add_generation_prompt'] for c in calls)
    assert calls[1]['messages'][-1] == {'role': 'assistant', 'reasoning_content': '推理', 'content': '<answer>3'}


@pytest.mark.parametrize('missing', [None, float('nan'), float('inf')])
def test_incomplete_target_likelihood_is_rejected(missing):
    replies = iter([{'prompt_token_ids': [10, 20]}, {
        'prompt_token_ids': [10, 20, 40, 50],
        'prompt_logprobs': [None, None, {'40': {'logprob': -.1}}, {'50': {'logprob': missing}}],
    }])
    serving = VisualCoTAPIServing('teacher', 'Qwen3.5-9B', 'http://unused', transport=lambda _: next(replies))
    with pytest.raises(RuntimeError, match='Incomplete or nonfinite'):
        serving.score_answer(prompt='问题', images=[], response='<think>推理</think><answer>xy</answer>')


def test_positive_token_logprob_cannot_be_hidden_by_a_negative_average():
    replies = iter([{'prompt_token_ids': [10, 20]}, {
        'prompt_token_ids': [10, 20, 40, 50],
        'prompt_logprobs': [None, None, {'40': {'logprob': 1.0}}, {'50': {'logprob': -3.0}}],
        'choices': [{'finish_reason': 'length'}],  # scoring requests deliberately generate only one token
    }])
    serving = VisualCoTAPIServing('teacher', 'Qwen3.5-9B', 'http://unused', transport=lambda _: next(replies))
    with pytest.raises(RuntimeError, match='positive'):
        serving.score_answer(prompt='问题', images=[], response='<think>推理</think><answer>xy</answer>')


def test_scoring_rejects_error_even_when_prompt_scores_are_present():
    replies = iter([{'prompt_token_ids': [10]}, {
        'prompt_token_ids': [10, 40], 'prompt_logprobs': [None, {'40': {'logprob': -.3}}],
        'choices': [{'finish_reason': 'error'}],
    }])
    serving = VisualCoTAPIServing('teacher', 'Qwen3.5-9B', 'http://unused', transport=lambda _: next(replies))
    with pytest.raises(RuntimeError, match='finish_reason'):
        serving.score_answer(prompt='问题', images=[], response='<think>推理</think><answer>3</answer>')


def test_missing_logprobs_never_falls_back_to_textual_score():
    serving = VisualCoTAPIServing('teacher', 'Qwen3.5-9B', 'http://localhost/v1',
                              transport=lambda p: {'prompt_token_ids': [1, 2]})
    with pytest.raises(RuntimeError, match='prompt_logprobs'):
        serving.score_answer(prompt='问题', images=[], response='<think>推理</think><answer>3</answer>')


def test_verifier_payload_is_text_only():
    calls = []
    def transport(payload):
        calls.append(payload)
        return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
    serving = VisualCoTAPIServing('judge', 'Qwen3.5-27B', 'http://localhost/v1', purpose='verifier', transport=transport)
    serving.complete(prompt='核验输入', images=[], phase='verify')
    payload, = calls
    assert payload['max_tokens'] == 512 and payload['top_k'] == 1
    assert all(isinstance(m['content'], str) for m in payload['messages'])
    with pytest.raises(ValueError, match='text only'):
        serving.complete(prompt='核验输入', images=['image.png'], phase='verify')


@pytest.mark.parametrize('reply', [
    {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'length'}]},
    {'choices': [{'message': {'content': '{}', 'refusal': 'denied'}, 'finish_reason': 'stop'}]},
    {'error': {'message': 'backend failure'}, 'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]},
])
def test_api_completion_rejects_explicit_backend_failure(reply):
    serving = VisualCoTAPIServing('judge', 'Qwen3.5-27B', 'http://unused', purpose='verifier', transport=lambda _: reply)
    with pytest.raises(RuntimeError):
        serving.complete(prompt='核验', images=[], phase='verify')


def test_thresholds_are_declared_and_all_model_roles_required(tmp_path):
    op, _ = make_case(tmp_path)
    assert op.manifest['raw_filter']['policy']['status'] == 'heuristic_thresholds'
    assert 'chinese_visual_cot.py' in op.manifest['implementation_sha256']
    with pytest.raises(ValueError, match='Required source-model order'):
        VisualCoTServingBundle(op.small, op.teachers[:2], op.verifier, op.cleaner)


def test_likelihood_reads_choice_level_token_data():
    replies = iter([
        {'choices': [{'finish_reason': 'length', 'prompt_tokens': ['10', '20']}]},
        {'choices': [{'finish_reason': 'length', 'prompt_tokens': ['10', '20', '40'],
                      'prompt_logprobs': [None, None, {'token_id': 40, 'logprob': -0.25}]}]},
    ])
    serving = VisualCoTAPIServing('teacher', 'Qwen3.5-9B', 'http://unused', transport=lambda _: next(replies))
    value = serving.score_answer(prompt='问题', images=[], response='<think>推理</think><answer>3</answer>')
    assert value['answer_logprob_mean'] == -0.25 and value['answer_token_count'] == 1


def test_clip_old_and_current_projected_feature_return_types():
    from types import SimpleNamespace
    features = np.ones((1, 8))
    assert projected_clip_output(features) is features
    assert projected_clip_output(SimpleNamespace(pooler_output=features)) is features
    with pytest.raises(RuntimeError, match='projected CLIP'):
        projected_clip_output(np.ones((1, 5, 8)))


@pytest.mark.parametrize('purpose,budget,temperature,top_k', [
    ('generation', 16384, 1.0, 20), ('cleaner', 16384, 0.7, 20), ('verifier', 512, 0.000001, 1),
])
def test_sampling_has_role_specific_budgets(purpose, budget, temperature, top_k):
    serving = VisualCoTAPIServing('model', 'Qwen3.5-27B', 'http://localhost/v1',
                              purpose=purpose, transport=lambda p: {})
    actual = serving.sampling()
    assert (actual['max_tokens'], actual['temperature'], actual['top_k']) == (budget, temperature, top_k)
    assert actual['chat_template_kwargs']['enable_thinking'] is False
