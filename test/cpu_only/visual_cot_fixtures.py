"""Deterministic offline model doubles; never evidence of model/data quality."""
import json
import re
import numpy as np
from PIL import Image
from dataflow.prompts.chinese_visual_cot import INPUT_MARKER


class DemoCLIP:
    model_id = 'MOCK/CLIP'
    mocked = True

    def encode(self, path):
        with Image.open(path) as image:
            color = np.asarray(image.convert('RGB'), dtype=float).mean(axis=(0, 1))
        return np.r_[color, 1.0]


class DemoModel:
    mocked = True
    revision = 'MOCK/offline-scripted-v1'

    def __init__(self, source_model, purpose='generation'):
        self.source_model = source_model
        self.purpose = purpose
        self.calls = []
        self.score_calls = []
        self.runtime_manifest = {'mocked': True, 'purpose': purpose}

    def complete(self, *, prompt, images, phase):
        self.calls.append(dict(prompt=prompt, images=images, phase=phase))
        fields = json.loads(prompt.split(INPUT_MARKER, 1)[1])
        if self.purpose == 'cleaner':
            question, reference = fields['question'], fields['reference_answer']
            outputs = {
                'language_standardization': {'standardized_question': question, 'is_translated': False},
                'irrelevant_content_removal': {'corrected text': question, 'error type': []},
                'answerability_filtering': {'answerable': True},
                'instruction_refinement': {'refined_question': question, 'has_low_quality_instruction': False,
                                           'rewritten_instruction': ''},
                'answer_extraction': {'extracted_answer': reference, 'has_reasoning_or_noise': False},
                'answer_canonicalization': {'canonical_answer': reference, 'answer_type': 'numeric',
                                             'recovered_option_label': ''},
                'domain_annotation': {'domain': 'general'},
            }
            obj = outputs[phase] if phase == 'domain_annotation' else dict(outputs[phase], reason='离线预置响应，仅验证数据流')
            return {'text': json.dumps(obj, ensure_ascii=False), 'finish_reason': 'stop'}
        if self.purpose == 'verifier':
            assert not images, 'Verifier must not receive images'
            reference, candidate = fields['reference_answer'], fields['candidate_response']
            matches = re.findall(r'<answer>(.*?)</answer>', candidate, re.S)
            correct = bool(matches and matches[-1].strip() == reference.strip())
            return {'text': json.dumps({'verdict': 'correct' if correct else 'incorrect',
                                       'reasoning_consistent': correct, 'reason': '离线预置核验'}, ensure_ascii=False)}
        i = int(phase.rsplit('_', 1)[-1])
        question = fields['question']
        if phase.startswith('difficulty'):
            answer = '3' if '左侧' in question and i == 0 else '0'
        elif self.source_model.endswith('9B'):
            answer = '3' if '红色' in question and i in (1, 2) else '0'
        elif self.source_model.endswith('27B'):
            answer = '2' if '蓝色' in question and i == 0 else '0'
        else:
            answer = '0'
        return {'text': f'<think>离线候选{i}：根据图形逐一计数，按题意得出数量。</think>\n<answer>{answer}</answer>',
                'finish_reason': 'stop'}

    def score_answer(self, *, prompt, images, response):
        self.score_calls.append(dict(prompt=prompt, images=images, response=response))
        i = int(re.search(r'候选(\d+)', response).group(1))
        value = -0.9 + 0.2 * i
        return dict(answer_logprob_mean=value, answer_logprob_sum=value * 2,
                    answer_token_count=2, mocked=True)


def make_demo(folder):
    from PIL import Image, ImageDraw
    source = folder / 'demo_input.jsonl'
    if source.exists():
        return source
    image_dir = folder / 'images'
    image_dir.mkdir(exist_ok=True)
    for name, count, color in [('three.png', 3, 'red'), ('two.png', 2, 'blue')]:
        image = Image.new('RGB', (240, 120), 'white')
        draw = ImageDraw.Draw(image)
        for i in range(count):
            draw.rectangle((10 + 75 * i, 20, 60 + 75 * i, 90), fill=color)
        image.save(image_dir / name)
    records = [
        {'id': 'demo-1', 'images': ['images/three.png'], 'question': '图中红色方块有几个？', 'answer': '3'},
        {'id': 'demo-2', 'images': ['images/two.png'], 'question': '蓝色长方形的数量是多少？', 'answer': '2'},
        {'id': 'demo-3', 'images': ['images/three.png'], 'question': '左侧图形应如何计数？', 'answer': '3'},
        {'id': 'demo-4', 'images': ['images/three.png'], 'question': '图中红色方块有几个？', 'answer': '3'},
        {'id': 'demo-5', 'images': ['images/two.png'], 'question': '只有图片没有参考答案的样本', 'answer': ''},
    ]
    source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    return source
