"""Generate and select Chinese CoT from image question-answer pairs."""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from .storage import publish_text, read_qa_frame
from .model_protocol import normalize_completion, parse_verdict
from .language import has_chinese_text, is_chinese_reasoning
from .contracts import VisualCoTImageFilter, source_model
from dataflow.prompts import chinese_visual_cot as prompts
from dataflow.prompts.chinese_visual_cot import ChineseVisualCoTPrompt
from .serving import VisualCoTServingBundle


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identity_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [identity_value(v) for v in value]
    return repr(value)


def source_label(row, key):
    value = row.get(key)
    if value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value))):
        return None
    return str(value).strip() or None


def last_json(text):
    """Cleaning prompts specify a JSON object on the last line."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        value = json.loads(text.strip().splitlines()[-1])
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def trajectory_parts(response):
    tags = re.findall(r'</?(?:think|answer)>', response.lower())
    if tags != ['<think>', '</think>', '<answer>', '</answer>']:
        return None
    match = re.fullmatch(
        r'\s*<think>(.*?)</think>\s*(?:therefore,\s*the final answer is\s*|(?:因此[，,]?\s*)?最终答案是[：:]?\s*)?'
        r'<answer>(.*?)</answer>\.?\s*', response, re.I | re.S)
    if match:
        reasoning, answer = (part.strip() for part in match.groups())
        if reasoning and answer:
            return reasoning, answer
    return None


@OPERATOR_REGISTRY.register()
class ChineseVisualCoTGenerator(OperatorABC):
    """Generate Chinese CoT with difficulty filtering and answer-likelihood selection.

    Services expose complete(prompt, images, phase) and teachers additionally
    score_answer(prompt, images, response). Inputs to the verifier are text only.
    The injected bundle exposes lifecycle to the framework. No training or server launch is performed.
    """
    def __init__(self, llm_serving: VisualCoTServingBundle, image_filter: VisualCoTImageFilter,
                 image_root: str | Path = '.', journal_dir: str | Path | None = None,
                 prompt_template: ChineseVisualCoTPrompt | None = None):
        super().__init__()
        if not isinstance(llm_serving, VisualCoTServingBundle):
            raise TypeError('llm_serving must be an VisualCoTServingBundle')
        self.llm_serving = llm_serving
        self.image_filter = image_filter
        self.prompt_template = prompt_template or ChineseVisualCoTPrompt()
        if not isinstance(self.prompt_template, ChineseVisualCoTPrompt):
            raise TypeError('prompt_template must be ChineseVisualCoTPrompt')
        if not image_filter:
            raise ValueError('Raw cleaning filter is mandatory; no silent omission')
        self.image_root = Path(image_root).resolve()
        self.journal_dir = Path(journal_dir).resolve() if journal_dir else None
        self.last_summary = {}
        support_dir = Path(__file__).parent
        implementation_files = [
            Path(__file__), Path(prompts.__file__),
            *(support_dir / name for name in ('preprocessing.py', 'language.py', 'model_protocol.py',
                                              'api_serving.py', 'serving.py', 'contracts.py', 'storage.py')),
        ]
        self.manifest = {
            'protocol': 'visual-cot-zh-v2',
            'target_language': 'zh-CN',
            'implementation_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in implementation_files},
            'rules': ['4B four rollouts; retain verified 0/4',
                      '9B -> 27B -> 122B-A10B; four independent rollouts per teacher',
                      'first teacher with a correct candidate', '27B text-only correctness/consistency verification',
                      'format filtering after correctness', 'normalized answer-token likelihood; stable rollout-id tie break',
                      'first schema-valid cleaning response of four'],
            'raw_filter': image_filter.manifest(),
            'limitations': [
                'The text verifier judges the response against the question and reference answer',
                'Image-filter thresholds are configurable heuristics',
                'API model identity/revision is declared configuration until verified against the actual serving process',
            ],
            'services': [dict(role=role, source_model=source_model(s),
                              revision=getattr(s, 'revision', 'unverified'),
                              runtime=getattr(s, 'runtime_manifest', {}))
                         for role, s in [('small', self.small), *[(f'teacher_{i}', s) for i, s in enumerate(self.teachers)],
                                         ('verifier', self.verifier), ('cleaner', self.cleaner)]],
            'mocked': any(bool(getattr(s, 'mocked', False)) for s in
                          [self.small, *self.teachers, self.verifier, self.cleaner])
                      or image_filter.manifest().get('mocked', False),
        }

    @property
    def small(self):
        return self.llm_serving.small_model

    @property
    def teachers(self):
        return self.llm_serving.teacher_servings

    @property
    def verifier(self):
        return self.llm_serving.verifier_serving

    @property
    def cleaner(self):
        return self.llm_serving.cleaner_serving

    @staticmethod
    def get_desc(lang: str = 'zh') -> str:
        if lang == 'en':
            return (
                "Chinese image VQA to Chinese CoT using a 4B difficulty filter, "
                "9B/27B/122B cascade, text-only 27B verification, Chinese reasoning screening and answer-token likelihood. "
                "__init__: llm_serving (required VisualCoTServingBundle with all model roles); "
                "image_filter (required stateful image-quality/duplicate filter); image_root='.' "
                "(base for local relative image paths); journal_dir=None (optional per-row resumable "
                "JSON cache, rejects changed input/config/implementation); prompt_template=None "
                "(registered ChineseVisualCoTPrompt default). run: storage (DataFlowStorage); "
                "input_image_key='images' (one local path or singleton path list); "
                "input_question_key='question' (nonempty Chinese string); input_answer_key='answer' "
                "(nonempty reference answer); output_key='visual_cot_result' (new dict column). "
                "Preserves all input rows, columns and order. Output status is accepted, rejected or "
                "invalid_input; reason and mocked are always present. Processed rows include audit, "
                "raw_filter and protocol; accepted rows additionally include cleaned question, "
                "reference_answer, domain, type, subtype, images, reasoning, answer, response, "
                "messages, teacher, teacher_index, difficulty, selected_rollout_id, candidate_scores "
                "and selection_method. Missing columns/output collision raise ValueError; invalid "
                "individual inputs are marked, service/scoring failures abort for retry. Returns the "
                "output column name (str); writes one storage step and updates last_summary."
            )
        if lang != 'zh':
            raise ValueError("lang must be 'zh' or 'en'")
        return (
            "从中文图文问答生成中文CoT。采用4B难度筛选、9B/27B/122B级联、27B纯文本核验和答案token似然择优。"
            "__init__：llm_serving（必填VisualCoTServingBundle，包含所有模型角色）；image_filter（必填有状态"
            "图片质量及去重过滤器）；image_root='.'（相对图片路径基准）；journal_dir=None（可选逐行JSON"
            "断点缓存，输入/配置/实现改变则拒绝复用）；prompt_template=None（默认注册的ChineseVisualCoTPrompt）。"
            "run：storage（DataFlowStorage）；input_image_key='images'（单张本地路径或单元素路径列表）；"
            "input_question_key='question'（非空中文字符串）；input_answer_key='answer'（非空参考答案）；"
            "output_key='visual_cot_result'（新字典列）。保留全部输入行、列及顺序。输出status为accepted、"
            "rejected或invalid_input，始终有reason和mocked；已处理行有audit、raw_filter和protocol；"
            "接受行还含清洗后的question、reference_answer、domain、type、subtype、images、reasoning、"
            "answer、response、messages、teacher、teacher_index、difficulty、selected_rollout_id、"
            "candidate_scores、selection_method。缺列/输出列冲突抛ValueError，单行无效输入标记状态，"
            "服务或评分错误中止以便重试。返回输出列名str；写一个Storage步骤并更新last_summary。"
        )

    def _image(self, value):
        paths = [value] if isinstance(value, str) else value
        if not isinstance(paths, (list, tuple)) or len(paths) != 1 or not isinstance(paths[0], str):
            raise ValueError('Expected one image per VQA row')
        path = Path(paths[0])
        path = (self.image_root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise ValueError('Missing image file: ' + str(path))
        return str(path)

    def _complete(self, service, prompt, images, phase, audit):
        value = service.complete(prompt=prompt, images=images, phase=phase)
        if not isinstance(value, dict) or not isinstance(value.get('text'), str) or not value['text'].strip():
            raise RuntimeError('Empty/invalid model-service response in ' + phase)
        # Standard DataFlow text-only wrappers do not expose completion metadata.
        # An explicit failure, however, cannot establish a valid draw or verdict.
        finish = value.get('finish_reason', 'unavailable_from_standard_interface')
        if finish not in ('stop', 'unavailable_from_standard_interface'):
            raise RuntimeError(f'Incomplete model reply in {phase}: finish_reason={finish!r}')
        audit.append({'phase': phase, 'model': source_model(service), **value})
        return value

    def _clean(self, row, images, audit):
        question, answer = str(row['question']), str(row['reference_answer'])
        stages = [('language_standardization', 'standardized_question'),
                  ('irrelevant_content_removal', 'corrected text'),
                  ('answerability_filtering', 'answerable'),
                  ('instruction_refinement', 'refined_question'),
                  ('answer_extraction', 'extracted_answer'),
                  ('answer_canonicalization', 'canonical_answer')]
        schemas = {
            'language_standardization': {'standardized_question': str, 'is_translated': bool, 'reason': str},
            'irrelevant_content_removal': {'corrected text': str, 'error type': list, 'reason': str},
            'answerability_filtering': {'answerable': bool, 'reason': str},
            'instruction_refinement': {'refined_question': str, 'has_low_quality_instruction': bool,
                                       'rewritten_instruction': str, 'reason': str},
            'answer_extraction': {'extracted_answer': str, 'has_reasoning_or_noise': bool, 'reason': str},
            'answer_canonicalization': {'canonical_answer': str, 'answer_type': str,
                                        'recovered_option_label': str, 'reason': str},
        }
        answer_type = source_label(row, 'type')
        for phase, field in stages:
            prompt = self.prompt_template.build_prompt(phase, question, answer)
            candidates = [self._complete(self.cleaner, prompt, images, phase, audit)['text'] for _ in range(4)]
            parsed = None
            for text in candidates:
                try:
                    candidate = last_json(text)
                    schema = schemas[phase]
                    if set(candidate) != set(schema) or any(type(candidate[k]) is not t for k, t in schema.items()):
                        continue
                    value = candidate.get(field)
                    if field == 'answerable':
                        if type(value) is not bool:
                            continue
                    elif not isinstance(value, str) or not value.strip():
                        continue
                    parsed = candidate
                    break
                except (ValueError, IndexError):
                    continue
            if parsed is None:
                raise RuntimeError('No schema-valid cleaning result in ' + phase)
            audit.append(dict(phase=phase + '_selection', selection='first_schema_valid_of_four', parsed=parsed))
            if field == 'answerable':
                if not parsed[field]:
                    return None
            elif phase in ('answer_extraction', 'answer_canonicalization'):
                answer = parsed[field].strip()
                answer_type = source_label(row, 'type') or parsed.get('answer_type') or answer_type
            else:
                question = parsed[field].strip()
        domain_prompt = self.prompt_template.build_prompt('domain_annotation', question, answer)
        domain = last_json(self._complete(self.cleaner, domain_prompt, images, 'domain_annotation', audit)['text'])
        allowed = {'science', 'math', 'logic/game/puzzle', 'chart/table/doc', 'general', 'spatial', 'gui/web'}
        if set(domain) != {'domain'} or domain.get('domain') not in allowed:
            raise RuntimeError('Invalid domain annotation')
        return dict(question=question, reference_answer=answer, domain=domain['domain'],
                    type=answer_type or 'unspecified', subtype=source_label(row, 'subtype'))

    def _verify(self, question, reference, responses, audit, phase):
        verdicts = []
        for i, response in enumerate(responses):
            prompt = self.prompt_template.build_prompt('verification', question, reference, response)
            value = self._complete(self.verifier, prompt, [], phase + f'_verify_{i}', audit)
            verdict = parse_verdict(value['text'])
            if verdict is None:
                raise RuntimeError('Unparseable verifier response; not a verified incorrect answer')
            verdicts.append(verdict)
        return verdicts

    def _rollouts(self, service, prompt, images, phase, audit):
        outputs = [self._complete(service, prompt, images, phase + f'_rollout_{i}', audit) for i in range(4)]
        result = []
        for value in outputs:
            response = value['text']
            if value.get('reasoning'):
                response = '<think>\n' + value['reasoning'] + '\n</think>\n' + response
            elif '<think>' not in response.lower():
                ending = re.search(r'(?im)^\s*因此[，,]\s*最终答案是\s*(?=<answer>)', response)
                if ending:
                    response = f'<think>{response[:ending.start()].strip()}</think>\n{response[ending.start():].strip()}'
            result.append(normalize_completion(response))
        return result

    def _process(self, row, image, raw_decision):
        audit = []
        result = dict(status='rejected', reason=None, images=[image], audit=audit,
                      raw_filter=raw_decision, protocol=self.manifest['protocol'],
                      mocked=self.manifest['mocked'])
        if not raw_decision['keep']:
            result['reason'] = raw_decision['reason']
            return result
        clean = self._clean(row, [image], audit)
        if clean is None:
            result['reason'] = 'unanswerable_after_semantic_cleaning'
            return result
        result.update(clean)
        question, reference = clean['question'], clean['reference_answer']
        if not has_chinese_text(question):
            result['reason'] = 'cleaned_question_not_chinese'
            return result
        prompt = self.prompt_template.build_prompt('difficulty', question)
        small = self._rollouts(self.small, prompt, [image], 'difficulty', audit)
        checks = self._verify(question, reference, small, audit, 'difficulty')
        result['difficulty'] = dict(correct_count=sum(checks), rollout_count=4)
        if any(checks):
            result['reason'] = 'solved_by_4B_filter'
            return result
        prompt = self.prompt_template.build_prompt('generation', question)
        for teacher_index, teacher in enumerate(self.teachers):
            candidates = self._rollouts(teacher, prompt, [image], f'teacher_{teacher_index}', audit)
            correct = self._verify(question, reference, candidates, audit, f'teacher_{teacher_index}')
            if not any(correct):
                continue
            # Once a teacher solves the question, select from that teacher's candidates.
            formatted = [(i, parts) for i, response in enumerate(candidates)
                         if correct[i] and (parts := trajectory_parts(response))]
            result['teacher'] = source_model(teacher)
            result['teacher_index'] = teacher_index
            if not formatted:
                result['reason'] = 'first_solving_teacher_has_no_format_valid_candidate'
                return result
            valid = [(i, candidates[i]) for i, parts in formatted if is_chinese_reasoning(parts[0])]
            if not valid:
                result['reason'] = 'reasoning_not_chinese'
                return result
            scores = []
            for i, response in valid:
                score = None
                if len(valid) > 1:
                    score = teacher.score_answer(prompt=prompt, images=[image], response=response)
                    value = score.get('answer_logprob_mean') if isinstance(score, dict) else None
                    if (isinstance(value, bool) or not isinstance(value, (float, int))
                            or not math.isfinite(value) or value > 0):
                        raise RuntimeError('Actual finite answer-token log-likelihood at or below zero required')
                scores.append(dict(rollout_id=i, response=response, score=score))
            chosen = max(scores, key=lambda s: (s['score']['answer_logprob_mean'], -s['rollout_id'])) if len(scores) > 1 else scores[0]
            response = chosen['response']
            reasoning, answer = trajectory_parts(response)
            training_response = f'<think>{reasoning}</think>\n<answer>{answer}</answer>'
            result.update(status='accepted', reason='verified_and_selected', reasoning=reasoning, answer=answer,
                          response=response, selected_rollout_id=chosen['rollout_id'], candidate_scores=scores,
                          selection_method='normalized_answer_likelihood' if len(scores) > 1 else 'single_correct_rollout',
                          messages=[{'role': 'user', 'content': '<image>\n' + question},
                                    {'role': 'assistant', 'content': training_response}])
            return result
        result['reason'] = 'unsolved_by_all_three_teachers'
        return result

    def run(self, storage: DataFlowStorage, input_image_key: str = 'images',
            input_question_key: str = 'question', input_answer_key: str = 'answer',
            output_key: str = 'visual_cot_result') -> str:
        self.last_summary = {}
        lock = nullcontext()
        if self.journal_dir is not None:
            from filelock import FileLock

            self.journal_dir.parent.mkdir(parents=True, exist_ok=True)
            lock = FileLock(self.journal_dir.with_name(self.journal_dir.name + '.lock'), timeout=0)
        with lock:
            return self._run_rows(storage, input_image_key, input_question_key, input_answer_key, output_key)

    def _run_rows(self, storage, input_image_key, input_question_key, input_answer_key, output_key):
        dataframe = read_qa_frame(storage)
        required = [input_image_key, input_question_key, input_answer_key]
        if any(k not in dataframe for k in required):
            raise ValueError('Required columns: ' + ', '.join(required))
        if output_key in dataframe:
            raise ValueError('Refusing to overwrite an existing output column')
        rows = dataframe.to_dict('records')
        input_identity = []
        for row in rows:
            try:
                image = self._image(row[input_image_key])
                s = Path(image).stat()
                image_identity = [image, s.st_size, s.st_mtime_ns]
            except ValueError:
                image_identity = row[input_image_key]
            input_identity.append(identity_value([image_identity, row[input_question_key], row[input_answer_key],
                                                  source_label(row, 'type'), source_label(row, 'subtype')]))
        fingerprint = canonical_hash(dict(manifest=self.manifest, input=input_identity, keys=required))
        if self.journal_dir:
            self.journal_dir.mkdir(parents=True, exist_ok=True)
            marker = self.journal_dir / 'manifest.json'
            expected = dict(fingerprint=fingerprint, method=self.manifest)
            if marker.exists():
                if json.loads(marker.read_text(encoding='utf-8')) != expected:
                    raise ValueError('Resume manifest differs: input, image identity, model or method changed')
            else:
                if any(self.journal_dir.iterdir()):
                    raise ValueError('Refusing to resume orphaned journal without manifest')
                publish_text(marker, json.dumps(expected, ensure_ascii=False, indent=2))
        self.image_filter.reset()
        outputs = []
        for position, raw in enumerate(rows):
            checkpoint = self.journal_dir / f'{position:09d}.json' if self.journal_dir else None
            try:
                image = self._image(raw[input_image_key])
                question, answer = raw[input_question_key], raw[input_answer_key]
                if not isinstance(question, str) or not has_chinese_text(question):
                    raise ValueError('A Chinese question is required')
                missing_answer = answer is None or (pd.api.types.is_scalar(answer) and bool(pd.isna(answer)))
                if missing_answer or not str(answer).strip():
                    raise ValueError('A nonempty reference answer is required')
                row = dict(raw, question=question, reference_answer=str(answer))
            except ValueError as exc:
                result = dict(status='invalid_input', reason=str(exc), mocked=self.manifest['mocked'])
            else:
                decision = self.image_filter.evaluate(image, question, str(answer))
                if checkpoint and checkpoint.exists():
                    saved = json.loads(checkpoint.read_text(encoding='utf-8'))
                    checksum = saved.pop('journal_sha256', None)
                    if checksum != canonical_hash([fingerprint, position, saved]):
                        raise ValueError(f'Journal record {position} differs from its saved checksum')
                    if saved['raw_filter'] != decision:
                        raise RuntimeError('Raw filter replay differs; refuse stale result reuse')
                    if saved['status'] == 'accepted':
                        parts = trajectory_parts(saved['response'])
                        expected_messages = [
                            {'role': 'user', 'content': '<image>\n' + saved['question']},
                            {'role': 'assistant', 'content':
                             f"<think>{saved['reasoning']}</think>\n<answer>{saved['answer']}</answer>"},
                        ]
                        if (parts != (saved['reasoning'], saved['answer']) or not is_chinese_reasoning(parts[0])
                                or saved['images'] != [image] or saved['messages'] != expected_messages):
                            raise ValueError('Journal contains an inconsistent or non-Chinese training response')
                    result = saved
                else:
                    result = self._process(row, image, decision)
            if checkpoint and not checkpoint.exists():
                record = dict(result, journal_sha256=canonical_hash([fingerprint, position, result]))
                publish_text(checkpoint, json.dumps(record, ensure_ascii=False, allow_nan=False))
            outputs.append(result)
        dataframe[output_key] = outputs if outputs else pd.Series(index=dataframe.index, dtype=object)
        storage.write(dataframe)
        self.last_summary = dict(Counter(r['status'] for r in outputs))
        return output_key
