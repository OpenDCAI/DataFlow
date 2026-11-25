"""HTTP serving for generation, verification and answer likelihood."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import re
import hashlib
from pathlib import Path
import time
import urllib.error
import urllib.request

from dataflow.core import LLMServingABC
from .model_protocol import answer_span, answer_token_logprobs, encode_image, prompt_ids
from dataflow.prompts.chinese_visual_cot import ChineseVisualCoTPrompt


class VisualCoTAPIServing(LLMServingABC):
    mocked = False

    def __init__(self, model_name, source_model, api_url, purpose='generation',
                 key_name_of_api_key='VISUAL_COT_API_KEY', revision='unverified', timeout=1800,
                 max_retries=3, transport=None):
        if purpose not in {'generation', 'cleaner', 'verifier'}:
            raise ValueError('Invalid serving purpose')
        if not api_url.startswith(('http://', 'https://')):
            raise ValueError('HTTP(S) model endpoint required')
        self.model_name, self.source_model = model_name, source_model
        self.api_url, self.purpose = api_url.rstrip('/'), purpose
        self.revision, self.timeout, self.max_retries = revision, timeout, max_retries
        if timeout <= 0 or max_retries < 1:
            raise ValueError('Positive timeout and retry count required')
        self.api_key = os.environ.get(key_name_of_api_key)
        if not self.api_key and transport is None:
            raise ValueError('Missing API key environment variable: ' + key_name_of_api_key)
        self.transport = transport
        self.runtime_manifest = dict(purpose=purpose, api_url=self.api_url, model_name=model_name,
                                     sampling=self.sampling(), actual_loaded_weights_verified=False,
                                     adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def sampling(self):
        if self.purpose == 'verifier':
            return dict(temperature=0.000001, top_p=0.9, top_k=1, max_tokens=512,
                        chat_template_kwargs={'enable_thinking': False})
        return dict(temperature=0.7 if self.purpose == 'cleaner' else 1.0,
                    top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
                    repetition_penalty=1.0, max_tokens=16384,
                    chat_template_kwargs={'enable_thinking': False})

    def start_serving(self):
        pass

    def cleanup(self):
        pass

    def _post(self, payload):
        if self.transport is not None:
            result = self.transport(deepcopy(payload))
        else:
            result = self._post_http(payload)
        if not isinstance(result, dict) or result.get('error') is not None:
            raise RuntimeError('Model service returned an error or invalid response: ' + self.model_name)
        return result

    def _post_http(self, payload):
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                self.api_url + '/chat/completions',
                data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                headers={'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'},
                method='POST')
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as result:
                    return json.loads(result.read())
            except urllib.error.HTTPError as exc:
                error = 'HTTP ' + str(exc.code)
                exc.close()
            except (urllib.error.URLError, TimeoutError, ValueError):
                error = 'network/JSON failure'
            if attempt + 1 < self.max_retries:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError('Model service failed: ' + error + '; model=' + self.model_name)

    @staticmethod
    def _content(prompt, images):
        content = [{'type': 'text', 'text': prompt}]
        for path in images:
            content.append({'type': 'image_url', 'image_url': {'url': encode_image(path)}})
        return content

    def complete(self, *, prompt, images, phase):
        if self.purpose == 'verifier':
            if images:
                raise ValueError('The verifier takes text only')
            messages = [{'role': 'system', 'content': ChineseVisualCoTPrompt().build_system_prompt('verification')},
                        {'role': 'user', 'content': prompt}]
        else:
            messages = [{'role': 'user', 'content': self._content(prompt, images)}]
        payload = dict(model=self.model_name, messages=messages, **self.sampling())
        result = self._post(payload)
        choices = result.get('choices')
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise RuntimeError('Expected a single completion from ' + self.model_name)
        choice = choices[0]
        if choice.get('finish_reason') != 'stop':
            raise RuntimeError(f"Incomplete model reply in {phase}: finish_reason={choice.get('finish_reason')!r}")
        message = choice.get('message')
        if not isinstance(message, dict) or message.get('refusal'):
            raise RuntimeError('Missing or refused model message in ' + phase)
        return dict(text=message.get('content') or '',
                    reasoning=message.get('reasoning_content') or message.get('reasoning') or '',
                    finish_reason=choice.get('finish_reason', ''), phase=phase)

    def score_answer(self, *, prompt, images, response):
        prefix, target = answer_span(response)
        content = self._content(prompt, images)
        common = dict(model=self.model_name, temperature=0.0, max_tokens=1,
                      prompt_logprobs=1, return_token_ids=True,
                      continue_final_message=True, add_generation_prompt=False,
                      chat_template_kwargs={'preserve_thinking': True, 'enable_thinking': True})
        results = []
        for assistant in [prefix, prefix + target]:
            message = {'role': 'assistant', 'content': assistant}
            thinking = re.fullmatch(r'\s*<think>(.*?)</think>\s*(.*)', assistant, re.I | re.S)
            if thinking:
                message.update(reasoning_content=thinking.group(1).strip(), content=thinking.group(2))
            result = self._post(dict(common, messages=[{'role': 'user', 'content': content}, message]))
            # The generated next token is discarded. Reaching max_tokens=1 is normal;
            # a backend error or filtered completion is not a usable scoring result.
            if 'choices' in result:
                choices = result['choices']
                if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                        or choices[0].get('finish_reason') not in ('stop', 'length')):
                    raise RuntimeError('Invalid scoring choice or finish_reason from ' + self.model_name)
            results.append(result)
        prefix_ids, full_ids = [prompt_ids(reply) for reply in results]
        start, selected = answer_token_logprobs(prefix_ids, full_ids, results[1])
        total = float(sum(selected))
        return dict(answer_logprob_sum=total, answer_token_count=len(selected),
                    answer_logprob_mean=total / len(selected), answer_target=target,
                    prefix_token_count=len(prefix_ids), full_token_count=len(full_ids),
                    target_start_token=start)

    def generate_from_input(self, user_inputs, system_prompt=''):
        return [self.complete(prompt=system_prompt + '\n' + text, images=[], phase='text')['text']
                for text in user_inputs]

    def generate_from_input_multi_images(self, list_of_image_paths, list_of_image_labels,
                                         system_prompt='', user_prompts=None):
        prompts = user_prompts or [''] * len(list_of_image_paths)
        if len(prompts) != len(list_of_image_paths):
            raise ValueError('Mismatched batch lengths')
        return [self.complete(prompt=system_prompt + '\n' + prompt, images=images, phase='vision')['text']
                for images, prompt in zip(list_of_image_paths, prompts)]
