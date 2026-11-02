"""Recorded requests retain ordering; opted-in budgets reap HTTP workers."""
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from dataflow.serving import APILLMServing_request
from dataflow.serving import api_llm_serving_request as api_module

pytestmark = pytest.mark.cpu

@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv('TEST_DF_API_KEY', 'test-only-key')
    return lambda **kwargs: APILLMServing_request(key_name_of_api_key='TEST_DF_API_KEY', model_name='fake', **kwargs)

@pytest.fixture
def local_api():
    bodies = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            bodies.append(body)
            text = body['messages'][-1]['content']
            if text == 'retry':
                self.send_response(429)
                self.end_headers()
                return
            data = json.dumps({'choices':[{'message':{'content':text.upper()}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data) + (30 if text == 'slow' else 0)))
            self.end_headers()
            try:
                if text == 'slow':
                    for _ in range(30):
                        self.wfile.write(b' ')
                        self.wfile.flush()
                        time.sleep(.05)
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', bodies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_recorded_results_follow_input_not_completion_order(api, monkeypatch):
    serving = api(record_results=True, max_workers=3)
    def request(id, **kwargs):
        serving._note_result(attempts=2, error_code=None)
        if id == 0:
            time.sleep(.02)
        if id == 1:
            serving._note_result(error_code='connection_error')
            raise RuntimeError('expected')
        return id, str(id)
    monkeypatch.setattr(serving, '_api_chat_id_retry', request)
    assert serving.generate_from_input(['a','b','c']) == ['0',None,'2']
    assert [event['index'] for event in serving.last_results] == [0,1,2]
    assert [event['attempts'] for event in serving.last_results] == [2,2,2]
    assert serving.last_results[1]['error_code'] == 'connection_error'
    assert 'test-only-key' not in json.dumps(serving.last_results)
    serving.cleanup()


def test_budget_healthy_requests_and_private_options_not_sent(api, local_api):
    url, bodies = local_api
    serving = api(api_url=url, request_budget_seconds=2, max_retries=1)
    assert serving.generate_from_input(['hello','你好']) == ['HELLO','你好']
    assert all(event['status'] == 'success' for event in serving.last_results)
    assert all(event['attempts'] == 1 for event in serving.last_results)
    assert all('request_budget_seconds' not in body and 'record_results' not in body for body in bodies)
    serving.cleanup()


def test_budget_covers_keepalive_and_retry_wait_and_reaps_children(api, local_api, monkeypatch):
    url, bodies = local_api
    children = []
    original = subprocess.Popen
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(api_module.subprocess, 'Popen', spawn)
    serving = api(api_url=url, request_budget_seconds=.5, read_timeout=3, max_workers=2, max_retries=4)
    for _ in range(2):
        started = time.monotonic()
        assert serving.generate_from_input(['slow', 'retry']) == [None, None]
        assert time.monotonic() - started < 2
        assert all(event['error_code'] == 'request_deadline' for event in serving.last_results)
        assert all(event['attempts'] == 1 for event in serving.last_results)
        assert all(child.poll() is not None for child in children)
    assert len(children) == 4
    serving.request_budget_seconds = 2
    assert serving.generate_from_input(['healthy']) == ['HEALTHY']
    assert all(child.poll() is not None for child in children)
    serving.cleanup()

@pytest.mark.parametrize('budget', [0, -1, float('nan'), float('inf')])
def test_invalid_budget_is_rejected(api, budget):
    with pytest.raises(ValueError, match='finite and positive'):
        api(request_budget_seconds=budget)


@pytest.fixture
def scripted_api():
    """Use real HTTP with a deterministic sequence, without patching retries."""
    state = {'responses': [], 'requests': []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            state['requests'].append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            status, value = state['responses'].pop(0)
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def completion(content, **extra):
    return {'choices': [{'message': {'content': content, **extra}}]}


@pytest.mark.parametrize('first', [completion(None), completion(''), completion(' \n'), {'choices': []}, {}])
def test_budget_retries_empty_or_malformed_completion(api, scripted_api, first):
    url, state = scripted_api
    state['responses'] = [(200, first), (200, completion('success-after-retry'))]
    serving = api(api_url=url, request_budget_seconds=5, max_retries=2, max_workers=1)
    try:
        assert serving.generate_from_input(['question']) == ['success-after-retry']
        assert len(state['requests']) == 2
        assert serving.last_results[0]['attempts'] == 2
        assert serving.last_results[0]['error_code'] is None
    finally:
        serving.cleanup()


def test_budget_keeps_one_retry_budget_across_http_error_and_empty_response(api, scripted_api):
    url, state = scripted_api
    state['responses'] = [(500, {}), (200, completion(None)), (200, completion('done'))]
    serving = api(api_url=url, request_budget_seconds=6, max_retries=3, max_workers=1)
    try:
        assert serving.generate_from_input(['question']) == ['done']
        assert len(state['requests']) == 3
        assert serving.last_results[0]['attempts'] == 3
    finally:
        serving.cleanup()


def test_budget_exhausted_empty_responses_report_attempts_and_last_reason(api, scripted_api):
    url, state = scripted_api
    state['responses'] = [(200, completion(None)), (200, completion(None))]
    serving = api(api_url=url, request_budget_seconds=5, max_retries=2, max_workers=1)
    try:
        assert serving.generate_from_input(['question']) == [None]
        assert len(state['requests']) == 2
        assert serving.last_results[0]['attempts'] == 2
        assert serving.last_results[0]['error_code'] == 'empty_response'
    finally:
        serving.cleanup()


@pytest.mark.parametrize('value,is_embedding,expected', [
    (completion('text'), False, 'text'),
    (completion(None), False, None),
    (completion(None, reasoning_content='reason'), False, '<think>reason</think>\n<answer></answer>'),
    (completion('<think>R</think><answer>A</answer>', reasoning_content='ignored'), False, '<think>R</think><answer>A</answer>'),
    ({'data': [{'embedding': [0.1, 0.2]}]}, True, [0.1, 0.2]),
])
def test_shared_formatter_preserves_existing_return_values(api, value, is_embedding, expected):
    serving = api()
    try:
        assert serving.format_response(value, is_embedding) == expected
    finally:
        serving.cleanup()


def test_budget_embedding_and_reasoning_response_use_the_same_formatter(api, scripted_api):
    url, state = scripted_api
    state['responses'] = [
        (200, {'data': [{'embedding': [0.1, 0.2]}]}),
        (200, completion(None, reasoning_content='reason')),
    ]
    serving = api(api_url=url, request_budget_seconds=5, max_retries=1, max_workers=1)
    try:
        assert serving.generate_embedding_from_input(['text']) == [[0.1, 0.2]]
        assert serving.generate_from_input(['question']) == ['<think>reason</think>\n<answer></answer>']
        assert len(state['requests']) == 2
    finally:
        serving.cleanup()
