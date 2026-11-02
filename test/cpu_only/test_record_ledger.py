"""Failures/replay are source-linked, ordered and idempotent."""
import json
from pathlib import Path
import pytest
from dataflow.utils.record_ledger import RecordLedger, RecordProcessingError, RecordResult
from dataflow.utils.storage import FileStorage, BatchedFileStorage, StreamBatchedFileStorage
from dataflow.utils.record_execution import RecordedGenerationRunner

pytestmark = pytest.mark.cpu

@pytest.fixture
def source(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text(''.join(json.dumps({'id': i, 'raw_content': f'q{i}', 'image': 'assets/a.png'}) + '\n' for i in range(100)))
    return source

def test_100_records_five_failures_only_five_replayed_and_merge_idempotent(tmp_path, source):
    called = []
    def first(row):
        called.append(row['id'])
        if row['id'] % 20 == 0:
            raise RecordProcessingError('read_timeout', 'request', attempts=2)
        return RecordResult({'result': row['id'] * 2}, attempts=1)
    state = tmp_path / 'state'
    with RecordLedger(state, source, {'model': 'fake', 'prompt': 'v1'}) as ledger:
        ledger.process(first)
        assert ledger.report() == {'schema_version': 1, 'identity': ledger.identity, 'input': 100, 'success': 95, 'failed': 5, 'skipped': 0, 'pending': 0, 'attempts': 105, 'complete': True}
        ledger.export_results(tmp_path / 'first.jsonl')
        assert len((tmp_path / 'first.jsonl').read_text().splitlines()) == 95
        failures = [json.loads(line) for line in (state / 'failures.jsonl').read_text().splitlines()]
        assert [row['line'] for row in failures] == [1, 21, 41, 61, 81]
        assert all(row['resource_base'] == str(source.parent) for row in failures)
        assert 'q20' not in (state / 'failures.jsonl').read_text()
        called.clear()
        def retry(row):
            called.append(row['id'])
            return {'result': row['id'] * 2}
        ledger.process(retry, only_failed=True)
        assert called == [0, 20, 40, 60, 80]
        ledger.export_results(tmp_path / 'merged.jsonl')
        actual = (tmp_path / 'merged.jsonl').read_bytes()
        assert [row['id'] for row in map(json.loads, actual.splitlines())] == list(range(100))
        assert [row['result'] for row in map(json.loads, actual.splitlines())] == [i * 2 for i in range(100)]
        called.clear()
        ledger.process(retry, only_failed=True)
        ledger.export_results(tmp_path / 'merged.jsonl')
        assert called == []
        assert (tmp_path / 'merged.jsonl').read_bytes() == actual
        assert ledger.report()['attempts'] == 110
        assert (state / 'failures.jsonl').read_text() == ''

def test_repeated_failure_and_programming_errors_remain_visible(tmp_path, source):
    with RecordLedger(tmp_path / 'state', source, {}) as ledger:
        def fail(row):
            raise RecordProcessingError('parse_error', 'parse')
        ledger.process(fail)
        ledger.process(fail, only_failed=True)
        assert ledger.report()['failed'] == 100
        assert ledger.report()['attempts'] == 200
        def bug(row):
            raise RuntimeError('programming bug')
        with pytest.raises(RuntimeError, match='programming bug'):
            ledger.process(bug, only_failed=True)
        assert ledger.report()['attempts'] == 200

def test_reopen_refuses_changed_source_or_configuration(tmp_path, source):
    state = tmp_path / 'state'
    with RecordLedger(state, source, {'prompt': 'v1'}):
        pass
    with pytest.raises(ValueError, match='changed'):
        RecordLedger(state, source, {'prompt': 'v2'})
    source.write_text(source.read_text() + '{}\n')
    with pytest.raises(ValueError, match='changed'):
        RecordLedger(state, source, {'prompt': 'v1'})

def test_interruption_reuses_successful_records(tmp_path, source):
    called = []
    with RecordLedger(tmp_path / 'state', source, {}) as ledger:
        def interrupt(row):
            if row['id'] == 3:
                raise RuntimeError('stop')
            return {'result': row['id']}
        with pytest.raises(RuntimeError):
            ledger.process(interrupt)
        assert ledger.report()['pending'] == 97
    with RecordLedger(tmp_path / 'state', source, {}) as ledger:
        def resume(row):
            called.append(row['id'])
            return {'result': row['id']}
        ledger.process(resume)
        assert called == list(range(3, 100))
        assert ledger.report()['success'] == 100

class FakeServing:
    record_results = True
    model_name = 'fake'
    api_url = 'http://localhost'
    max_retries = 1
    configs = {}
    def __init__(self):
        self.failed = True
        self.calls = []
    def generate_from_input(self, texts, system_prompt):
        text = texts[0]
        self.calls.append(text)
        fail = self.failed and text in {'q0','q20','q40','q60','q80'}
        self.last_results = [{'index': 0, 'attempts': 1, 'error_code': 'read_timeout' if fail else None}]
        return [None if fail else text.upper()]

def test_unicode_ledger_identity_reopens_with_non_utf8_default(tmp_path, source, monkeypatch):
    original = Path.read_text

    def windows_read(path, encoding=None, errors=None):
        return original(path, encoding=encoding or 'cp1252', errors=errors)

    monkeypatch.setattr(Path, 'read_text', windows_read)
    unicode_source = tmp_path / '中文输入.jsonl'
    unicode_source.write_bytes(source.read_bytes())
    directory = tmp_path / '中文账本'
    with RecordLedger(directory, unicode_source, {'revision': 1}) as ledger:
        ledger.process(lambda row: {'result': row['id']})
    with RecordLedger(directory, unicode_source, {'revision': 1}) as ledger:
        assert ledger.report()['success'] == 100


def test_runner_aligned_failures_and_explicit_success_view(tmp_path, source):
    serving=FakeServing();runner=RecordedGenerationRunner(serving,tmp_path/'ledger',{'parser_revision':1})
    success=tmp_path/'success.jsonl';outcomes=tmp_path/'aligned.jsonl'
    report=runner.run(source,success,outcomes_output=outcomes)
    assert (report['success'],report['failed'])==(95,5)
    rows=list(map(json.loads,outcomes.read_text().splitlines()))
    assert [row['id'] for row in rows]==list(range(100))
    assert sum(row['generation_outcome']['status']=='failed' for row in rows)==5
    assert all(row['generated_content'] is None for row in rows if row['id']%20==0)
    assert len(success.read_text().splitlines())==95
    serving.failed=False;serving.calls.clear()
    runner.run(source,success,only_failed=True,outcomes_output=outcomes)
    assert serving.calls==['q0','q20','q40','q60','q80']
    before=outcomes.read_bytes();serving.calls.clear()
    runner.run(source,success,only_failed=True,outcomes_output=outcomes)
    assert serving.calls==[] and outcomes.read_bytes()==before
    assert len(success.read_text().splitlines())==100


def test_parser_expected_error_is_recorded_but_programming_error_escapes(tmp_path):
    source=tmp_path/'input.jsonl';source.write_text('{"raw_content":"secret"}\n')
    runner=RecordedGenerationRunner(FakeServing(),tmp_path/'ledger',{},parse_response=json.loads)
    runner.run(source,tmp_path/'success.jsonl')
    failure=json.loads((tmp_path/'ledger/failures.jsonl').read_text())
    assert (failure['stage'],failure['code'])==('parse','invalid_model_output')
    assert 'secret' not in (tmp_path/'ledger/failures.jsonl').read_text()
    def bug(value):raise TypeError('parser programming error')
    runner=RecordedGenerationRunner(FakeServing(),tmp_path/'bug-ledger',{},parse_response=bug)
    with pytest.raises(TypeError,match='programming error'):runner.run(source,tmp_path/'other.jsonl')

@pytest.mark.parametrize('count',[0,1,3])
def test_empty_objects_and_empty_stage_have_terminal_reports(tmp_path,count):
    source=tmp_path/'input.jsonl';source.write_text('{}\n'*count)
    serving=FakeServing();runner=RecordedGenerationRunner(serving,tmp_path/'ledger',{})
    report=runner.run(source,tmp_path/'success.jsonl')
    assert report['input']==count and report['failed']==count and report['pending']==0 and report['complete']
    assert serving.calls==[]
    rows=list(map(json.loads,(tmp_path/'success.outcomes.jsonl').read_text().splitlines()))
    assert [row['generation_outcome']['line'] for row in rows]==list(range(1,count+1))
    assert all(row['generation_outcome']['code']=='invalid_input' for row in rows)


def test_input_output_status_collisions_fail_before_model_calls(tmp_path):
    source=tmp_path/'input.jsonl';source.write_text('{"raw_content":"q","generated_content":"keep"}\n')
    serving=FakeServing();runner=RecordedGenerationRunner(serving,tmp_path/'ledger',{})
    for target in [source,tmp_path/'success.jsonl',tmp_path/'ledger/output.jsonl']:
        with pytest.raises(ValueError):runner.run(source,target)
    assert serving.calls==[] and source.read_text().endswith('"keep"}\n')


def test_physical_source_identity_survives_bad_lines_and_duplicate_content(tmp_path):
    from dataflow.utils.jsonl_reader import JSONLReadPolicy
    source=tmp_path/'input.jsonl';source.write_text('bad\n{"raw_content":"q0"}\n{"raw_content":"q0"}\n{"raw_content":"q1"}\n')
    serving=FakeServing();runner=RecordedGenerationRunner(serving,tmp_path/'ledger',{})
    policy=JSONLReadPolicy(tmp_path/'read-errors.jsonl')
    report=runner.run(source,tmp_path/'success.jsonl',read_policy=policy)
    assert (report['input'],report['failed'])==(3,2)
    rows=list(map(json.loads,(tmp_path/'success.outcomes.jsonl').read_text().splitlines()))
    assert [row['generation_outcome']['line'] for row in rows]==[2,3,4]
    serving.failed=False;serving.calls.clear()
    runner.run(source,tmp_path/'success.jsonl',only_failed=True,read_policy=policy)
    assert serving.calls==['q0','q0']


def test_runner_resume_after_interruption_does_not_recall_recorded_success(tmp_path,source):
    serving=FakeServing();original=serving.generate_from_input
    def interrupt(texts,prompt):
        if texts==['q3']:raise RuntimeError('interrupt')
        return original(texts,prompt)
    serving.generate_from_input=interrupt
    runner=RecordedGenerationRunner(serving,tmp_path/'ledger',{})
    with pytest.raises(RuntimeError):runner.run(source,tmp_path/'success.jsonl')
    assert serving.calls==['q0','q1','q2']
    serving.generate_from_input=original;serving.calls.clear()
    report=runner.run(source,tmp_path/'success.jsonl')
    assert serving.calls==[f'q{i}' for i in range(3,100)]
    assert report['complete'] and report['failed']==5

@pytest.mark.parametrize('boundary',['before_commit','after_commit','before_export'])
def test_real_sigkill_recovery_rebuilds_outputs(tmp_path,boundary):
    import os, subprocess, sys, signal
    if os.name == 'nt':
        pytest.skip('Native POSIX SIGKILL boundary; Windows transaction tests are separate')
    source=tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps({'raw_content':str(i)})+'\n' for i in range(10)))
    script=tmp_path/'worker.py'
    script.write_text('''import json, os, signal, sys
from pathlib import Path
from dataflow.utils.record_execution import RecordedGenerationRunner
from dataflow.utils.record_ledger import RecordLedger
root=Path(sys.argv[1]); boundary=sys.argv[2]
class Serving:
    record_results=True
    def generate_from_input(self, inputs, prompt):
        with (root/'calls.jsonl').open('a') as stream: stream.write(json.dumps(inputs[0])+'\\n')
        self.last_results=[{'attempts':1}]
        return [inputs[0].upper()]
original=RecordLedger.record
def record(self,line,*args,**kwargs):
    if line==3 and boundary=='before_commit':os.kill(os.getpid(),signal.SIGKILL)
    value=original(self,line,*args,**kwargs)
    if line==3 and boundary=='after_commit':os.kill(os.getpid(),signal.SIGKILL)
    return value
RecordLedger.record=record
if boundary=='before_export':
    def kill(*args,**kwargs):os.kill(os.getpid(),signal.SIGKILL)
    RecordLedger.export_results=kill
RecordedGenerationRunner(Serving(),root/'ledger',{}).run(root/'source.jsonl',root/'success.jsonl')
''')
    result=subprocess.run([sys.executable,str(script),str(tmp_path),boundary],capture_output=True,timeout=30)
    assert result.returncode==-signal.SIGKILL, result.stderr.decode()
    result=subprocess.run([sys.executable,str(script),str(tmp_path),'none'],capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr.decode()
    rows=list(map(json.loads,(tmp_path/'success.outcomes.jsonl').read_text().splitlines()))
    assert [row['raw_content'] for row in rows]==list(map(str,range(10)))
    assert all(row['generation_outcome']['status']=='success' for row in rows)
    calls=list(map(json.loads,(tmp_path/'calls.jsonl').read_text().splitlines()))
    assert len(calls)==(11 if boundary=='before_commit' else 10)
    assert calls.count('2')==(2 if boundary=='before_commit' else 1)
