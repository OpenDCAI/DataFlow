"""Failures/replay are source-linked, ordered and idempotent."""
import json
from pathlib import Path
import pytest
from dataflow.utils.record_ledger import RecordLedger, RecordProcessingError, RecordResult
from dataflow.utils.storage import FileStorage, BatchedFileStorage, StreamBatchedFileStorage
from dataflow.operators.core_text import RecordedPromptedGenerator

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

def test_generator_output_report_and_public_replay(tmp_path, source):
    serving = FakeServing()
    operator = RecordedPromptedGenerator(serving, tmp_path / 'state', {'parser_revision': 1})
    storage = FileStorage(str(source), cache_path=str(tmp_path / 'cache')).step()
    operator.run(storage)
    assert operator.last_report['failed'] == 5
    assert len(Path(storage._get_cache_file_path(1)).read_text().splitlines()) == 95
    serving.failed = False
    serving.calls.clear()
    operator.replay_failed(source, tmp_path / 'merged.jsonl')
    assert serving.calls == ['q0','q20','q40','q60','q80']
    assert len((tmp_path / 'merged.jsonl').read_text().splitlines()) == 100
    assert operator.last_report['success'] == 100
    serving.calls.clear()
    operator.replay_failed(source, tmp_path / 'merged.jsonl')
    assert serving.calls == []

def test_parser_error_records_failure_without_logging_raw_response(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text('{"raw_content":"secret"}\n')
    serving = FakeServing()
    operator = RecordedPromptedGenerator(serving, tmp_path / 'state', {}, parse_response=json.loads)
    operator.run(FileStorage(str(source), cache_path=str(tmp_path / 'cache')).step())
    failure = json.loads((tmp_path / 'state/failures.jsonl').read_text())
    assert (failure['stage'], failure['code']) == ('parse', 'invalid_model_output')
    assert 'secret' not in (tmp_path / 'state/failures.jsonl').read_text()

def test_recorded_generator_stream_resume_rebuilds_uncommitted_output_without_recalling_successes(tmp_path, source, monkeypatch):
    from dataflow.pipeline import StreamBatchedPipelineABC
    from dataflow.utils.storage import StreamBatchedFileStorage
    class RecordedPipeline(StreamBatchedPipelineABC):
        def __init__(self):
            super().__init__()
            self.storage = StreamBatchedFileStorage(str(source), cache_path=str(tmp_path / 'cache'))
            self.generate = RecordedPromptedGenerator(serving, tmp_path / 'ledger', {'revision': 1})
        def forward(self):
            self.generate.run(self.storage.step())
    serving = FakeServing()
    original_write = StreamBatchedFileStorage.write
    calls = []
    def crash(storage, data):
        output = original_write(storage, data)
        if storage.batch_step == 1:
            raise RuntimeError('interrupt after output')
        return output
    with monkeypatch.context() as patch:
        patch.setattr(StreamBatchedFileStorage, 'write', crash)
        pipeline = RecordedPipeline()
        pipeline.compile()
        with pytest.raises(RuntimeError):
            pipeline.forward(batch_size=7, checkpoint_dir=tmp_path / 'checkpoint', checkpoint_config={'revision': 1})
    assert len(serving.calls) == 14
    pipeline = RecordedPipeline()
    pipeline.compile()
    pipeline.forward(batch_size=7, checkpoint_dir=tmp_path / 'checkpoint', checkpoint_config={'revision': 1})
    assert len(serving.calls) == 100
    actual = [json.loads(line) for line in Path(pipeline.storage._get_cache_file_path(1)).read_text().splitlines()]
    assert len(actual) == 95
    assert [row['id'] for row in actual] == [i for i in range(100) if i % 20]
    assert pipeline.generate._operator.last_report['complete'] is True


def test_invalid_parser_type_is_not_silently_treated_as_bad_model_data(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text('{"raw_content":"valid"}\n')
    def bug(value):
        raise TypeError('parser programming error')
    generator = RecordedPromptedGenerator(FakeServing(), tmp_path / 'ledger', {}, parse_response=bug)
    with pytest.raises(TypeError, match='programming error'):
        generator.run(FileStorage(str(source), cache_path=str(tmp_path / 'cache')).step())

def test_empty_pandas_jsonl_output_is_a_complete_zero_record_stage(tmp_path):
    source = tmp_path / 'empty.jsonl'
    source.write_text('\n')
    generator = RecordedPromptedGenerator(FakeServing(), tmp_path / 'ledger', {})
    generator.run(FileStorage(str(source), cache_path=str(tmp_path / 'cache')).step())
    assert generator.last_report['input'] == 0
    assert generator.last_report['complete'] is True
    assert generator.last_report['attempts'] == 0


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('storage_type', [FileStorage, BatchedFileStorage, StreamBatchedFileStorage])
def test_empty_objects_have_terminal_failures_in_all_materialized_modes(tmp_path, count, storage_type):
    source = tmp_path / 'input.jsonl'
    source.write_text('{}\n' * count, encoding='utf-8')
    serving = FakeServing()
    generator = RecordedPromptedGenerator(serving, tmp_path / 'ledger', {})
    storage = storage_type(str(source), cache_path=str(tmp_path / 'cache')).step()
    if storage_type is FileStorage:
        generator.run(storage)
    else:
        storage.batch_size = 2
        if storage_type is StreamBatchedFileStorage:
            for index, chunk in enumerate(storage.iter_chunks()):
                storage.batch_step = index
                storage._current_streaming_chunk = chunk
                generator.run(storage)
            storage._current_streaming_chunk = None
        else:
            for index in range((count + 1) // 2):
                storage.batch_step = index
                generator.run(storage)
    assert serving.calls == []
    assert generator.last_report['input'] == count
    assert generator.last_report['failed'] == count
    assert generator.last_report['pending'] == 0
    assert generator.last_report['complete'] is True
    failures = [json.loads(line) for line in (tmp_path / 'ledger/failures.jsonl').read_text().splitlines()]
    assert [(row['line'], row['code'], row['attempts']) for row in failures] == [
        (index + 1, 'invalid_input', 0) for index in range(count)
    ]
    generator.replay_failed(source, tmp_path / 'merged.jsonl')
    assert serving.calls == []
    assert generator.last_report['failed'] == count
    assert generator.last_report['pending'] == 0


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
