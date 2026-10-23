"""Offline example: bad-line isolation, aligned outcomes and explicit failure replay."""
import argparse
import json
from pathlib import Path

from dataflow.utils.record_execution import RecordedGenerationRunner
from dataflow.utils.jsonl_reader import JSONLReadPolicy

class DemoServing:
    """A deterministic test double; makes no network requests."""
    record_results = True
    model_name = 'offline-demo'
    api_url = 'local-test-double'
    configs = {}
    max_retries = 1
    def __init__(self):
        self.failures = True
        self.calls = []
    def generate_from_input(self, inputs, system_prompt):
        text = inputs[0]
        self.calls.append(text)
        failed = self.failures and text in {'sample-2', 'sample-7'}
        self.last_results = [{'index': 0, 'attempts': 1, 'error_code': 'read_timeout' if failed else None}]
        return [None if failed else text.upper()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    root = parser.parse_args().output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source.jsonl'
    source.write_text('bad-json\n' + ''.join(json.dumps({'id': i, 'raw_content': f'sample-{i}'}) + '\n' for i in range(10)))
    serving = DemoServing()
    policy = JSONLReadPolicy(root / 'read-errors.jsonl')
    generator = RecordedGenerationRunner(serving, root / 'ledger', {'revision': 1})
    generator.run(source, root / 'first.jsonl', read_policy=policy)
    assert len((root / 'first.outcomes.jsonl').read_text().splitlines()) == 10
    assert policy.last_report['rejected'] == 1
    report = json.loads((root / 'ledger/report.json').read_text())
    assert (report['success'], report['failed']) == (8, 2)
    serving.failures = False
    serving.calls.clear()
    report = generator.run(source, root / 'merged.jsonl', only_failed=True, read_policy=policy)
    assert serving.calls == ['sample-2', 'sample-7']
    assert report['success'] == 10 and report['failed'] == 0
    generator.run(source, root / 'merged.jsonl', only_failed=True, read_policy=policy)
    assert len(serving.calls) == 2
    assert len((root / 'merged.jsonl').read_text().splitlines()) == 10
    print(f'1 malformed line isolated; 2 failures replayed; 10 records exported: {root / "merged.jsonl"}')

if __name__ == '__main__':
    main()
