import json, resource, sys, tempfile, time
from pathlib import Path
from dataflow.utils.jsonl_reader import JSONLReadPolicy
from dataflow.utils.record_ledger import RecordLedger, RecordProcessingError

def main():
    mode, count = sys.argv[1], int(sys.argv[2])
    with tempfile.TemporaryDirectory(prefix='dataflow-benchmark-') as directory:
        root = Path(directory)
        source = root / 'input.jsonl'
        with source.open('w') as stream:
            for i in range(count):
                if mode == 'reader' and i % 997 == 0:
                    stream.write('bad\n')
                else:
                    stream.write(json.dumps({'id': i, 'text': 'x' * 256}) + '\n')
        start = time.perf_counter()
        if mode == 'reader':
            policy = JSONLReadPolicy(root / 'errors.jsonl')
            accepted = sum(len(frame) for frame in policy.iter_chunks(source, 128))
            assert accepted + policy.last_report['rejected'] == count
        else:
            with RecordLedger(root / 'ledger', source, {'version':1}) as ledger:
                def process(row):
                    if row['id'] % 100 == 0:
                        raise RecordProcessingError('test_failure')
                    return {'answer': row['id']}
                ledger.process(process)
                report = ledger.report()
                accepted = report['success']
                assert report['success'] + report['failed'] == count
                ledger.export_results(root / 'output.jsonl')
        elapsed = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
        artifact_bytes = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
        print(json.dumps({'mode':mode,'input':count,'accepted':accepted,'seconds':round(elapsed,3),'peak_rss_mib':round(peak,2),'artifact_bytes':artifact_bytes}))

if __name__ == "__main__":
    main()
