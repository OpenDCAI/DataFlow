"""Public domain operators expose fields and never hide row removal."""
import json
import pandas as pd
import pytest
from dataflow.operators.text_sft import QAMessagesGenerator, AlpacaMessagesGenerator
from dataflow.operators.conversations import (ShareGPTMessagesGenerator,
    ConversationStructureSampleEvaluator, ValidationFlagFilter)
from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage

pytestmark = pytest.mark.cpu

class MemoryStorage:
    def __init__(self, rows):
        self.frame = pd.DataFrame(rows)
        self.writes = 0
    def read(self, kind):
        return self.frame.copy()
    def write(self, frame):
        self.frame = frame
        self.writes += 1


def test_qa_keeps_source_order_and_whitespace():
    s = MemoryStorage([{'prompt': ' Q ', 'answer': ' A ', 'id': 8}])
    QAMessagesGenerator().run(s)
    assert s.frame.iloc[0]['messages'] == [{'role':'user','content':' Q '},{'role':'assistant','content':' A '}]
    assert s.frame.iloc[0]['prompt'] == ' Q '


def test_conversion_never_quarantines_implicitly():
    s = MemoryStorage([{'prompt':'q','answer':'a'},{'prompt':'q','answer':''}])
    with pytest.raises(ValueError):
        QAMessagesGenerator().run(s)
    assert len(s.frame) == 2 and s.writes == 0


def test_alpaca_optional_fields_are_explicit():
    row = {'instruction':'q','output':'a','input':'context','history':[['old q','old a']]}
    s = MemoryStorage([row])
    AlpacaMessagesGenerator().run(s)
    assert len(s.frame.iloc[0]['messages']) == 2
    s = MemoryStorage([row])
    AlpacaMessagesGenerator().run(s, input_context_key='input', input_history_key='history')
    assert len(s.frame.iloc[0]['messages']) == 4
    assert s.frame.iloc[0]['messages'][-2]['content'] == 'q\n\ncontext'


def test_sharegpt_is_row_preserving():
    s = MemoryStorage([{'conversations':[{'from':'human','value':'q'},{'from':'gpt','value':'a'}]}])
    ShareGPTMessagesGenerator().run(s)
    assert len(s.frame) == 1 and s.frame.iloc[0]['messages'][1]['role'] == 'assistant'


def test_evaluation_and_filtering_are_separate():
    good = [{'role':'user','content':'q'},{'role':'assistant','content':'a'}]
    s = MemoryStorage([{'id': i, 'messages': good if i < 97 else []} for i in range(100)])
    ConversationStructureSampleEvaluator().run(s)
    assert len(s.frame) == 100
    assert s.frame['conversation_valid'].tolist() == [True] * 97 + [False] * 3
    assert s.frame.iloc[-1]['conversation_error']['field']
    expected = s.frame.iloc[:97].copy()
    ValidationFlagFilter().run(s, input_key='conversation_valid')
    pd.testing.assert_frame_equal(s.frame, expected)

@pytest.mark.parametrize('flag', ['True', None, 1, 0])
def test_filter_rejects_non_boolean_flags(flag):
    with pytest.raises(ValueError):
        ValidationFlagFilter().run(MemoryStorage([{'conversation_valid':flag}]), input_key='conversation_valid')

@pytest.mark.parametrize('cls', [QAMessagesGenerator, AlpacaMessagesGenerator, ShareGPTMessagesGenerator,
                               ConversationStructureSampleEvaluator])
def test_compile_rejects_missing_default_inputs(tmp_path, cls):
    path = tmp_path / 'input.jsonl'
    path.write_text(json.dumps({'unrelated':'x'})+'\n')
    class Pipeline(PipelineABC):
        def __init__(self):
            super().__init__()
            self.storage = FileStorage(first_entry_file_name=str(path), cache_path=str(tmp_path/'cache'), cache_type='jsonl')
            self.op = cls()
        def forward(self):
            self.op.run(self.storage.step())
    with pytest.raises(Exception, match='(?i)key|field'):
        Pipeline().compile()


def test_compile_and_forward_custom_fields(tmp_path):
    path = tmp_path/'input.jsonl'
    path.write_text(json.dumps({'q':'Q','a':'A'})+'\n')
    class Pipeline(PipelineABC):
        def __init__(self):
            super().__init__()
            self.storage = FileStorage(first_entry_file_name=str(path), cache_path=str(tmp_path/'cache'), cache_type='jsonl')
            self.convert = QAMessagesGenerator()
            self.evaluate = ConversationStructureSampleEvaluator()
            self.keep = ValidationFlagFilter()
        def forward(self):
            self.convert.run(self.storage.step(), input_key='q', input_answer_key='a', output_key='dialogue')
            self.evaluate.run(self.storage.step(), input_key='dialogue')
            self.keep.run(self.storage.step(), input_key='conversation_valid')
    p = Pipeline()
    p.compile()
    assert p.op_nodes_list[1].input_keys == ['q','a']
    p.forward()


def test_compile_ignores_disabled_optional_fields(tmp_path, capsys):
    path = tmp_path / 'input.jsonl'
    path.write_text(json.dumps({'instruction': 'Q', 'output': 'A'}) + '\n')

    class Pipeline(PipelineABC):
        def __init__(self):
            super().__init__()
            self.storage = FileStorage(first_entry_file_name=str(path), cache_path=str(tmp_path / 'cache'), cache_type='jsonl')
            self.op = AlpacaMessagesGenerator()

        def forward(self):
            self.op.run(self.storage.step())

    Pipeline().compile()
    assert "Unexpected key 'input_" not in capsys.readouterr().out

@pytest.mark.parametrize('cls', [QAMessagesGenerator, AlpacaMessagesGenerator, ShareGPTMessagesGenerator,
                               ConversationStructureSampleEvaluator, ValidationFlagFilter])
def test_descriptions_explain_public_parameters(cls):
    for lang in ['zh', 'en']:
        description = cls.get_desc(lang)
        assert 'input_key' in description and len(description) > 100

@pytest.mark.parametrize('name,rows', [
    ('QAStructureSampleEvaluator', [{'prompt':'q','answer':'a'}, {'prompt':'q','answer':''}]),
    ('AlpacaStructureSampleEvaluator', [{'instruction':'q','output':'a'}, {'instruction':'','output':'a'}]),
    ('ShareGPTStructureSampleEvaluator', [{'conversations':[{'from':'human','value':'q'},{'from':'gpt','value':'a'}]}, {'conversations':[]}]),
])
def test_source_evaluate_filter_convert_count(name, rows):
    import importlib
    domain = 'conversations' if name.startswith('ShareGPT') else 'text_sft'
    cls = getattr(importlib.import_module('dataflow.operators.'+domain), name)
    s = MemoryStorage(rows)
    cls().run(s)
    assert len(s.frame) == 2 and s.frame['source_valid'].tolist() == [True,False]
    ValidationFlagFilter().run(s, input_key='source_valid')
    assert len(s.frame) == 1


def test_independent_runner_replays_only_failed(tmp_path):
    from dataflow.utils.record_execution import RecordedGenerationRunner
    class Serving:
        record_results = True
        def __init__(self): self.calls = []; self.fail = True
        def generate_from_input(self, inputs, system_prompt):
            self.calls.extend(inputs)
            bad = self.fail and inputs == ['second']
            self.last_results = [{'attempts':1, 'error_code':'test_failure' if bad else None}]
            return [None if bad else inputs[0].upper()]
    source = tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps({'raw_content':v})+'\n' for v in ['first','second']))
    serving = Serving()
    runner = RecordedGenerationRunner(serving, tmp_path/'ledger', {'parser':'uppercase'})
    runner.run(source, tmp_path/'success.jsonl')
    assert len((tmp_path/'success.jsonl').read_text().splitlines()) == 1
    serving.fail = False
    runner.run(source, tmp_path/'success.jsonl', only_failed=True)
    assert serving.calls == ['first','second','second']
    assert len((tmp_path/'success.jsonl').read_text().splitlines()) == 2
    runner.run(source, tmp_path/'success.jsonl', only_failed=True)
    assert serving.calls == ['first','second','second']

@pytest.mark.parametrize('cls,kwargs',[
    (AlpacaMessagesGenerator, {'input_context_key':'not_present'}),
    (ShareGPTMessagesGenerator, {'input_system_key':'not_present'}),
])
def test_direct_call_optional_missing_is_field_error(cls,kwargs):
    from training_fixtures import ALPACA, SHAREGPT
    from dataflow.utils.training_data import TrainingDataError
    s=MemoryStorage([{**ALPACA,**SHAREGPT}])
    with pytest.raises(TrainingDataError) as caught:cls().run(s,**kwargs)
    assert (caught.value.field,caught.value.code)==('not_present','missing_field')
    assert s.writes==0


def test_filter_subsets_physical_locations_with_retained_rows():
    s=MemoryStorage([{'conversation_valid':False},{'conversation_valid':True}])
    s.frame.attrs['dataflow_source_locations']=[{'line':2},{'line':8}]
    ValidationFlagFilter().run(s, input_key='conversation_valid')
    assert s.frame.attrs['dataflow_source_locations']==[{'line':8}]


def test_conversion_rejects_overwrite_before_writing():
    s=MemoryStorage([{'prompt':'q','answer':'a','messages':'preserve'}])
    with pytest.raises(ValueError,match='overwrite'):QAMessagesGenerator().run(s)
    assert s.writes==0 and s.frame.iloc[0]['messages']=='preserve'
