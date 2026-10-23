"""Stable field contracts and explicit source validation in each Storage mode."""
import json
from pathlib import Path
import pandas as pd
import pytest
from training_fixtures import QA_ERRORS, ALPACA, ALPACA_EXPECTED, SHAREGPT, SHAREGPT_EXPECTED, TOOL_MESSAGES
from test_training_operator_standardization import MemoryStorage
from dataflow.operators.text_sft import QAMessagesGenerator, QAStructureSampleEvaluator, AlpacaMessagesGenerator, AlpacaStructureSampleEvaluator
from dataflow.operators.conversations import ShareGPTMessagesGenerator, ShareGPTStructureSampleEvaluator, ConversationStructureSampleEvaluator, ConversationStructureFilter
from dataflow.operators.text_preference import PreferenceStructureSampleEvaluator, PreferenceStructureFilter
from dataflow.utils.training_data import TrainingDataError
from dataflow.utils.training_data_formats import map_preference_fields
from dataflow.pipeline import PipelineABC, BatchedPipelineABC, StreamBatchedPipelineABC
from dataflow.utils.storage import FileStorage, LazyFileStorage, BatchedFileStorage, StreamBatchedFileStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

pytestmark = pytest.mark.cpu

@pytest.mark.parametrize('record,field,code', QA_ERRORS)
def test_qa_content_rules_preserved(record,field,code):
    s=MemoryStorage([record])
    with pytest.raises(TrainingDataError) as e: QAMessagesGenerator().run(s)
    assert (e.value.field,e.value.code)==(field,code)
    assert s.writes==0

def test_batch_is_atomic_even_when_later_record_fails():
    s=MemoryStorage([{'prompt':'q','answer':'a'},{'prompt':'q','answer':''}])
    with pytest.raises(TrainingDataError): QAMessagesGenerator().run(s)
    assert s.writes==0 and len(s.frame)==2

@pytest.mark.parametrize('value',[None,'','  ',3,['text']])
def test_dpo_negative_answer_stays_required(value):
    s=MemoryStorage([{'prompt':'q','chosen':'a','rejected':value}])
    PreferenceStructureSampleEvaluator().run(s)
    assert len(s.frame)==1 and not s.frame.iloc[0]['preference_valid']
    assert s.frame.iloc[0]['preference_error']['field']=='rejected'

def test_preference_mapping_preserves_source_and_both_answers():
    row={'q':' Q\n','good':' A ','bad':' B\n','metadata':{'id':4}}
    output=map_preference_fields(row,'q','good','bad')
    assert output=={**row,'prompt':' Q\n','chosen':' A ','rejected':' B\n'}
    assert 'prompt' not in row
    with pytest.raises(ValueError): map_preference_fields(output,'q','good','bad')
    with pytest.raises(ValueError): map_preference_fields(row,'q','good','bad',output_rejected_key='chosen')

class Prepare(PipelineABC):
    def __init__(self,storage,kind):
        super().__init__(); self.storage=storage; self.kind=kind
        self.source_check={'qa':QAStructureSampleEvaluator,'alpaca':AlpacaStructureSampleEvaluator,
                           'sharegpt':ShareGPTStructureSampleEvaluator,'messages':ConversationStructureSampleEvaluator,
                           'dpo':PreferenceStructureSampleEvaluator}[kind]()
        self.keep=PreferenceStructureFilter() if kind=='dpo' else ConversationStructureFilter()
        if kind in ['qa','alpaca','sharegpt']:
            self.convert={'qa':QAMessagesGenerator,'alpaca':AlpacaMessagesGenerator,'sharegpt':ShareGPTMessagesGenerator}[kind]()
        self.final_check=PreferenceStructureSampleEvaluator() if kind=='dpo' else ConversationStructureSampleEvaluator()
    def forward(self):
        if self.kind=='qa': self.source_check.run(self.storage.step(),input_key='q',input_answer_key='a')
        elif self.kind=='alpaca': self.source_check.run(self.storage.step(),input_context_key='input',input_history_key='history',input_system_key='system')
        elif self.kind=='dpo': self.source_check.run(self.storage.step())
        else: self.source_check.run(self.storage.step())
        self.keep.run(self.storage.step(), input_key='preference_valid' if self.kind=='dpo' else 'conversation_valid' if self.kind=='messages' else 'source_valid')
        if self.kind=='qa': self.convert.run(self.storage.step(),input_key='q',input_answer_key='a')
        elif self.kind=='alpaca': self.convert.run(self.storage.step(),input_context_key='input',input_history_key='history',input_system_key='system')
        elif self.kind=='sharegpt': self.convert.run(self.storage.step())
        self.final_check.run(self.storage.step(),output_valid_key='final_valid',output_error_key='final_error')

class BatchPrepare(Prepare,BatchedPipelineABC):
    _compiled_forward=BatchedPipelineABC._compiled_forward
class StreamPrepare(Prepare,StreamBatchedPipelineABC):
    _compiled_forward=StreamBatchedPipelineABC._compiled_forward

@pytest.mark.parametrize('storage_cls,pipeline_cls',[(FileStorage,Prepare),(LazyFileStorage,Prepare),
 (BatchedFileStorage,BatchPrepare),(StreamBatchedFileStorage,StreamPrepare)])
@pytest.mark.parametrize('kind,good,bad,expected',[
 ('qa',{'q':'Q','a':' A '},{'q':'Q','a':''},[{'role':'user','content':'Q'},{'role':'assistant','content':' A '}]),
 ('alpaca',ALPACA,{**ALPACA,'output':''},ALPACA_EXPECTED),
 ('sharegpt',SHAREGPT,{**SHAREGPT,'conversations':[{'from':'human','value':'unfinished'}]},SHAREGPT_EXPECTED),
 ('messages',{'messages':TOOL_MESSAGES},{'messages':TOOL_MESSAGES[:2]},TOOL_MESSAGES),
 ('dpo',{'prompt':' Q ','chosen':' A ','rejected':' B '},{'prompt':'q','chosen':'a','rejected':''},None)])
def test_task_in_file_lazy_batch_stream(tmp_path,storage_cls,pipeline_cls,kind,good,bad,expected):
    rows=[{**bad,'id':'bad'+str(i)} for i in range(2)]+[{**good,'id':'ok'+str(i)} for i in range(3)]
    source=tmp_path/'source.jsonl'; source.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    before=source.read_bytes()
    extra={'save_on_exit':False} if storage_cls is LazyFileStorage else {}
    storage=storage_cls(str(source),cache_path=str(tmp_path/'cache'),cache_type='jsonl',**extra)
    p=pipeline_cls(storage,kind);p.compile()
    if storage_cls in [BatchedFileStorage,StreamBatchedFileStorage]: p.forward(batch_size=2,resume_from_last=False)
    else: p.forward()
    if storage_cls is LazyFileStorage: storage.flush_all()
    step=4 if kind in ['qa','alpaca','sharegpt'] else 3
    actual=[json.loads(line) for line in Path(storage._get_cache_file_path(step)).read_text().splitlines()]
    assert [row['id'] for row in actual]==['ok0','ok1','ok2']
    assert all(row['final_valid'] is True for row in actual)
    if expected is not None: assert [row['messages'] for row in actual]==[expected]*3
    else: assert all((row['prompt'],row['chosen'],row['rejected'])==(' Q ',' A ',' B ') for row in actual)
    assert source.read_bytes()==before

@pytest.mark.parametrize('cls',[QAMessagesGenerator,AlpacaMessagesGenerator,ShareGPTMessagesGenerator,
 QAStructureSampleEvaluator,AlpacaStructureSampleEvaluator,ShareGPTStructureSampleEvaluator,
 ConversationStructureSampleEvaluator,ConversationStructureFilter,PreferenceStructureSampleEvaluator,PreferenceStructureFilter])
def test_public_registration_and_description(cls):
    assert OPERATOR_REGISTRY.get(cls.__name__) is cls
    for lang in ['zh','en']: assert 'input_' in cls.get_desc(lang)

@pytest.mark.parametrize('cls,kwargs',[(AlpacaMessagesGenerator,{'input_history_key':'missing'}),
 (AlpacaStructureSampleEvaluator,{'input_system_key':'missing'}),(ShareGPTMessagesGenerator,{'input_system_key':'missing'})])
def test_enabled_optional_columns_compile_before_runtime(tmp_path,cls,kwargs):
    source=tmp_path/'source.jsonl'; source.write_text(json.dumps({**ALPACA,**SHAREGPT})+'\n')
    class P(PipelineABC):
        def __init__(self):
            super().__init__();self.storage=FileStorage(str(source),cache_path=str(tmp_path/'cache'));self.op=cls()
        def forward(self): self.op.run(self.storage.step(),**kwargs)
    with pytest.raises(Exception,match='(?i)key|field'):P().compile()

@pytest.mark.parametrize('cls',[QAMessagesGenerator,QAStructureSampleEvaluator,AlpacaMessagesGenerator,
 AlpacaStructureSampleEvaluator,ShareGPTMessagesGenerator,ShareGPTStructureSampleEvaluator,ConversationStructureSampleEvaluator])
def test_invalid_keys_and_output_collisions_fail_without_writing(cls):
    s=MemoryStorage([{**ALPACA,**SHAREGPT,'prompt':'q','answer':'a','messages':TOOL_MESSAGES}])
    with pytest.raises(ValueError):cls().run(s,input_key='')
    assert s.writes==0

def test_empty_batches_preserve_schema():
    s=MemoryStorage([]);s.frame=pd.DataFrame(columns=['prompt','answer'])
    QAStructureSampleEvaluator().run(s);ConversationStructureFilter().run(s,input_key='source_valid');QAMessagesGenerator().run(s)
    assert len(s.frame)==0 and 'messages' in s.frame

def test_zero_column_record_is_a_missing_field_error():
    for cls in [QAMessagesGenerator,QAStructureSampleEvaluator,ConversationStructureSampleEvaluator]:
        s=MemoryStorage([{}])
        with pytest.raises(TrainingDataError):cls().run(s)
        assert s.writes==0
