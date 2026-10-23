"""Content oracles preserved while the public API is split into domain operators."""
from copy import deepcopy
import pytest
from training_fixtures import *
from dataflow.utils.training_data import TrainingDataError, validate_conversation, validate_messages
from dataflow.utils.training_data_formats import convert_alpaca, convert_sharegpt, convert_messages, add_system, optional_text
from dataflow.operators.conversations import ConversationStructureSampleEvaluator
from test_training_operator_standardization import MemoryStorage

pytestmark = pytest.mark.cpu


def convert_record(kind, row, system_prompt=None):
    if kind == 'alpaca':
        messages = convert_alpaca(row, 'instruction', 'output', 'input', 'history')
    elif kind == 'sharegpt':
        messages = convert_sharegpt(row, 'conversations')
    else:
        messages = convert_messages(row, 'messages')
    key = 'messages' if kind == 'messages' else 'conversations' if kind == 'sharegpt' else 'instruction'
    return add_system(messages, system_prompt, optional_text(row, 'system'), 'system', key)

@pytest.mark.parametrize('kind,row,expected', [
    ('alpaca', ALPACA, ALPACA_EXPECTED), ('sharegpt', SHAREGPT, SHAREGPT_EXPECTED),
    ('messages', {'messages':TOOL_MESSAGES}, TOOL_MESSAGES)])
def test_exact_format_content_and_source_immutability(kind, row, expected):
    source = deepcopy(row)
    output = convert_record(kind, row)
    assert output == expected and row == source
    if kind == 'messages':
        output[1]['tool_calls'][0]['function']['arguments'] = 'changed'
        assert row == source

@pytest.mark.parametrize('source_format,row,overrides,field,code', CONVERSION_ERRORS)
def test_conversion_error_oracles(source_format,row,overrides,field,code):
    with pytest.raises(TrainingDataError) as caught:
        convert_record(source_format,row,**overrides)
    assert (caught.value.field,caught.value.code) == (field,code)

@pytest.mark.parametrize('messages', [
    [user('Q'),assistant('A')],SHAREGPT_EXPECTED,ALPACA_EXPECTED,TOOL_MESSAGES,
    [user('Q'),{'role':'assistant','content':'','tool_calls':[call()]},result(),assistant('A')]])
def test_complete_conversations_still_preserve_content(messages):
    s=MemoryStorage([{'id':1,'messages':messages}]);before=deepcopy(s.frame.to_dict('records'))
    ConversationStructureSampleEvaluator().run(s)
    assert s.frame['conversation_valid'].tolist()==[True]
    assert s.frame[['id','messages']].to_dict('records')==before

@pytest.mark.parametrize('messages,field,code', CONVERSATION_ERRORS)
def test_conversation_and_tool_error_oracles(messages,field,code):
    with pytest.raises(TrainingDataError) as caught:
        validate_conversation({'messages':messages},'messages')
    assert (caught.value.field,caught.value.code)==(field,code)
    s=MemoryStorage([{'messages':messages}])
    ConversationStructureSampleEvaluator().run(s)
    error=s.frame.iloc[0]['conversation_error']
    assert not s.frame.iloc[0]['conversation_valid']
    assert (error['field'],error['code'])==(field,code)
    assert len(s.frame)==1

@pytest.mark.parametrize('broken_call,field,code', CALL_ERRORS)
def test_tool_payload_error_oracles(broken_call,field,code):
    messages=[user('Q'),{'role':'assistant','tool_calls':[broken_call]},result(),assistant('A')]
    with pytest.raises(TrainingDataError) as caught:
        validate_conversation({'messages':messages},'messages')
    assert (caught.value.field,caught.value.code)==('messages[1].tool_calls[0]'+field,code)

@pytest.mark.parametrize('messages,field,code', SINGLE_TURN_ERRORS)
def test_single_turn_rule_oracles(messages,field,code):
    with pytest.raises(TrainingDataError) as caught:
        validate_messages({'messages':messages},'messages')
    assert (caught.value.field,caught.value.code)==(field,code)

def test_single_turn_profile_is_explicit():
    s=MemoryStorage([{'messages':SHAREGPT_EXPECTED}])
    ConversationStructureSampleEvaluator(profile='single_turn').run(s)
    assert s.frame['conversation_valid'].tolist()==[False]
    with pytest.raises(ValueError): ConversationStructureSampleEvaluator(profile='auto')
