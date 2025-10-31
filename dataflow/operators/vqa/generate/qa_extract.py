from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.vqa import QAExtractPrompt
import os
import json

from dataflow.core.prompt import prompt_restrict 

@prompt_restrict(QAExtractPrompt)
@OPERATOR_REGISTRY.register()
class QAExtractor(OperatorABC):
    def __init__(self,
                llm_serving: LLMServingABC = None,
                ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt = QAExtractPrompt()
        
    def _convert_json(self, input_file, output_file):
        with open(input_file, 'r') as infile:
            data = list(json.load(infile))
        
        new_data = []
        
        id = 0
        for item in data:
            item['id'] = id
            item.pop('bbox', None)
            item.pop('page_idx', None)
            if item.get('type','') == 'list':
                if item['sub_type'] == 'text':
                    for idx, list_item in enumerate(item.get('list_items', [])):
                        new_item = {
                            'type': 'text',
                            'text': list_item,
                            'id': id + idx,
                        }
                        new_data.append(new_item)
                    id += len(item.get('list_items', []))
            else:
                new_data.append(item)
                id += 1
        
        with open(output_file, 'w') as outfile:
            json.dump(new_data, outfile, ensure_ascii=False)

    def run(self, storage, input_json_path: str, input_subject: str):

        system_prompt = self.prompt.build_prompt(input_subject)
        
        self._convert_json(input_json_path, input_json_path.replace('.json', '_converted.json'))
        
        with open(input_json_path.replace('.json', '_converted.json'), 'r') as infile:
            data = json.load(infile)
            user_input = json.dumps(data, ensure_ascii=False)

        responses = self.llm_serving.generate_from_input([user_input], system_prompt)

        return responses[0]