"""Map string-valued preference records without losing either response."""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.training_data import check_keys, check_policy, process_batch, require_text


@OPERATOR_REGISTRY.register()
class TrainingPreferenceConverter(OperatorABC):
    def __init__(self, on_invalid="raise", errors_path=None):
        super().__init__()
        check_policy(on_invalid, errors_path)
        self.on_invalid = on_invalid
        self.errors_path = errors_path
        self.last_report = {}

    def run(self, storage: DataFlowStorage, input_key: str = "prompt",
            input_chosen_key: str = "chosen", input_rejected_key: str = "rejected",
            output_prompt_key: str = "prompt", output_chosen_key: str = "chosen",
            output_rejected_key: str = "rejected"):
        """Add mapped prompt/chosen/rejected columns, preserving source fields."""
        outputs = [output_prompt_key, output_chosen_key, output_rejected_key]
        check_keys(input_key, input_chosen_key, input_rejected_key, *outputs)
        if len(set(outputs)) != 3:
            raise ValueError("DPO output keys must be distinct")

        def convert(row):
            return dict(zip(outputs, [
                require_text(row, input_key),
                require_text(row, input_chosen_key),
                require_text(row, input_rejected_key),
            ]))

        process_batch(self, storage, convert, outputs)
        return outputs

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return "映射字符串型 DPO 字段，完整保留问题、正回答和负回答，支持错误隔离。"
        return "Map string-valued DPO fields while preserving prompt, chosen and rejected verbatim."
