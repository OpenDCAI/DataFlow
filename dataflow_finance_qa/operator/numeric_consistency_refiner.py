"""Numeric-consistency refiner operator.

Rewrites financial QA answers so that every number, unit and ratio in the
answer strictly matches the source text. Used after (or instead of) the
hallucination filter to repair, rather than discard, borderline samples.
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

REFINE_PROMPT_ZH = (
    "你是金融数据修订专家。核对答案中的每个数字、单位、比例与来源文本，"
    "只修正不一致之处，保持其余表述不变；若答案与来源无矛盾则原样输出。\n"
    "仅输出修订后的答案文本，不要解释。"
)


@OPERATOR_REGISTRY.register()
class NumericConsistencyRefiner(OperatorABC):
    def __init__(self, llm_serving):
        super().__init__()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "精炼金融 QA 答案：逐项核对数字/单位/比例与来源文本的一致性，"
                "仅修正矛盾表述，输出修订后的答案。"
            )
        return (
            "Refine financial QA answers by checking numeric/unit/ratio "
            "consistency against the source text; only fix inconsistencies."
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_text: str = "text",
        input_answer: str = "answer",
        output_answer: str = "answer_refined",
    ):
        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")
        refined = []
        for text, answer in zip(dataframe[input_text], dataframe[input_answer]):
            if not answer:
                refined.append(answer)
                continue
            user_input = f"来源文本：\n{text}\n\n答案：{answer}"
            refined.append(
                self.llm_serving.generate_from_input(
                    user_inputs=[user_input], system_prompt=REFINE_PROMPT_ZH
                )[0].strip()
            )
        dataframe[output_answer] = refined
        storage.write(dataframe)
        return [output_answer]
