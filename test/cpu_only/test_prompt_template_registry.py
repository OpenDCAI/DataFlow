from dataflow.operators.reasoning import ReasoningAnswerGenerator
from dataflow.core.prompt import DIYPromptABC
from dataflow.prompts.reasoning.math import MathAnswerGeneratorPrompt
from dataflow.prompts.reasoning.general import GeneralAnswerGeneratorPrompt
from dataflow.prompts.reasoning.diy import DiyAnswerGeneratorPrompt
from typing import Union
from types import UnionType
from inspect import signature, _empty

if __name__ == "__main__":
    # print(isinstance(DIYPromptABC, type))
    init_sig = signature(ReasoningAnswerGenerator.__init__)

    print(init_sig)
    if 'prompt_template' in init_sig.parameters:
        default_value = init_sig.parameters['prompt_template'].default
        print(default_value)
        error_str = f"Class defined in {ReasoningAnswerGenerator.__module__}.{ReasoningAnswerGenerator.__qualname__} failed to pass the AUTO CHECK:\n  - `prompt_template` parameter in __init__ should include `dataflow.core.prompt.DIYPromptABC` as default allowed type. \n  - And other allowed types should be subclasses of `dataflow.core.prompt.PromptABC`. \n  - Please refer to implementation of `dataflow.operators.reasoning.generate.ReasoningAnswerGenerator` for details."
        if default_value is not _empty:
            if isinstance(default_value, type):
                if not issubclass(default_value, DIYPromptABC):
                    raise TypeError(error_str)
            elif isinstance(default_value, UnionType):
                if not any(
                issubclass(t, DIYPromptABC) for t in default_value.__args__ if isinstance(t, type)
                ):
                    raise TypeError(error_str)
            else:
                raise TypeError("Unexpected type for prompt_template default value.\n" + error_str) 
        
        else:
            raise TypeError(error_str)
        
    print(isinstance(default_value, Union[type]))
    print(type(default_value))

    a = issubclass(DiyAnswerGeneratorPrompt, DIYPromptABC)
    print(a)

    # print(default_value.__args__)