from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
from typing import Any, Iterable, Optional, Set
import string

@PROMPT_REGISTRY.register()
class StrFormatPrompt(PromptABC):
    """
    只需要提供 f_str_template。
    - 自动从模板中解析出需要的字段（self.fields）
    - build_prompt(**kwargs) 用 kwargs 直接渲染
    - on_missing: 'raise' | 'empty'，控制缺失字段时的行为
    """
    def __init__(self, f_str_template: str = "{input_text}", on_missing: str = "raise"):
        self.f_str_template = f_str_template
        if on_missing not in ("raise", "empty"):
            raise ValueError("on_missing must be 'raise' or 'empty'")
        self.on_missing = on_missing

    
    def build_prompt(self, need_fields, **kwargs):
        # 校验缺失字段
        missing = [f for f in need_fields if f not in kwargs]
        if missing:
            if self.on_missing == "raise":
                raise KeyError(f"Missing fields for prompt: {missing}")
            # 宽松模式：用空串补齐
            for f in missing:
                kwargs[f] = ""
        
        prompt = self.f_str_template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt
