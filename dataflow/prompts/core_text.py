from dataflow.core.prompt import PromptABC
from typing import Set
import string

class StrFormatPrompt(PromptABC):
    """
    只需要提供 f_str_template。
    - 自动从模板中解析出需要的字段（self.fields）
    - build_prompt(**kwargs) 用 kwargs 直接渲染
    - on_missing: 'raise' | 'empty'，控制缺失字段时的行为
    """
    def __init__(self, f_str_template: str = "{input_text}", on_missing: str = "raise"):
        self.f_str_template = f_str_template
        self.fields = self._extract_fields(f_str_template)
        if on_missing not in ("raise", "empty"):
            raise ValueError("on_missing must be 'raise' or 'empty'")
        self.on_missing = on_missing

    def _extract_fields(self, fmt: str) -> Set[str]:
        """自动解析 str.format 模板中的占位符字段名。"""
        fields = set()
        for _, field_name, _, _ in string.Formatter().parse(fmt):
            if field_name:  # 跳过纯文字片段
                # 只支持简单字段名（不在这里展开索引/属性链）
                fields.add(field_name.split(".")[0].split("[")[0])
        return fields
    
    def build_prompt(self, **kwargs):
        # 校验缺失字段
        missing = [f for f in self.fields if f not in kwargs]
        if missing:
            if self.on_missing == "raise":
                raise KeyError(f"Missing fields for prompt: {missing}")
            # 宽松模式：用空串补齐
            for f in missing:
                kwargs[f] = ""
        return self.f_str_template.format(**kwargs)
