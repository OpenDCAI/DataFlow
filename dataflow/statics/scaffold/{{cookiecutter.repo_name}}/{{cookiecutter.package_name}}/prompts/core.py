from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class MyFormatStringPrompt(PromptABC):
    """
    Only the f_str_template needs to be provided.
    - Automatically parses the required fields from the template (self.fields)
    - build_prompt(**kwargs) renders directly using kwargs
    - on_missing: 'raise' | 'empty', controls behavior when fields are missing
    """
    def __init__(self, f_str_template: str = "{input_text}", on_missing: str = "raise"):
        self.f_str_template = f_str_template
        if on_missing not in ("raise", "empty"):
            raise ValueError("on_missing must be 'raise' or 'empty'")
        self.on_missing = on_missing

    
    def build_prompt(self, need_fields, **kwargs):
        # Validate missing fields
        missing = [f for f in need_fields if f not in kwargs]
        if missing:
            if self.on_missing == "raise":
                raise KeyError(f"Missing fields for prompt: {missing}")
            # Lenient mode: fill missing fields with empty strings
            for f in missing:
                kwargs[f] = ""
        {% raw %}
        prompt = self.f_str_template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt
        {% endraw %}