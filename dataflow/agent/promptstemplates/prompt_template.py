"""
prompts_template.py  ── Simple prompt template manager and renderer
Author  : [Zhou Liu]
License : MIT
Created : 2024-06-24

This module provides the `PromptsTemplateGenerator` class, a lightweight
singleton for managing and rendering prompt templates.

Features:
* Load and manage built-in and external prompt templates (including JSON forms)
* Cache and render templates with context variables, with safe formatting
* Support for per-operator, per-language prompt structure
* Arbitrary per-template and per-operator key-value data
* Helper functions for adding new templates

It is completely thread-safe in CPython **only if** each request handler
works on its own event-loop thread; otherwise add locking by yourself.
"""

import pathlib, json, re
from string import Formatter
from typing import Any, Dict

class PromptsTemplateGenerator:
    ANSWER_SUFFIX = ".(Answer in {lang}!!!)"

    def __new__(cls, *args, **kwargs):
        """
        Override the __new__ method to implement singleton behavior.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = super(PromptsTemplateGenerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, output_language: str):
        """
        Initialize the PromptsTemplateGenerator with a specific output language. 
        Loads built-in templates and external templates.
        """
        self.output_language = output_language
        self.prefix_system = "system_prompt_for_"
        self.prefix_task = "task_prompt_for_"
        
        # Built-in templates
        self.templates: Dict[str, str] = {}
        self.json_form_templates: Dict[str, str] = {}
        self.code_debug_templates: Dict[str, str] = {}
        self.operator_templates: Dict[str, Dict] = {}

        # Load external templates
        self._load_external_templates()
        self._load_external_json_forms()
        self._load_code_debug_templates()
        self._load_operator_templates()

    @staticmethod
    def _safe_format(tpl: str, **kwargs) -> str:
        """
        Perform safe string formatting. If a key is missing, it keeps the placeholder intact.
        """
        class _SD(dict):
            def __missing__(self, k): 
                return "{" + k + "}"
        try:
            return Formatter().vformat(tpl, [], _SD(**kwargs))
        except Exception:
            # Fallback: use regex to replace placeholders
            for v in re.findall(r"{(.*?)}", tpl):
                if v in kwargs:
                    tpl = tpl.replace("{" + v + "}", str(kwargs[v]))
            return tpl
        
    def _load_code_debug_templates(self):
        """
        Load templates from `resources/code_debug_template.json`.
        Structure: { "template_name": "… {code_snippet} …", ... }
        """
        path = pathlib.Path(__file__).parent / "resources" / "code_debug_template.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict):
                self.code_debug_templates.update(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load code_debug_template.json: {e}")

    def _load_external_templates(self):
        """
        Load templates from `resources/template.json`.
        External templates override built-in templates if keys overlap.
        """
        path = pathlib.Path(__file__).parent / "resources" / "template.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict):
                self.templates.update(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load template.json: {e}")

    def _load_external_json_forms(self):
        """
        Load JSON form templates from `resources/json_form_template.json`.
        """
        path = pathlib.Path(__file__).parent / "resources" / "json_form_template.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict):
                self.json_form_templates.update(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load json_form_template.json: {e}")

    def _load_operator_templates(self):
        """
        Load operator templates from `resources/operator_template.json`.
        """
        path = pathlib.Path(__file__).parent / "resources" / "operator_template.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict):
                self.operator_templates.update(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load operator_template.json: {e}")

    def render(self, template_name: str, add_suffix: bool = False, **kwargs):
        """
        Render a template by name with optional suffix and provided context variables.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' does not exist")
        txt = self._safe_format(self.templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_json_form(self, template_name: str, add_suffix: bool = False, **kwargs):
        """
        Render a JSON form template by name with optional suffix and provided context variables.
        """
        if template_name not in self.json_form_templates:
            raise ValueError(f"Template '{template_name}' does not exist")
        txt = self._safe_format(self.json_form_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_operator_prompt(
        self, operator_name: str, prompt_type: str = "task",
        language: str | None = None, add_suffix: bool = False, **kwargs
    ):
        """
        Render a prompt for a specific operator and prompt type. 
        Supports multi-language output and optional suffix.
        """
        lang = language or self.output_language
        op = self.operator_templates.get(operator_name)
        if not op:
            raise ValueError(f"Operator '{operator_name}' not found")
        try:
            tpl = op["prompts"][lang][prompt_type]
        except KeyError:
            raise KeyError(f"Missing prompt: operator={operator_name}, language={lang}, type={prompt_type}")
        txt = self._safe_format(tpl, **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=lang) if add_suffix else "")
    
    def render_code_debug(self, template_name: str, *, add_suffix: bool = False, **kwargs: Any) -> str:
        """
        Render a code-debugging prompt template that lives in
        `code_debug_template.json`.
        """
        if template_name not in self.code_debug_templates:
            raise ValueError(f"Code-debug template '{template_name}' does not exist")
        txt = self._safe_format(self.code_debug_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def add_sys_template(self, name: str, template: str):
        """Add a new system-level prompt template."""
        self.templates[f"system_prompt_for_{name}"] = template

    def add_task_template(self, name: str, template: str):
        """Add a new task-level prompt template."""
        self.templates[f"task_prompt_for_{name}"] = template

    def add_json_form_template(self, task_name: str, json_template: dict):
        """
        Add a new JSON form template. Supports both dict and JSON string inputs.
        """
        import json
        if isinstance(json_template, str):
            try:
                json_template = json.loads(json_template)
            except Exception as e:
                raise ValueError(f"JSON template string could not be parsed into a dict: {e}")
        self.json_form_templates[task_name] = json_template


# if __name__ == "__main__":
#     g = PromptsTemplateGenerator(output_language="en")
#     html_code = """
#     <script>alert('xss')</script>
#     <div>Safe text</div>
#     """
#     prompt = g.render_operator_prompt(
#         operator_name="html_sanitizer",
#         html=html_code  
#     )
#     print(prompt)