"""
prompts_template.py ── Prompt Template Manager
Author  : Zhou Liu
License : MIT
Updated : 2025-09-17

All templates are dynamically loaded from Python modules; no JSON resource
files are read any longer.
"""

from __future__ import annotations

import importlib
import inspect
import re
from string import Formatter
from typing import Any, Dict, Sequence


class PromptsTemplateGenerator:
    ANSWER_SUFFIX = ".(Answer in {lang}!!!)"

    # ---------- Singleton ----------
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------- Init ----------
    def __init__(
        self,
        output_language: str,
        *,
        python_modules: Sequence[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        output_language : str
            The language in which the model should answer finally.
        python_modules : Sequence[str] | None, optional
            A list of module names to be scanned (can be more than one).
            Defaults to ["prompts_repo"].  If the default module does not
            exist you must pass this argument explicitly.
        """
        self.output_language = output_language
        self.templates: Dict[str, str] = {}
        self.json_form_templates: Dict[str, str] = {}
        self.code_debug_templates: Dict[str, str] = {}
        self.operator_templates: Dict[str, Dict] = {}

        self._load_python_templates(python_modules or ["prompts_repo"])

    # ---------- Safe formatter ----------
    @staticmethod
    def _safe_format(tpl: str, **kwargs) -> str:
        class _Missing(dict):
            def __missing__(self, k):  # keep the placeholder
                return "{" + k + "}"

        try:
            return Formatter().vformat(tpl, [], _Missing(**kwargs))
        except Exception:  # fall back in extreme cases
            for k in re.findall(r"{(.*?)}", tpl):
                tpl = tpl.replace("{" + k + "}", str(kwargs.get(k, "{"+k+"}")))
            return tpl

    # ---------- Loader ----------
    def _load_python_templates(self, modules: Sequence[str]) -> None:
        """
        Scan all classes and top-level variables in the given modules and archive
        every string template or operator dict into the internal dictionaries.
        """
        for mod_name in modules:
            mod = importlib.import_module('.prompts_repo', package=__package__)

            # 1. Attributes inside classes
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                self._collect_from_mapping(vars(cls))

            # 2. Top-level variables
            self._collect_from_mapping(vars(mod))

    # ---------- Collect helper ----------
    def _collect_from_mapping(self, mapping: dict) -> None:
        for attr, value in mapping.items():
            if attr.startswith("_"):
                continue
            # ---- operator dict ----
            if attr == "operator_templates" and isinstance(value, dict):
                self.operator_templates.update(value)
                continue
            # ---- string templates ----
            if not isinstance(value, str):
                continue
            if attr.startswith("system_prompt_for_") or attr.startswith("task_prompt_for_"):
                self.templates[attr] = value
            elif attr.startswith("json_form_template_for_"):
                key = attr.replace("json_form_template_for_", "")
                self.json_form_templates[key] = value
            elif attr.startswith("code_debug_template_for_"):
                key = attr.replace("code_debug_template_for_", "")
                self.code_debug_templates[key] = value
            else:
                self.templates[attr] = value

    # ---------- Renderers ----------
    def render(self, template_name: str, *, add_suffix: bool = False, **kwargs) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        txt = self._safe_format(self.templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_json_form(self, template_name: str, *, add_suffix=False, **kwargs) -> str:
        if template_name not in self.json_form_templates:
            raise ValueError(f"JSON-form template '{template_name}' not found")
        txt = self._safe_format(self.json_form_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_code_debug(self, template_name: str, *, add_suffix=False, **kwargs) -> str:
        if template_name not in self.code_debug_templates:
            raise ValueError(f"Code-debug template '{template_name}' not found")
        txt = self._safe_format(self.code_debug_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_operator_prompt(
        self,
        operator_name: str,
        prompt_type: str = "task",
        language: str | None = None,
        *,
        add_suffix: bool = False,
        **kwargs,
    ) -> str:
        lang = language or self.output_language
        op = self.operator_templates.get(operator_name)
        if not op:
            raise ValueError(f"Operator '{operator_name}' not found")
        try:
            tpl = op["prompts"][lang][prompt_type]
        except KeyError:
            raise KeyError(
                f"Missing prompt (operator={operator_name}, lang={lang}, type={prompt_type})"
            )
        txt = self._safe_format(tpl, **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=lang) if add_suffix else "")

    # ---------- Runtime add ----------
    def add_sys_template(self, name: str, template: str) -> None:
        self.templates[f"system_prompt_for_{name}"] = template

    def add_task_template(self, name: str, template: str) -> None:
        self.templates[f"task_prompt_for_{name}"] = template

    def add_json_form_template(self, task_name: str, template: str | dict) -> None:
        if isinstance(template, dict):
            import json
            template = json.dumps(template, ensure_ascii=False, indent=2)
        self.json_form_templates[task_name] = template


if __name__ == "__main__":
    ptg = PromptsTemplateGenerator(
        output_language="zh",
        python_modules=["prompts_repo"],
    )

    print(ptg.render("system_prompt_for_data_content_classification"))
    print(
        ptg.render(
            "task_prompt_for_data_content_classification",
            local_tool_for_sample="《红楼梦》…",
            local_tool_for_get_categories="文学, 小说, 诗歌",
            add_suffix=True,
        )
    )