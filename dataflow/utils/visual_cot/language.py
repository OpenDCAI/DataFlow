"""Conservative, dependency-free language screen for reasoning prose."""
import re
import unicodedata


_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
_WORD = re.compile(r"[A-Za-z]+")
# Keep OCR evidence and LaTeX commands out of prose classification. The text
# inside command arguments is still checked, including prose inside \text{}.
_OCR_LABEL = re.compile(
    r"(?:写着|印着|标有|标注为|文字(?:为|是)|标签(?:为|是)|内容为)\s*"
    r'(?:["“「‘][A-Za-z0-9][A-Za-z0-9 ._/\-]{0,63}["”」’]|[A-Za-z][A-Za-z0-9_/-]*)')
_TECHNICAL = frozenset(
    "mm cm dm km ms kg mg ml mol hz khz mhz fps px min sin cos tan log ln exp max "
    "sqrt frac dfrac tfrac times cdot div pm pi theta alpha beta gamma delta omega "
    "left right mathrm mathbf text begin end sum int lim leq geq approx neq quad "
    "boxed circ angle parallel perp langle rangle overline infty cdots ldots".split()
)
_ENGLISH_PROSE = frozenset(
    "the this that these those it is are was were be has have with then therefore because "
    "thus we see shows means answer first next finally hence so yes no he she they "
    "their his her my our you your does did will can should".split()
)


def has_chinese_text(text: str) -> bool:
    return bool(_HAN.search(text))


def is_chinese_reasoning(text: str) -> bool:
    text = unicodedata.normalize("NFKC", text)
    if len(_HAN.findall(text)) < 2:
        return False
    text = _OCR_LABEL.sub("标签", text)
    text = re.sub(r"\\[A-Za-z]+", "∗", text)
    # Treat each clause independently so a Chinese paragraph cannot mask an English sentence.
    for clause in re.split(r"[。！？!?；;，,\n]+|(?<!\d)\.(?!\d)", text):
        words = _WORD.findall(clause)
        if words and not _HAN.search(clause):
            # A clause with no Chinese prose must be a variable or a symbolic
            # expression. Capitalization alone does not turn a sentence into a formula.
            if any(len(w) > 1 for w in words) and not re.search(r"[0-9=+*/^<>×÷≈≤≥∗∠√−-]", clause):
                return False
        if any(c.isalpha() and not (c.isascii() or _HAN.fullmatch(c)
                                    or "GREEK" in unicodedata.name(c, "")) for c in clause):
            return False
        if any(w.casefold() in _ENGLISH_PROSE and (not w.isupper() or len(words) > 1)
               for w in words):
            return False
        if any(len(w) > 1 and w.islower() and w not in _TECHNICAL for w in words):
            return False
        # Allow proper names embedded in Chinese, and standalone formula or unit lines.
        if not _HAN.search(clause) and any(len(w) > 1 and not w.isupper()
                                         and w.lower() not in _TECHNICAL for w in words):
            return False
        if re.search(r"[A-Z]{2,}\s+[A-Z]{2,}|(?:[A-Z][A-Za-z]+\s+){3}[A-Z][A-Za-z]+", clause):
            return False
    return True
