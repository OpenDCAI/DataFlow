import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "LowercaseRefiner": ("dataflow/process/text/refiners/lowercase_refiner.py", "LowercaseRefiner"),
    "PIIAnonymizeRefiner": ("dataflow/process/text/refiners/pii_anonymize_refiner.py", "PIIAnonymizeRefiner"),
    "RemovePunctuationRefiner": ("dataflow/process/text/refiners/remove_punctuation_refiner.py", "RemovePunctuationRefiner"),
    "RemoveNumberRefiner": ("dataflow/process/text/refiners/remove_number_refiner.py", "RemoveNumberRefiner"),
    "RemoveExtraSpacesRefiner": ("dataflow/process/text/refiners/remove_extra_spaces_refiner.py", "RemoveExtraSpacesRefiner"),
    "RemoveRepetitionsPunctuationRefiner": ("dataflow/process/text/refiners/remove_repetitions_punctuation_refiner.py", "RemoveRepetitionsPunctuationRefiner"),
    "RemoveEmojiRefiner": ("dataflow/process/text/refiners/remove_emoji_refiner.py", "RemoveEmojiRefiner"),
    "RemoveEmoticonsRefiner": ("dataflow/process/text/refiners/remove_emoticons_refiner.py", "RemoveEmoticonsRefiner"),
    "RemoveContractionsRefiner": ("dataflow/process/text/refiners/remove_contractions_refiner.py", "RemoveContractionsRefiner"),
    "HtmlUrlRemoverRefiner": ("dataflow/process/text/refiners/html_url_remover_refiner.py", "HtmlUrlRemoverRefiner"),
    "TextNormalizationRefiner": ("dataflow/process/text/refiners/text_normalization_refiner.py", "TextNormalizationRefiner"),
    "NERRefiner": ("dataflow/process/text/refiners/ner_refiner.py", "NERRefiner"),
    "StemmingLemmatizationRefiner": ("dataflow/process/text/refiners/stemming_lemmatization_refiner.py", "StemmingLemmatizationRefiner"),
    "SpellingCorrectionRefiner": ("dataflow/process/text/refiners/spelling_correction_refiner.py", "SpellingCorrectionRefiner"),
    "RemoveStopwordsRefiner": ("dataflow/process/text/refiners/remove_stopwords_refiner.py", "RemoveStopwordsRefiner")
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/process/text/refiners", _import_structure)
