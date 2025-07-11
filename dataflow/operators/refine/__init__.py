import sys
from dataflow.utils.registry import LazyLoader

cur_path = "dataflow/operators/refine/"

_import_structure = {
    "HtmlUrlRemoverRefiner": (cur_path + "GeneralText/html_url_remover_refiner.py", "HtmlUrlRemoverRefiner"),
    "HtmlEntityRefiner": (cur_path + "GeneralText/html_entity_refiner.py", "HtmlEntityRefiner"),
    "LowercaseRefiner": (cur_path + "GeneralText/lowercase_refiner.py", "LowercaseRefiner"),
    "NERRefiner": (cur_path + "GeneralText/ner_refiner.py", "NERRefiner"),
    "PIIAnonymizeRefiner": (cur_path + "GeneralText/pii_anonymize_refiner.py", "PIIAnonymizeRefiner"),
    "ReferenceRemoverRefiner": (cur_path + "GeneralText/ref_removal_refiner.py", "ReferenceRemoverRefiner"),
    "RemoveContractionsRefiner": (cur_path + "GeneralText/remove_contractions_refiner.py", "RemoveContractionsRefiner"),
    "RemoveEmoticonsRefiner": (cur_path + "GeneralText/remove_emoticons_refiner.py", "RemoveEmoticonsRefiner"),
    "RemoveExtraSpacesRefiner": (cur_path + "GeneralText/remove_extra_spaces_refiner.py", "RemoveExtraSpacesRefiner"),
    "RemoveEmojiRefiner": (cur_path + "GeneralText/remove_emoji_refiner.py", "RemoveEmojiRefiner"),
    "RemoveImageRefsRefiner": (cur_path + "GeneralText/remove_image_ref_refiner.py", "RemoveImageRefsRefiner"),
    "RemoveNumberRefiner": (cur_path + "GeneralText/remove_number_refiner.py", "RemoveNumberRefiner"),
    "RemovePunctuationRefiner": (cur_path + "GeneralText/remove_punctuation_refiner.py", "RemovePunctuationRefiner"),
    "RemoveRepetitionsPunctuationRefiner": (cur_path + "GeneralText/remove_repetitions_punctuation_refiner.py", "RemoveRepetitionsPunctuationRefiner"),
    "RemoveStopwordsRefiner": (cur_path + "GeneralText/remove_stopwords_refiner.py", "RemoveStopwordsRefiner"),
    "SpellingCorrectionRefiner": (cur_path + "GeneralText/spelling_correction_refiner.py", "SpellingCorrectionRefiner"),
    "StemmingLemmatizationRefiner": (cur_path + "GeneralText/stemming_lemmatization_refiner.py", "StemmingLemmatizationRefiner"),
    "TextNormalizationRefiner": (cur_path + "GeneralText/text_normalization_refiner.py", "TextNormalizationRefiner"),
}

sys.modules[__name__] = LazyLoader(__name__, cur_path, _import_structure)