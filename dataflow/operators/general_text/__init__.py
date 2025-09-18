from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .filter.rule_based_filter import ColonEndFilter
    from .filter.rule_based_filter import SentenceNumberFilter
    from .filter.rule_based_filter import LineEndWithEllipsisFilter
    from .filter.rule_based_filter import ContentNullFilter
    from .filter.rule_based_filter import SymbolWordRatioFilter
    from .filter.rule_based_filter import AlphaWordsFilter
    from .filter.rule_based_filter import HtmlEntityFilter
    from .filter.rule_based_filter import IDCardFilter
    from .filter.rule_based_filter import NoPuncFilter
    from .filter.rule_based_filter import SpecialCharacterFilter
    from .filter.rule_based_filter import WatermarkFilter
    from .filter.rule_based_filter  import MeanWordLengthFilter
    from .filter.rule_based_filter import StopWordFilter
    from .filter.rule_based_filter import CurlyBracketFilter
    from .filter.rule_based_filter import CapitalWordsFilter
    from .filter.rule_based_filter import LoremIpsumFilter
    from .filter.rule_based_filter import UniqueWordsFilter
    from .filter.rule_based_filter import CharNumberFilter
    from .filter.rule_based_filter import LineStartWithBulletpointFilter
    from .filter.rule_based_filter import LineWithJavascriptFilter
    from .filter.langkit_filter import LangkitFilter
    from .filter.lexical_diversity_filter import LexicalDiversityFilter
    from .filter.ngram_filter import NgramFilter
    from .filter.presidio_filter import PresidioFilter
    from .filter.blocklist_filter import BlocklistFilter
    from .filter.hash_deduplicate_filter import HashDeduplicateFilter
    from .filter.language_filter import LanguageFilter
    from .filter.llm_language_filter import LLMLanguageFilter
    from .filter.minhash_deduplicate_filter import MinHashDeduplicateFilter
    from .filter.ngramhash_deduplicate_filter import NgramHashDeduplicateFilter
    from .filter.perspective_filter import PerspectiveFilter
    from .filter.sem_deduplicate_filter import SemDeduplicateFilter
    from .filter.simhash_deduplicate_filter import SimHashDeduplicateFilter
    from .filter.word_number_filter import WordNumberFilter

    # refine
    from .refine.html_entity_refiner import HtmlEntityRefiner
    from .refine.html_url_remover_refiner import HtmlUrlRemoverRefiner
    from .refine.lowercase_refiner import LowercaseRefiner
    from .refine.ner_refiner import NERRefiner
    from .refine.pii_anonymize_refiner import PIIAnonymizeRefiner
    from .refine.ref_removal_refiner import ReferenceRemoverRefiner
    from .refine.remove_contractions_refiner import RemoveContractionsRefiner
    from .refine.remove_emoji_refiner import RemoveEmojiRefiner
    from .refine.remove_emoticons_refiner import RemoveEmoticonsRefiner
    from .refine.remove_extra_spaces_refiner import RemoveExtraSpacesRefiner
    from .refine.remove_image_ref_refiner import RemoveImageRefsRefiner
    from .refine.remove_number_refiner import RemoveNumberRefiner
    from .refine.remove_punctuation_refiner import RemovePunctuationRefiner
    from .refine.remove_repetitions_punctuation_refiner import RemoveRepetitionsPunctuationRefiner
    from .refine.remove_stopwords_refiner import RemoveStopwordsRefiner
    from .refine.spelling_correction_refiner import SpellingCorrectionRefiner
    from .refine.stemming_lemmatization_refiner import StemmingLemmatizationRefiner
    from .refine.text_normalization_refiner import TextNormalizationRefiner
    
    # eval
    from .eval.ngram_sample_evaluator import NgramSampleEvaluator
    from .eval.lexical_diversity_sample_evaluator import LexicalDiversitySampleEvaluator
    from .eval.langkit_sample_evaluator import LangkitSampleEvaluator
    from .eval.presidio_sample_evaluator import PresidioSampleEvaluator
    from .eval.bert_sample_evaluator import BertSampleEvaluator
    from .eval.bleu_sample_evaluator import BleuSampleEvaluator
    from .eval.cider_sample_evaluator import CiderSampleEvaluator
    from .eval.perspective_sample_evaluator import PerspectiveSampleEvaluator
    from .eval.task2vec_dataset_evaluator import Task2VecDatasetEvaluator
    from .eval.vendi_dataset_evaluator import VendiDatasetEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/general_text/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/general_text/", _import_structure)
