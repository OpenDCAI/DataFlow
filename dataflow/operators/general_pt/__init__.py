from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .filter.ccnet_deduplicate_filter import CCNetDeduplicateFilter
    from .filter.debertav3_filter import DebertaV3Filter
    from .filter.fineweb_edu_filter import FineWebEduFilter
    from .filter.heuristics_filter import ColonEndFilter
    from .filter.heuristics_filter import SentenceNumberFilter
    from .filter.heuristics_filter import LineEndWithEllipsisFilter
    from .filter.heuristics_filter import ContentNullFilter
    from .filter.heuristics_filter import SymbolWordRatioFilter
    from .filter.heuristics_filter import AlphaWordsFilter
    from .filter.heuristics_filter import HtmlEntityFilter
    from .filter.heuristics_filter import IDCardFilter
    from .filter.heuristics_filter import NoPuncFilter
    from .filter.heuristics_filter import SpecialCharacterFilter
    from .filter.heuristics_filter import WatermarkFilter
    from .filter.heuristics_filter  import MeanWordLengthFilter
    from .filter.heuristics_filter import StopWordFilter
    from .filter.heuristics_filter import CurlyBracketFilter
    from .filter.heuristics_filter import CapitalWordsFilter
    from .filter.heuristics_filter import LoremIpsumFilter
    from .filter.heuristics_filter import UniqueWordsFilter
    from .filter.heuristics_filter import CharNumberFilter
    from .filter.heuristics_filter import LineStartWithBulletpointFilter
    from .filter.heuristics_filter import LineWithJavascriptFilter
    from .filter.langkit_filter import LangkitFilter
    from .filter.lexical_diversity_filter import LexicalDiversityFilter
    from .filter.ngram_filter import NgramFilter
    from .filter.pair_qual_filter import PairQualFilter
    from .filter.perplexity_filter import PerplexityFilter
    from .filter.presidio_filter import PresidioFilter
    from .filter.qurating_filter import QuratingFilter
    from .filter.text_book_filter import TextbookFilter

    # generate
    from .generate.phi4qa_generator import Phi4QAGenerator

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
    from .eval.debertav3_sample_evaluator import DebertaV3SampleEvaluator
    from .eval.fineweb_edu_sample_evaluator import FineWebEduSampleEvaluator
    from .eval.pair_qual_sample_evaluator import PairQualSampleEvaluator
    from .eval.presidio_sample_evaluator import PresidioSampleEvaluator
    from .eval.textbook_sample_evaluator import TextbookSampleEvaluator
    from .eval.qurating_sample_evaluator import QuratingSampleEvaluator
    from .eval.perplexity_sample_evaluator import PerplexitySampleEvaluator
    from .eval.meta_sample_evaluator import MetaSampleEvaluator
    from .eval.bert_sample_evaluator import BertSampleEvaluator
    from .eval.bleu_sample_evaluator import BleuSampleEvaluator
    from .eval.cider_sample_evaluator import CiderSampleEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/general_pt/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/general_pt/", _import_structure)
