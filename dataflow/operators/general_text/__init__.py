from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .filter.alpagasus_filter import AlpagasusFilter
    from .filter.ccnet_deduplicator import CCNetDeduplicator
    from .filter.debertav3_filter import DebertaV3Filter
    from .filter.deita_complexity_filter import DeitaComplexityFilter
    from .filter.deita_quality_filter import DeitaQualityFilter
    from .filter.fineweb_edu_filter import FineWebEduFilter
    from .filter.general_filter import GeneralFilter
    from .filter.hash_deduplicator import HashDeduplicator
    from .filter.heuristics import ColonEndFilter
    from .filter.heuristics import WordNumberFilter
    from .filter.heuristics import SentenceNumberFilter
    from .filter.heuristics import LineEndWithEllipsisFilter
    from .filter.heuristics import ContentNullFilter
    from .filter.heuristics import SymbolWordRatioFilter
    from .filter.heuristics import AlphaWordsFilter
    from .filter.heuristics import HtmlEntityFilter
    from .filter.heuristics import IDCardFilter
    from .filter.heuristics import NoPuncFilter
    from .filter.heuristics import SpecialCharacterFilter
    from .filter.heuristics import WatermarkFilter
    from .filter.heuristics  import MeanWordLengthFilter
    from .filter.heuristics import StopWordFilter
    from .filter.heuristics import CurlyBracketFilter
    from .filter.heuristics import CapitalWordsFilter
    from .filter.heuristics import LoremIpsumFilter
    from .filter.heuristics import UniqueWordsFilter
    from .filter.heuristics import CharNumberFilter
    from .filter.heuristics import LineStartWithBulletpointFilter
    from .filter.heuristics import LineWithJavascriptFilter
    from .filter.heuristics import BlocklistFilter
    from .filter.instag_filter import InstagFilter
    from .filter.langkit_filter import LangkitFilter
    from .filter.language_filter import LanguageFilter
    from .filter.lexical_diversity_filter import LexicalDiversityFilter
    from .filter.llm_language_filter import LLMLanguageFilter
    from .filter.minhash_deduplicator import MinHashDeduplicator
    from .filter.ngram_filter import NgramFilter
    from .filter.ngramhash_deduplicator import NgramHashDeduplicator
    from .filter.pair_qual_filter import PairQualFilter
    from .filter.perplexity_filter import PerplexityFilter
    from .filter.perspective_filter import PerspectiveFilter
    from .filter.presidio_filter import PresidioFilter
    from .filter.qurating_filter import QuratingFilter
    from .filter.reward_model_filter import RMFilter
    from .filter.sem_deduplicator import SemDeduplicator
    from .filter.simhash_deduplicator import SimHashDeduplicator
    from .filter.superfiltering_filter import SuperfilteringFilter
    from .filter.text_book_filter import TextbookFilter
    from .filter.treeinstruct_filter import TreeinstructFilter

    # generate
    from .generate.condor_generator import CondorGenerator
    from .generate.pretrain_generator import PretrainGenerator
    from .generate.sft_generator_from_seed import SFTGeneratorSeed

    # refine
    from .refine.condor_refiner import CondorRefiner
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
    from .eval.statistics.ngram_scorer import NgramScorer
    from .eval.statistics.lexical_diversity_scorer import LexicalDiversityScorer
    from .eval.statistics.langkit_scorer import LangkitScorer

    from .eval.models.deita_quality_scorer import DeitaQualityScorer
    from .eval.models.instag_scorer import InstagScorer
    from .eval.models.debertav3_scorer import DebertaV3Scorer
    from .eval.models.deita_complexity_scorer import DeitaComplexityScorer
    from .eval.models.fineweb_edu_scorer import FineWebEduScorer
    from .eval.models.pair_qual_scorer import PairQualScorer
    from .eval.models.presidio_scorer import PresidioScorer
    from .eval.models.rm_scorer import RMScorer
    from .eval.models.textbook_scorer import TextbookScorer
    from .eval.models.superfiltering_scorer import SuperfilteringScorer
    from .eval.models.qurating_scorer import QuratingScorer
    from .eval.models.perplexity_scorer import PerplexityScorer

    from .eval.APIcaller.alpagasus_scorer import AlpagasusScorer
    from .eval.APIcaller.treeinstruct_scorer import TreeinstructScorer
    from .eval.APIcaller.perspective_scorer import PerspectiveScorer
    from .eval.APIcaller.meta_scorer import MetaScorer

    from .eval.diversity.vendi_scorer import VendiScorer
    from .eval.diversity.task2vec_scorer import Task2VecScorer
    
    from .eval.gen.bert_scorer import BERTScorer
    from .eval.gen.bleu_scorer import BleuScorer
    from .eval.gen.cider_scorer import CiderScorer
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/general_text/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/general_text/", _import_structure)
