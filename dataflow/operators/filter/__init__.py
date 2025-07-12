from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # GeneralText - Primary filters
    from .GeneralText.ngram_filter import NgramFilter
    from .GeneralText.language_filter import LanguageFilter
    from .GeneralText.deita_quality_filter import DeitaQualityFilter
    from .GeneralText.deita_complexity_filter import DeitaComplexityFilter
    from .GeneralText.instag_filter import InstagFilter
    from .GeneralText.pair_qual_filter import PairQualFilter
    from .GeneralText.qurating_filter import QuratingFilter
    from .GeneralText.superfiltering_filter import SuperfilteringFilter
    from .GeneralText.fineweb_edu_filter import FineWebEduFilter
    from .GeneralText.text_book_filter import TextbookFilter
    from .GeneralText.alpagasus_filter import AlpagasusFilter
    from .GeneralText.debertav3_filter import DebertaV3Filter
    from .GeneralText.langkit_filter import LangkitFilter
    from .GeneralText.lexical_diversity_filter import LexicalDiversityFilter
    from .GeneralText.perplexity_filter import PerplexityFilter
    from .GeneralText.perspective_filter import PerspectiveFilter
    from .GeneralText.presidio_filter import PresidioFilter
    from .GeneralText.reward_model_filter import RMFilter
    from .GeneralText.treeinstruct_filter import TreeinstructFilter

    # GeneralText - Heuristic filters (from heuristics.py)
    from .GeneralText.heuristics import (
        ColonEndFilter,
        WordNumberFilter,
        BlocklistFilter,
        SentenceNumberFilter,
        LineEndWithEllipsisFilter,
        ContentNullFilter,
        MeanWordLengthFilter,
        SymbolWordRatioFilter,
        HtmlEntityFilter,
        IDCardFilter,
        NoPuncFilter,
        SpecialCharacterFilter,
        WatermarkFilter,
        StopWordFilter,
        CurlyBracketFilter,
        CapitalWordsFilter,
        LoremIpsumFilter,
        UniqueWordsFilter,
        CharNumberFilter,
        LineStartWithBulletpointFilter,
        LineWithJavascriptFilter,
    )

    # GeneralText - Deduplicators
    from .GeneralText.minhash_deduplicator import MinHashDeduplicator
    from .GeneralText.ccnet_deduplicator import CCNetDeduplicator
    from .GeneralText.hash_deduplicator import HashDeduplicator
    from .GeneralText.ngramhash_deduplicator import NgramHashDeduplicator
    from .GeneralText.sem_deduplicator import SemDeduplicator
    from .GeneralText.simhash_deduplicator import SimHashDeduplicator

    # Reasoning filters
    from .Reasoning.answer_formatter_filter import AnswerFormatterFilter
    from .Reasoning.answer_groundtruth_filter import AnswerGroundTruthFilter
    from .Reasoning.answer_judger_mathverify import AnswerJudger_MathVerify
    from .Reasoning.answer_ngram_filter import AnswerNgramFilter
    from .Reasoning.answer_pipeline_root import AnswerPipelineRoot
    from .Reasoning.answer_token_length_filter import AnswerTokenLengthFilter
    from .Reasoning.question_filter import QuestionFilter

    # AgenticRAG
    from .AgenticRAG.content_chooser import ContentChooser
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    # from .Reasoning import *
    # from .GeneralText import *
    # from .AgenticRAG import *
    cur_path = "dataflow/operators/filter/"
    # _import_structure = {
    #     # GeneralText
    #     # Primary filters
    #     "NgramFilter": (cur_path + "GeneralText/ngram_filter.py", "NgramFilter"),
    #     "LanguageFilter": (cur_path + "GeneralText/language_filter.py", "LanguageFilter"),
    #     "DeitaQualityFilter": (cur_path + "GeneralText/deita_quality_filter.py", "DeitaQualityFilter"),
    #     "DeitaComplexityFilter": (cur_path + "GeneralText/deita_complexity_filter.py", "DeitaComplexityFilter"),
    #     "InstagFilter": (cur_path + "GeneralText/instag_filter.py", "InstagFilter"),
    #     "PairQualFilter": (cur_path + "GeneralText/pair_qual_filter.py", "PairQualFilter"),
    #     "QuratingFilter": (cur_path + "GeneralText/qurating_filter.py", "QuratingFilter"),
    #     "SuperfilteringFilter": (cur_path + "GeneralText/superfiltering_filter.py", "SuperfilteringFilter"),
    #     "FineWebEduFilter": (cur_path + "GeneralText/fineweb_edu_filter.py", "FineWebEduFilter"),
    #     "TextbookFilter": (cur_path + "GeneralText/text_book_filter.py", "TextbookFilter"),
    #     "AlpagasusFilter": (cur_path + "GeneralText/alpagasus_filter.py", "AlpagasusFilter"),
    #     "DebertaV3Filter": (cur_path + "GeneralText/debertav3_filter.py", "DebertaV3Filter"),
    #     "LangkitFilter": (cur_path + "GeneralText/langkit_filter.py", "LangkitFilter"),
    #     "LexicalDiversityFilter": (cur_path + "GeneralText/lexical_diversity_filter.py", "LexicalDiversityFilter"),
    #     "PerplexityFilter": (cur_path + "GeneralText/perplexity_filter.py", "PerplexityFilter"),
    #     "PerspectiveFilter": (cur_path + "GeneralText/perspective_filter.py", "PerspectiveFilter"),
    #     "PresidioFilter": (cur_path + "GeneralText/presidio_filter.py", "PresidioFilter"),
    #     "RMFilter": (cur_path + "GeneralText/reward_model_filter.py", "RMFilter"),
    #     "TreeinstructFilter": (cur_path + "GeneralText/treeinstruct_filter.py", "TreeinstructFilter"),

    #     # Heuristic filters
    #     "ColonEndFilter": (cur_path + "GeneralText/heuristics.py", "ColonEndFilter"),
    #     "WordNumberFilter": (cur_path + "GeneralText/heuristics.py", "WordNumberFilter"),
    #     "BlocklistFilter": (cur_path + "GeneralText/heuristics.py", "BlocklistFilter"),
    #     "SentenceNumberFilter": (cur_path + "GeneralText/heuristics.py", "SentenceNumberFilter"),
    #     "LineEndWithEllipsisFilter": (cur_path + "GeneralText/heuristics.py", "LineEndWithEllipsisFilter"),
    #     "ContentNullFilter": (cur_path + "GeneralText/heuristics.py", "ContentNullFilter"),
    #     "MeanWordLengthFilter": (cur_path + "GeneralText/heuristics.py", "MeanWordLengthFilter"),
    #     "SymbolWordRatioFilter": (cur_path + "GeneralText/heuristics.py", "SymbolWordRatioFilter"),
    #     "HtmlEntityFilter": (cur_path + "GeneralText/heuristics.py", "HtmlEntityFilter"),
    #     "IDCardFilter": (cur_path + "GeneralText/heuristics.py", "IDCardFilter"),
    #     "NoPuncFilter": (cur_path + "GeneralText/heuristics.py", "NoPuncFilter"),
    #     "SpecialCharacterFilter": (cur_path + "GeneralText/heuristics.py", "SpecialCharacterFilter"),
    #     "WatermarkFilter": (cur_path + "GeneralText/heuristics.py", "WatermarkFilter"),
    #     "StopWordFilter": (cur_path + "GeneralText/heuristics.py", "StopWordFilter"),
    #     "CurlyBracketFilter": (cur_path + "GeneralText/heuristics.py", "CurlyBracketFilter"),
    #     "CapitalWordsFilter": (cur_path + "GeneralText/heuristics.py", "CapitalWordsFilter"),
    #     "LoremIpsumFilter": (cur_path + "GeneralText/heuristics.py", "LoremIpsumFilter"),
    #     "UniqueWordsFilter": (cur_path + "GeneralText/heuristics.py", "UniqueWordsFilter"),
    #     "CharNumberFilter": (cur_path + "GeneralText/heuristics.py", "CharNumberFilter"),
    #     "LineStartWithBulletpointFilter": (cur_path + "GeneralText/heuristics.py", "LineStartWithBulletpointFilter"),
    #     "LineWithJavascriptFilter": (cur_path + "GeneralText/heuristics.py", "LineWithJavascriptFilter"),

    #     # Deduplicators
    #     "MinHashDeduplicator": (cur_path + "GeneralText/minhash_deduplicator.py", "MinHashDeduplicator"),
    #     "CCNetDeduplicator": (cur_path + "GeneralText/ccnet_deduplicator.py", "CCNetDeduplicator"),
    #     "HashDeduplicator": (cur_path + "GeneralText/hash_deduplicator.py", "HashDeduplicator"),
    #     "NgramHashDeduplicator": (cur_path + "GeneralText/ngramhash_deduplicator.py", "NgramHashDeduplicator"),
    #     "SemDeduplicator": (cur_path + "GeneralText/sem_deduplicator.py", "SemDeduplicator"),
    #     "SimHashDeduplicator": (cur_path + "GeneralText/simhash_deduplicator.py", "SimHashDeduplicator"),

    #     # Reasoning
    #     "AnswerFormatterFilter": (cur_path + "Reasoning/answer_formatter_filter.py", "AnswerFormatterFilter"),
    #     "AnswerGroundTruthFilter": (cur_path + "Reasoning/answer_groundtruth_filter.py", "AnswerGroundTruthFilter"),
    #     "AnswerJudger_MathVerify": (cur_path + "Reasoning/answer_judger_mathverify.py", "AnswerJudger_MathVerify"),
    #     "AnswerNgramFilter": (cur_path + "Reasoning/answer_ngram_filter.py", "AnswerNgramFilter"),
    #     "AnswerPipelineRoot": (cur_path + "Reasoning/answer_pipeline_root.py", "AnswerPipelineRoot"),
    #     "AnswerTokenLengthFilter": (cur_path + "Reasoning/answer_token_length_filter.py", "AnswerTokenLengthFilter"),
    #     "QuestionFilter": (cur_path + "Reasoning/question_filter.py", "QuestionFilter"),
        
    #     # AgenticRAG
    #     "ContentChooser": (cur_path + "AgenticRAG/content_chooser.py", "ContentChooser")
    # }
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/filter/", _import_structure)
