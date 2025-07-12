from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .GeneralText.statistics.ngram_scorer import NgramScorer
    from .GeneralText.statistics.lexical_diversity_scorer import LexicalDiversityScorer
    from .GeneralText.statistics.langkit_scorer import LangkitScorer

    from .GeneralText.models.deita_quality_scorer import DeitaQualityScorer
    from .GeneralText.models.instag_scorer import InstagScorer
    from .GeneralText.models.debertav3_scorer import DebertaV3Scorer
    from .GeneralText.models.deita_complexity_scorer import DeitaComplexityScorer
    from .GeneralText.models.fineweb_edu_scorer import FineWebEduScorer
    from .GeneralText.models.pair_qual_scorer import PairQualScorer
    from .GeneralText.models.presidio_scorer import PresidioScorer
    from .GeneralText.models.rm_scorer import RMScorer
    from .GeneralText.models.textbook_scorer import TextbookScorer
    from .GeneralText.models.superfiltering_scorer import SuperfilteringScorer
    from .GeneralText.models.qurating_scorer import QuratingScorer
    from .GeneralText.models.perplexity_scorer import PerplexityScorer

    from .GeneralText.APIcaller.alpagasus_scorer import AlpagasusScorer
    from .GeneralText.APIcaller.treeinstruct_scorer import TreeinstructScorer
    from .GeneralText.APIcaller.perspective_scorer import PerspectiveScorer
    from .GeneralText.APIcaller.meta_scorer import MetaScorer

    from .GeneralText.diversity.vendi_scorer import VendiScorer
    from .GeneralText.diversity.task2vec_scorer import Task2VecScorer

    from .GeneralText.gen.bleu_scorer import BleuScorer
    from .GeneralText.gen.cider_scorer import CiderScorer
    from .GeneralText.gen.bert_scorer import BERTScorer

    from .Reasoning.category_info import CategoryInfo
    from .Reasoning.difficulty_info import DifficultyInfo
    from .Reasoning.token_info import ToKenInfo

    from .AgenticRAG.statistics.f1_scorer import F1Scorer

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/eval/"

    # _import_structure = {
    #     # Statistics
    #     "NgramScorer": (cur_path + "GeneralText/statistics/ngram_scorer.py", "NgramScorer"),
    #     "LexicalDiversityScorer": (cur_path + "GeneralText/statistics/lexical_diversity_scorer.py", "LexicalDiversityScorer"),
    #     "LangkitScorer": (cur_path + "GeneralText/statistics/langkit_scorer.py", "LangkitScorer"),

    #     # Models
    #     "DeitaQualityScorer": (cur_path + "GeneralText/models/deita_quality_scorer.py", "DeitaQualityScorer"),
    #     "InstagScorer": (cur_path + "GeneralText/models/instag_scorer.py", "InstagScorer"),
    #     "DebertaV3Scorer": (cur_path + "GeneralText/models/debertav3_scorer.py", "DebertaV3Scorer"),
    #     "DeitaComplexityScorer": (cur_path + "GeneralText/models/deita_complexity_scorer.py", "DeitaComplexityScorer"),
    #     "FineWebEduScorer": (cur_path + "GeneralText/models/fineweb_edu_scorer.py", "FineWebEduScorer"),
    #     "PairQualScorer": (cur_path + "GeneralText/models/pair_qual_scorer.py", "PairQualScorer"),
    #     "PresidioScorer": (cur_path + "GeneralText/models/presidio_scorer.py", "PresidioScorer"),
    #     "RMScorer": (cur_path + "GeneralText/models/rm_scorer.py", "RMScorer"),
    #     "TextbookScorer": (cur_path + "GeneralText/models/textbook_scorer.py", "TextbookScorer"),
    #     "SuperfilteringScorer": (cur_path + "GeneralText/models/superfiltering_scorer.py", "SuperfilteringScorer"),
    #     "QuratingScorer": (cur_path + "GeneralText/models/qurating_scorer.py", "QuratingScorer"),
    #     "PerplexityScorer": (cur_path + "GeneralText/models/perplexity_scorer.py", "PerplexityScorer"),

    #     # API Caller
    #     "AlpagasusScorer": (cur_path + "GeneralText/APIcaller/alpagasus_scorer.py", "AlpagasusScorer"),
    #     "TreeinstructScorer": (cur_path + "GeneralText/APIcaller/treeinstruct_scorer.py", "TreeinstructScorer"),
    #     "PerspectiveScorer": (cur_path + "GeneralText/APIcaller/perspective_scorer.py", "PerspectiveScorer"),
    #     "MetaScorer": (cur_path + "GeneralText/APIcaller/meta_scorer.py", "MetaScorer"),

    #     # Diversity
    #     "VendiScorer": (cur_path + "GeneralText/diversity/vendi_scorer.py", "VendiScorer"),
    #     "Task2VecScorer": (cur_path + "GeneralText/diversity/task2vec_scorer.py", "Task2VecScorer"),

    #     # Generation Metrics
    #     "BleuScorer": (cur_path + "GeneralText/gen/bleu_scorer.py", "BleuScorer"),
    #     "CiderScorer": (cur_path + "GeneralText/gen/cider_scorer.py", "CiderScorer"),
    #     "BERTScorer": (cur_path + "GeneralText/gen/bert_scorer.py", "BERTScorer"),

    #     # Reasoning (额外添加的 Reasoning 模块，如不需要可移除)
    #     "CategoryInfo": (cur_path + "Reasoning/category_info.py", "CategoryInfo"),
    #     "DifficultyInfo": (cur_path + "Reasoning/difficulty_info.py", "DifficultyInfo"),
    #     "ToKenInfo": (cur_path + "Reasoning/token_info.py", "ToKenInfo"),
    # }
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/eval/", _import_structure)