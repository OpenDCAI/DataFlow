import sys
from dataflow.utils.registry import LazyLoader
from .GeneralText import *

cur_path = "dataflow/operators/eval/"
_import_structure = {  
    "NgramScorer": (cur_path + "GeneralText/statistics/ngram_scorer.py", "NgramScorer"),
    "LexicalDiversityScorer": (cur_path + "GeneralText/statistics/lexical_diversity_scorer.py", "LexicalDiversityScorer"),
    "LangkitScorer": (cur_path + "GeneralText/statistics/langkit_scorer.py", "LangkitScorer"),
    
    "DeitaQualityScorer": (cur_path + "GeneralText/models/deita_quality_scorer.py", "DeitaQualityScorer"),
    "InstagScorer": (cur_path + "GeneralText/models/instag_scorer.py", "InstagScorer"),
    "DebertaV3Scorer": (cur_path + "GeneralText/models/debertav3_scorer.py", "DebertaV3Scorer"),
    "DeitaComplexityScorer": (cur_path + "GeneralText/models/deita_complexity_scorer.py", "DeitaComplexityScorer"),
    "FineWebEduScorer": (cur_path + "GeneralText/models/fineweb_edu_scorer.py", "FineWebEduScorer"),
    "PairQualScorer": (cur_path + "GeneralText/models/pair_qual_scorer.py", "PairQualScorer"),
    "PresidioScorer": (cur_path + "GeneralText/models/presidio_scorer.py", "PresidioScorer"),
    "RMScorer": (cur_path + "GeneralText/models/rm_scorer.py", "RMScorer"),
    "TextbookScorer": (cur_path + "GeneralText/models/textbook_scorer.py", "TextbookScorer"),
    "SuperfilteringScorer": (cur_path + "GeneralText/models/superfiltering_scorer.py", "SuperfilteringScorer"),
    "QuratingScorer": (cur_path + "GeneralText/models/qurating_scorer.py", "QuratingScorer"),
    "PerplexityScorer": (cur_path + "GeneralText/models/perplexity_scorer.py", "PerplexityScorer"),

    "CategoryInfo": (cur_path + "Reasoning/category_info.py", "CategoryInfo"),
    "DifficultyInfo": (cur_path + "Reasoning/difficulty_info.py", "DifficultyInfo"),
    "ToKenInfo": (cur_path + "Reasoning/TokenInfo.py", "token_info"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/eval/", _import_structure)
