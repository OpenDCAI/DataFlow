# from .apicaller.alpagasus_scorer import AlpagasusScorer
# from .apicaller.perspective_scorer import PerspectiveScorer
# from .apicaller.treeinstruct_scorer import TreeinstructScorer

# from .models.unieval_scorer import UnievalScorer
# from .models.instag_scorer import InstagScorer
# from .models.textbook_scorer import TextbookScorer
# from .models.fineweb_edu_scorer import FineWebEduScorer
# from .models.debertav3_scorer import DebertaV3Scorer
# from .models.perplexity_scorer import PerplexityScorer
# from .models.qurating_scorer import QuratingScorer
# from .models.superfiltering_scorer import SuperfilteringScorer
# from .models.deita_quality_scorer import DeitaQualityScorer
# from .models.deita_complexity_scorer import DeitaComplexityScorer
# from .models.presidio_scorer import PresidioScorer
# from .models.rm_scorer import RMScorer

# from .diversity.vendi_scorer import VendiScorer
# from .diversity.task2vec_scorer import Task2VecScorer

# from .statistics.langkit_scorer import LangkitScorer
# from .statistics.lexical_diversity_scorer import LexicalDiversityScorer
# from .statistics.ngram_scorer import NgramScorer


# __all__ = [
#     'AlpagasusScorer',
#     'PerspectiveScorer',
#     'TreeinstructScorer',
#     'UnievalScorer',
#     'InstagScorer',
#     'TextbookScorer',
#     'FineWebEduScorer',
#     'VendiScorer',
#     'Task2VecScorer',
#     'LangkitScorer',
#     'LexicalDiversityScorer',
#     'NgramScorer',
#     'DebertaV3Scorer',
#     'PerplexityScorer',
#     'QuratingScorer',
#     'SuperfilteringScorer',
#     'DeitaQualityScorer',
#     'DeitaComplexityScorer',
#     'PresidioScorer',
#     'RMScorer'
# ]

# from .gen.bleuscorer import BleuScorer
# from .gen.ciderscorer import CiderScorer
# from .gen.meteorscorer import MeteorScorer
# from .gen.rougescorer import RougeScorer
# from .gen.spicescorer import SpiceScorer
# from .gen.wsdscorer import WsdScorer
# from .gen.hleporscorer import HLEPORScorer
# from .gen.chrfscorer import CHRFScorer
# from .gen.chrfppscorer import CHRFppScorer


# __all__ = [
#     # 'AlpagasusScorer',
#     # 'PerspectiveScorer',
#     # 'TreeinstructScorer',
#     # 'UnievalScorer',
#     # 'InstagScorer',
#     # 'TextbookScorer',
#     # 'FineWebEduScorer',
#     # 'VendiScorer',
#     # 'Task2VecScorer',
#     # 'LangkitScorer',
#     # 'LexicalDiversityScorer',
#     # 'NgramScorer',
#     # 'DebertaV3Scorer',
#     # 'PerplexityScorer',
#     # 'QuratingScorer',
#     # 'SuperfilteringScorer',
#     # 'DeitaQualityScorer',
#     # 'DeitaComplexityScorer',
#     # 'PresidioScorer',
#     # 'RMScorer',
#     'BleuScorer',
#     'CiderScorer',
#     'MeteorScorer',
#     'RougeScorer',
#     'SpiceScorer',
#     'WsdScorer',
#     'HLEPORScorer',
#     'TERScorer',
#     'CHRFScorer',
#     'CHRFppScorer'
# ]


import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {  
    'AlpagasusScorer': ('dataflow/Eval/Text/apicaller/alpagasus_scorer.py', 'AlpagasusScorer'),  
    'PerspectiveScorer': ('dataflow/Eval/Text/apicaller/perspective_scorer.py', 'PerspectiveScorer'),  
    'TreeinstructScorer': ('dataflow/Eval/Text/apicaller/treeinstruct_scorer.py', 'TreeinstructScorer'),  
    'UnievalScorer': ('dataflow/Eval/Text/models/unieval_scorer.py', 'UnievalScorer'),  
    'InstagScorer': ('dataflow/Eval/Text/models/instag_scorer.py', 'InstagScorer'),  
    'TextbookScorer': ('dataflow/Eval/Text/models/textbook_scorer.py', 'TextbookScorer'),  
    'FineWebEduScorer': ('dataflow/Eval/Text/models/fineweb_edu_scorer.py', 'FineWebEduScorer'),  
    'DebertaV3Scorer': ('dataflow/Eval/Text/models/debertav3_scorer.py', 'DebertaV3Scorer'),  
    'PerplexityScorer': ('dataflow/Eval/Text/models/perplexity_scorer.py', 'PerplexityScorer'),  
    'QuratingScorer': ('dataflow/Eval/Text/models/qurating_scorer.py', 'QuratingScorer'),  
    'SuperfilteringScorer': ('dataflow/Eval/Text/models/superfiltering_scorer.py', 'SuperfilteringScorer'),  
    'DeitaQualityScorer': ('dataflow/Eval/Text/models/deita_quality_scorer.py', 'DeitaQualityScorer'),  
    'DeitaComplexityScorer': ('dataflow/Eval/Text/models/deita_complexity_scorer.py', 'DeitaComplexityScorer'),  
    'PresidioScorer': ('dataflow/Eval/Text/models/presidio_scorer.py', 'PresidioScorer'),  
    'RMScorer': ('dataflow/Eval/Text/models/rm_scorer.py', 'RMScorer'),  
    'VendiScorer': ('dataflow/Eval/Text/diversity/vendi_scorer.py', 'VendiScorer'),  
    'Task2VecScorer': ('dataflow/Eval/Text/diversity/task2vec_scorer.py', 'Task2VecScorer'),  
    'LangkitScorer': ('dataflow/Eval/Text/statistics/langkit_scorer.py', 'LangkitScorer'),  
    'LexicalDiversityScorer': ('dataflow/Eval/Text/statistics/lexical_diversity_scorer.py', 'LexicalDiversityScorer'),  
    'NgramScorer': ('dataflow/Eval/Text/statistics/ngram_scorer.py', 'NgramScorer'),  

    'BleuScorer': ('dataflow/Eval/Text/gen/bleu_scorer.py', 'BleuScorer'),  
    'CiderScorer': ('dataflow/Eval/Text/gen/cider_scorer.py', 'CiderScorer'),  
    'MeteorScorer': ('dataflow/Eval/Text/gen/meteor_scorer.py', 'MeteorScorer'),  
    'RougeScorer': ('dataflow/Eval/Text/gen/rouge_scorer.py', 'RougeScorer'),   
    'WsdScorer': ('dataflow/Eval/Text/gen/wsd_scorer.py', 'WsdScorer'),  
    'HLEPORScorer': ('dataflow/Eval/Text/gen/hlepor_scorer.py', 'HLEPORScorer'),  
    'TERScorer': ('dataflow/Eval/Text/gen/ter_scorer.py', 'TERScorer'),  
    'CHRFScorer': ('dataflow/Eval/Text/gen/chrf_scorer.py', 'CHRFScorer'),  
    'BERTScoreScorer':('dataflow/Eval/Text/gen/bert_scorer.py','BERTScoreScorer'),
    'BARTScorer':('dataflow/Eval/Text/gen/bart_scorer.py','BARTScorer'),
    'BleurtScorer':('dataflow/Eval/Text/gen/bleurt_scorer.py','BleurtScorer'),
    'EmbeddingAverageScorer':('dataflow/Eval/Text/gen/embedding_average_scorer.py','EmbeddingAverageScorer'),
    'GreedyMatchingScorer': ('dataflow/Eval/Text/gen/greedy_matching_scorer.py','GreedyMatchingScorer'),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/Eval/Text", _import_structure)
