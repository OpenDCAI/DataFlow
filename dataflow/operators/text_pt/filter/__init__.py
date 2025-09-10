# import sys
# from dataflow.utils.registry import LazyLoader

# cur_path = "dataflow/operators/filter/"

# _import_structure = {
#     # Primary filters
#     "NgramFilter": (cur_path + "ngram_filter.py", "NgramFilter"),
#     "LanguageFilter": (cur_path + "language_filter.py", "LanguageFilter"),
#     "DeitaQualityFilter": (cur_path + "deita_quality_filter.py", "DeitaQualityFilter"),
#     "DeitaComplexityFilter": (cur_path + "deita_complexity_filter.py", "DeitaComplexityFilter"),
#     "InstagFilter": (cur_path + "instag_filter.py", "InstagFilter"),
#     "PairQualFilter": (cur_path + "pair_qual_filter.py", "PairQualFilter"),
#     "QuratingFilter": (cur_path + "qurating_filter.py", "QuratingFilter"),
#     "SuperfilteringFilter": (cur_path + "superfiltering_filter.py", "SuperfilteringFilter"),
#     "FineWebEduFilter": (cur_path + "fineweb_edu_filter.py", "FineWebEduFilter"),
#     "TextbookFilter": (cur_path + "text_book_filter.py", "TextbookFilter"),
#     "AlpagasusFilter": (cur_path + "alpagasus_filter.py", "AlpagasusFilter"),
#     "DebertaV3Filter": (cur_path + "debertav3_filter.py", "DebertaV3Filter"),
#     "LangkitFilter": (cur_path + "langkit_filter.py", "LangkitFilter"),
#     "LexicalDiversityFilter": (cur_path + "lexical_diversity_filter.py", "LexicalDiversityFilter"),
#     "PerplexityFilter": (cur_path + "perplexity_filter.py", "PerplexityFilter"),
#     "PerspectiveFilter": (cur_path + "perspective_filter.py", "PerspectiveFilter"),
#     "PresidioFilter": (cur_path + "presidio_filter.py", "PresidioFilter"),
#     "RMFilter": (cur_path + "reward_model_filter.py", "RMFilter"),
#     "TreeinstructFilter": (cur_path + "treeinstruct_filter.py", "TreeinstructFilter"),

#     # Heuristic filters
#     "ColonEndFilter": (cur_path + "heuristics.py", "ColonEndFilter"),
#     "WordNumberFilter": (cur_path + "heuristics.py", "WordNumberFilter"),
#     "BlocklistFilter": (cur_path + "heuristics.py", "BlocklistFilter"),
#     "SentenceNumberFilter": (cur_path + "heuristics.py", "SentenceNumberFilter"),
#     "LineEndWithEllipsisFilter": (cur_path + "heuristics.py", "LineEndWithEllipsisFilter"),
#     "ContentNullFilter": (cur_path + "heuristics.py", "ContentNullFilter"),
#     "MeanWordLengthFilter": (cur_path + "heuristics.py", "MeanWordLengthFilter"),
#     "SymbolWordRatioFilter": (cur_path + "heuristics.py", "SymbolWordRatioFilter"),
#     "HtmlEntityFilter": (cur_path + "heuristics.py", "HtmlEntityFilter"),
#     "IDCardFilter": (cur_path + "heuristics.py", "IDCardFilter"),
#     "NoPuncFilter": (cur_path + "heuristics.py", "NoPuncFilter"),
#     "SpecialCharacterFilter": (cur_path + "heuristics.py", "SpecialCharacterFilter"),
#     "WatermarkFilter": (cur_path + "heuristics.py", "WatermarkFilter"),
#     "StopWordFilter": (cur_path + "heuristics.py", "StopWordFilter"),
#     "CurlyBracketFilter": (cur_path + "heuristics.py", "CurlyBracketFilter"),
#     "CapitalWordsFilter": (cur_path + "heuristics.py", "CapitalWordsFilter"),
#     "LoremIpsumFilter": (cur_path + "heuristics.py", "LoremIpsumFilter"),
#     "UniqueWordsFilter": (cur_path + "heuristics.py", "UniqueWordsFilter"),
#     "CharNumberFilter": (cur_path + "heuristics.py", "CharNumberFilter"),
#     "LineStartWithBulletpointFilter": (cur_path + "heuristics.py", "LineStartWithBulletpointFilter"),
#     "LineWithJavascriptFilter": (cur_path + "heuristics.py", "LineWithJavascriptFilter"),

#     # Deduplicators
#     "MinHashDeduplicator": (cur_path + "minhash_deduplicator.py", "MinHashDeduplicator"),
#     "CCNetDeduplicator": (cur_path + "ccnet_deduplicator.py", "CCNetDeduplicator"),
#     "HashDeduplicator": (cur_path + "hash_deduplicator.py", "HashDeduplicator"),
#     "NgramHashDeduplicator": (cur_path + "ngramhash_deduplicator.py", "NgramHashDeduplicator"),
#     "SemDeduplicator": (cur_path + "sem_deduplicator.py", "SemDeduplicator"),
#     "SimHashDeduplicator": (cur_path + "simhash_deduplicator.py", "SimHashDeduplicator"),
# }

# sys.modules[__name__] = LazyLoader(__name__, cur_path, _import_structure)