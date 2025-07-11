import pandas as pd
import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import LangkitScorer

@OPERATOR_REGISTRY.register()
class LangkitFilter(OperatorABC):
    def __init__(self, 
                 min_scores = {
                    "flesch_reading_ease": 0,     # max(−144.8, 55.19−18.03)
                    "automated_readability_index": 0,  # max(0.9, 11.77−4.41)
                    "aggregate_reading_level": 0,  # max(0.0, 11.23−3.70)
                    "syllable_count": 32.0,         # max(32,   815.4−1516.6 → clip to 32)
                    "lexicon_count": 23.0,          # max(23,   524.2−1029.8 → clip to 23)
                    "sentence_count": 1.0,          # max(1,    29.0−60.1 → clip to 1)
                    "character_count": 118.0,       # max(118,  2610.2−4856.0 → clip to 118)
                    "letter_count": 109.0,          # max(109,  2513.5−4679.5 → clip to 109)
                    "polysyllable_count": 0.0,      # max(0,    78.9−137.5 → clip to 0)
                    "monosyllable_count": 13.0,     # max(13,   334.7−709.4 → clip to 13)
                    "difficult_words": 4.0,         # max(4,    93.4−120.0 → clip to 4)
                },
                max_scores = {
                    "flesch_reading_ease": 100,    # min(106.4, 55.19+18.03)
                    "automated_readability_index": 100, # min(98.2, 11.77+4.41)
                    "aggregate_reading_level": 100, # min(77.0, 11.23+3.70)
                    "syllable_count": 2331.9,       # min(43237, 815.4+1516.6)
                    "lexicon_count": 1554.0,        # min(33033, 524.2+1029.8)
                    "sentence_count": 89.1,         # min(2193,  29.0+60.1)
                    "character_count": 7466.3,      # min(139807,2610.2+4856.0)
                    "letter_count": 7193.0,         # min(134507,2513.5+4679.5)
                    "polysyllable_count": 216.4,    # min(3261,  78.9+137.5)
                    "monosyllable_count": 1044.1,   # min(25133,334.7+709.4)
                    "difficult_words": 213.4,       # min(2366,  93.4+120.0)
                },
                metrics_to_keep: list = [
                    "flesch_reading_ease",
                    "automated_readability_index",
                    "aggregate_reading_level",
                    "syllable_count",
                    "lexicon_count",
                    "sentence_count",
                    "character_count",
                    "letter_count",
                    "polysyllable_count",
                    "monosyllable_count",
                    "difficult_words",
                 ]):
        self.min_scores = min_scores
        self.max_scores = max_scores
        self.metric_name_map = {
            'flesch_reading_ease': 'LangkitFleschReadingEaseScore',
            'automated_readability_index': 'LangkitAutomatedReadabilityIndexScore',
            'aggregate_reading_level': 'LangkitAggregateReadingLevelScore',
            'syllable_count': 'LangkitSyllableCountScore',
            'lexicon_count': 'LangkitLexiconCountScore',
            'sentence_count': 'LangkitSentenceCountScore',
            'character_count': 'LangkitCharacterCountScore',
            'letter_count': 'LangkitLetterCountScore',
            'polysyllable_count': 'LangkitPolysyllableCountScore',
            'monosyllable_count': 'LangkitMonosyllableCountScore',
            'difficult_words': 'LangkitDifficultWordsScore'
        }
        if not self.min_scores.keys() == self.max_scores.keys():
            raise ValueError("min_scores and max_scores must have the same keys")  
        self.logger = get_logger()
        self.scorer = LangkitScorer()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_scores: {self.min_scores} and max_scores: {self.max_scores}...")
        
    def run(self, storage: DataFlowStorage, input_key: str, output_keys: list = ["flesch_reading_ease", "automated_readability_index", "aggregate_reading_level", "syllable_count", "lexicon_count", "sentence_count", "character_count", "letter_count", "polysyllable_count", "monosyllable_count", "difficult_words"]):
        self.input_key = input_key
        self.output_keys = output_keys
        if not list(self.min_scores.keys()) == output_keys:
            raise ValueError("min_scores and output_keys must have the same keys")  
        self.logger.info("Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        results = np.ones(len(dataframe), dtype=int)
        for _label in self.output_keys:
            label = self.metric_name_map[_label]
            min_score = self.min_scores[_label]
            max_score = self.max_scores[_label]
            dataframe[label] = pd.DataFrame(scores)[label]
            metric_scores = np.array(dataframe[label])
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            results = results & metric_filter.astype(int)
            self.logger.debug(f"Filtered by {_label}, {np.sum(results)} data remained")
            dataframe[f"{label}_label"] = metric_filter.astype(int)
        filtered_dataframe = dataframe[results == 1]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [f"{label}_label" for label in self.output_keys]

