from dataflow.operators.filter import (
    MinHashDeduplicator,
    LanguageFilter,
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
    CurlyBracketFilter,
    CapitalWordsFilter,
    LoremIpsumFilter,
    UniqueWordsFilter,
    CharNumberFilter,
    LineStartWithBulletpointFilter,
    LineWithJavascriptFilter,
    PairQualFilter
)
from dataflow.operators.refine import (
    HtmlUrlRemoverRefiner,
    RemoveEmojiRefiner,
    RemoveExtraSpacesRefiner
)

from dataflow.utils.storage import FileStorage

class PTTextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.language_filter = LanguageFilter(allowed_languages = '__label__eng_Latn', model_cache_dir = self.model_cache_dir)        
        self.remove_extra_spaces_refiner = RemoveExtraSpacesRefiner()
        self.remove_emoji_refiner = RemoveEmojiRefiner()
        self.html_remove_refiner = HtmlUrlRemoverRefiner()
        self.minhash_deduplicator = MinHashDeduplicator(num_perm=128, threshold=0.9, use_n_gram=True, ngram=5)
        self.blocklist_filter = BlocklistFilter()
        self.word_number_filter = WordNumberFilter(min_words=20, max_words=100000)
        self.colon_end_filter = ColonEndFilter()
        self.sentence_number_filter = SentenceNumberFilter(min_sentences=3, max_sentences=7500)
        self.line_end_with_ellipsis_filter = LineEndWithEllipsisFilter(threshold=0.3)
        self.content_null_filter = ContentNullFilter()
        self.mean_word_length_filter = MeanWordLengthFilter(min_length=3, max_length=10)
        self.symbol_word_ratio_filter = SymbolWordRatioFilter(threshold=0.4)
        self.html_entity_filter = HtmlEntityFilter()
        self.id_card_filter = IDCardFilter(threshold=3)
        self.no_punc_filter = NoPuncFilter(threshold=112)
        self.special_character_filter = SpecialCharacterFilter()
        self.watermark_filter = WatermarkFilter(watermarks=['Copyright', 'Watermark', 'Confidential'])
        self.curly_bracket_filter = CurlyBracketFilter(threshold=0.025)
        self.capital_words_filter = CapitalWordsFilter(threshold=0.2, use_tokenizer=False)
        self.lorem_ipsum_filter = LoremIpsumFilter(threshold=3e-8)
        self.unique_words_filter = UniqueWordsFilter(threshold=0.1)
        self.char_number_filter = CharNumberFilter(threshold=100)
        self.line_start_with_bulletpoint_filter = LineStartWithBulletpointFilter(threshold=0.9)
        self.line_with_javascript_filter = LineWithJavascriptFilter(threshold=3)
        self.quality_filter = PairQualFilter(min_score=0, max_score=10000, lang='en')
    
    def forward(self):
        # Initial filters
        self.language_filter.run(
            storage = self.storage.step(),
            input_key = "raw_content"
        )
        # refiners
        self.remove_extra_spaces_refiner.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.remove_emoji_refiner.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.html_remove_refiner.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )
        self.minhash_deduplicator.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.blocklist_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.word_number_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.colon_end_filter.run(
            storage = self.storage.step(),
            input_key = 'raw_content'
        )
        self.sentence_number_filter.run(
            storage = self.storage.step(),
            input_key = 'raw_content'
        )
        self.line_end_with_ellipsis_filter.run(
            storage = self.storage.step(),
            input_key = 'raw_content'
        )
        self.content_null_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.mean_word_length_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.symbol_word_ratio_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.html_entity_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.id_card_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.no_punc_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.special_character_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.watermark_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.curly_bracket_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.capital_words_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.lorem_ipsum_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.unique_words_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.char_number_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.line_start_with_bulletpoint_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.line_with_javascript_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        self.quality_filter.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = PTTextPipeline()
    model.forward()
