from dataflow.operators.refine import (
    HtmlEntityRefiner,
    LowercaseRefiner,
    NERRefiner,
    PIIAnonymizeRefiner,
    ReferenceRemoverRefiner,
    RemoveContractionsRefiner,
    RemoveEmoticonsRefiner,
    RemoveImageRefsRefiner,
    RemoveNumberRefiner,
    RemovePunctuationRefiner,
    RemoveRepetitionsPunctuationRefiner,
    RemoveStopwordsRefiner,
    SpellingCorrectionRefiner,
    StemmingLemmatizationRefiner,
    TextNormalizationRefiner
)
from dataflow.utils.storage import FileStorage

class RefinePipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.html_entity_refiner = HtmlEntityRefiner()
        self.lower_case_refiner = LowercaseRefiner()
        self.ner_refiner = NERRefiner()
        self.pii_anonymize_refiner = PIIAnonymizeRefiner()
        self.reference_remover_refiner = ReferenceRemoverRefiner()
        self.remove_contraction_refiner = RemoveContractionsRefiner()
        self.remove_emoticons_refiner = RemoveEmoticonsRefiner()
        self.remove_image_ref_refiner = RemoveImageRefsRefiner()
        self.remove_number_refiner = RemoveNumberRefiner()
        self.remove_punctuation_refiner = RemovePunctuationRefiner()
        self.remove_repetitions_refiner = RemoveRepetitionsPunctuationRefiner()
        # self.remove_stopwords_refiner = RemoveStopwordsRefiner()
        # self.spelling_correction_refiner = SpellingCorrectionRefiner()
        # self.stemming_lemmatization_refiner = StemmingLemmatizationRefiner()
        # self.text_normalization_refiner = TextNormalizationRefiner()
        
    def forward(self):
        
        self.html_entity_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.lower_case_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.ner_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.pii_anonymize_refiner.run(
            self.storage.step(),
            input_key='url'
        )
        self.reference_remover_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.remove_contraction_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.remove_emoticons_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.remove_image_ref_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.remove_number_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.remove_punctuation_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.remove_repetitions_refiner.run(
            self.storage.step(),
            input_key='raw_content'
        )
        # self.remove_stopwords_refiner.run(
        #     self.storage.step(),
        #     input_key='raw_content'
        # )
        # self.spelling_correction_refiner.run(
        #     self.storage.step(),
        #     input_key='raw_content'
        # )
        # self.stemming_lemmatization_refiner.run(
        #     self.storage.step(),
        #     input_key='raw_content'
        # )
        # self.text_normalization_refiner.run(
        #     self.storage.step(),
        #     input_key='raw_content'
        # )
        
if __name__ == "__main__":
    pipeline = RefinePipeline()
    pipeline.forward()