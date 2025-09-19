import pandas as pd
from dataflow.utils.storage import DataFlowStorage
from dataflow.operators.code.doc_quality_filter import DocQualityFilter

# Construct test data
sample_data = [
    {
        'content': (
            'This is a normal English document. It has enough words and sentences to pass most filters. '
            'There are no duplicate lines or excessive n-grams. The quick brown fox jumps over the lazy dog. '
            'Python is a popular programming language. This document is for testing the DocQualityFilter.'
        ),
        'filename': 'normal.txt',
        'language': 'en',
    },
    {
        'content': (
            '这是一个正常的中文文档。它包含足够的字数和句子，不会被过滤掉。'
            '没有重复的行，也没有过多的n-gram。用于测试文档质量过滤器。'
        ),
        'filename': 'normal_zh.txt',
        'language': 'zh',
    },
    {
        'content': 'A short doc.',
        'filename': 'readme.md',
        'language': 'en',
    },
    {
        'content': 'DUPLICATE LINE\n' * 50 + 'Unique line at the end.',
        'filename': 'duplicate_lines.txt',
        'language': 'en',
    },
    {
        'content': 'ALL CAPS WORDS HERE! THIS IS AN EXAMPLE WITH MANY CAPS.',
        'filename': 'all_caps.txt',
        'language': 'en',
    },
    {
        'content': 'Low entropy text with some common words.',
        'filename': 'low_entropy.txt',
        'language': 'en',
    }
]
df = pd.DataFrame(sample_data)

# Use in-memory DataFlowStorage
class DummyStorage(DataFlowStorage):
    def __init__(self, df):
        self._df = df
        self._written = None
    def read(self, key):
        return self._df
    def write(self, df):
        self._written = df
        return 'dummy_output.csv'
    def get_keys_from_dataframe(self, df):
        # Minimal implementation for abstract method
        return []

storage = DummyStorage(df)

# Configure thresholds
thresholds = {
    'min_num_chars': 1,
    'max_num_chars': 1000,
    'min_num_words': 4,
    'max_num_words': 100,
    'max_frac_duplicate_lines': 0.5,
    'max_frac_duplicate_2gram': 1,
    'max_frac_duplicate_3gram': 1,
    'max_frac_duplicate_5gram': 1,
    'max_frac_curly_bracket': 1,
    'max_frac_all_caps_words': 1,
    'min_entropy_unigram': 0,
}

# Run the filter
op = DocQualityFilter()
op.run(storage, thresholds=thresholds)

# Output filtering results
test_result = storage._written
print('Filtered DataFrame:')
print(test_result)
