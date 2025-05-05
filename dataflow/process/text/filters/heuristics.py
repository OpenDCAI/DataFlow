from typing import Callable, Tuple
from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
import re
from nltk.tokenize import word_tokenize, WordPunctTokenizer
from tqdm import tqdm

@PROCESSOR_REGISTRY.register()
class ColonEndFilter(TextFilter):
    # check whether the last char is ':'
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'ColonEndFilter'

    def filter_func(self, dataset):
        colon_end_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            colon_end_checks.append(not text.endswith(':'))
        return np.array(colon_end_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class WordNumberFilter(TextFilter):
    # check whether the number of word in [20, 100000]
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.min_words = args_dict.get('min_words')
        self.max_words = args_dict.get('max_words')
        self.filter_name = 'WordNumberFilter'

    def filter_func(self, dataset):
        word_counts = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            normalized_content = normalize(text)
            normalized_words = tuple(normalized_content.split())
            num_normalized_words = len(normalized_words)
            word_counts.append(num_normalized_words)

        word_counts = np.array(word_counts)
        metric_filter = (self.min_words <= word_counts) & (word_counts < self.max_words)
        return metric_filter.astype(int)

@PROCESSOR_REGISTRY.register()
class SentenceNumberFilter(TextFilter):
    # check whether the number of sentences in [3, 7500]
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.min_sentences = args_dict.get('min_sentences')
        self.max_sentences = args_dict.get('max_sentences')
        self.filter_name = 'SentenceNumberFilter'

    def filter_func(self, dataset):
        valid_check = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            SENT_PATTERN = re.compile(r'\b[^.!?\n]+[.!?]*', flags=re.UNICODE)
            num_sentence = len(SENT_PATTERN.findall(text))
            valid_check.append(num_sentence >= self.min_sentences and num_sentence <= self.max_sentences)

        return np.array(valid_check, dtype=int)

class TextSlice:
    # A slice of text from a document.
    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end

def split_paragraphs(
        text: str, normalizer: Callable[[str], str], remove_empty: bool = True
) -> Tuple[TextSlice]:
    """
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters, followed
    by a newline character, or a sequence of one or more characters, followed by the end of the string.
    """
    text_slices = tuple(
        TextSlice(normalizer(text[match.start():match.end()]), match.start(), match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    )

    if remove_empty is True:
        text_slices = tuple(
            text_slice for text_slice in text_slices if text_slice.text.strip()
        )

    return text_slices

def normalize(
        text: str,
        remove_punct: bool = True,
        lowercase: bool = True,
        nfd_unicode: bool = True,
        white_space: bool = True
) -> str:
    import string
    import unicodedata
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # lowercase
    if lowercase:
        text = text.lower()

    if white_space:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

    # NFD unicode normalization
    if nfd_unicode:
        text = unicodedata.normalize('NFD', text)

    return text

@PROCESSOR_REGISTRY.register()
class LineEndWithEllipsisFilter(TextFilter):
    # check whether lines end with ellipsis ratio < 0.3
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'LineEndWithEllipsisFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        ellipsis_checks = []
        ellipsis = ["...", "…"]
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            raw_lines: Tuple[TextSlice] = split_paragraphs(
                text=text, normalizer=lambda x: x, remove_empty=True
            )
            num_lines = len(raw_lines)
            if num_lines == 0:
                ellipsis_checks.append(False)
                continue

            num_occurrences = sum([line.text.rstrip().endswith(tuple(ellipsis)) for line in raw_lines])
            ratio = num_occurrences / num_lines
            ellipsis_checks.append(ratio < self.threshold)

        return np.array(ellipsis_checks, dtype=int)


@PROCESSOR_REGISTRY.register()
class LineEndWithTerminalFilter(TextFilter):
    # check whether lines end with terminal punctuation mark ratio > 0.6
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'LineEndWithTerminalFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        
        terminal_checks = []
        ternimal = [".", "!", "?", "”", "\""]
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            raw_lines: Tuple[TextSlice] = split_paragraphs(
                text=text, normalizer=lambda x: x, remove_empty=True
            )
            num_lines = len(raw_lines)
            if num_lines == 0:
                terminal_checks.append(False)
                continue
            num_occurrences = sum([line.text.rstrip().endswith(tuple(ternimal)) for line in raw_lines])
            ratio = num_occurrences / num_lines
            terminal_checks.append(ratio > self.threshold)

        return np.array(terminal_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class ContentNullFilter(TextFilter):
    # check whether content is null
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'ContentNullFilter'

    def filter_func(self, dataset):
        null_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            null_checks.append(text is not None and text.strip() != '')
        return np.array(null_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class SymbolWordRatioFilter(TextFilter):
    # check whether the ratio of symbols / words is < 0.4
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.threshold = args_dict.get('threshold')
        self.filter_name = 'SymbolWordRatioFilter'
        self.symbol = ["#", "...", "…"]

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            raw_words = tuple(WordPunctTokenizer().tokenize(text))
            num_raw_words = len(raw_words)

            num_words = num_raw_words
            num_symbols = float(sum(
                text.count(x) for x in self.symbol
            ))
            if num_words == 0:
                valid_checks.append(False)
                continue
            ratio = num_symbols / num_words
            valid_checks.append(ratio < self.threshold)

        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class AlphaWordsFilter(TextFilter):
    # check whether the ratio of words that contain at least one alphabetic character > 0.6
    def __init__(self, args_dict: dict):
        import nltk
        nltk.download('punkt_tab')
        super().__init__(args_dict)
        self.threshold = args_dict.get('threshold')
        self.filter_name = 'AlphaWordsFilter'
        self.use_tokenizer = args_dict.get('use_tokenizer')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            if self.use_tokenizer:
                words = word_tokenize(text)
            else:
                words = text.split()
            alpha_count = sum(1 for word in words if re.search(r'[a-zA-Z]', word))
            word_count = len(words)
            if word_count > 0:
                ratio = alpha_count / word_count
                valid_checks.append(ratio > self.threshold)
            else:
                valid_checks.append(False)

        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class HtmlEntityFilter(TextFilter):
    # Check whether content has HTML entity
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'HtmlEntityFilter'

    def filter_func(self, dataset):
        valid_checks = []
        html_entity = ["nbsp", "lt", "gt", "amp", "quot", "apos", "hellip", "ndash", "mdash", "lsquo", "rsquo", "ldquo", "rdquo"]
        full_entities_1 = [f"&{entity}；" for entity in html_entity]
        full_entities_2 = [f"&{entity};" for entity in html_entity]
        full_entities_3 = [f"＆{entity};" for entity in html_entity]
        full_entities_4 = [f"＆{entity}；" for entity in html_entity]
        half_entities = [f"＆{entity}" for entity in html_entity] + [f"&{entity}" for entity in html_entity]
        all_entities = full_entities_1 + full_entities_2 + full_entities_3 + full_entities_4 + half_entities
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            content = data.get(dataset.keys)  
            has_html_entity = any(entity in content for entity in all_entities)
            valid_checks.append(not has_html_entity)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class IDCardFilter(TextFilter):
    # check if the content contains ID card.
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'IDCardFilter'

    def filter_func(self, dataset):
        valid_checks = []
        pattern = re.compile(r"(身\s{0,10}份|id\s{0,10}number\s{0,10}|identification|identity|\s{0,10}ID\s{0,10}No\s{0,10}|id\s{0,10}card\s{0,10}|NRIC\s{0,10}number\s{0,10}|IC\s{0,10}number\s{0,10}|resident\s{0,10}registration\s{0,10}|I.D.\s{0,10}Number\s{0,10})", re.I)
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            has_id_card = bool(pattern.search(text))
            valid_checks.append(not has_id_card)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class NoPuncFilter(TextFilter):
    # check whether content has paragraph without punctuations
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'NoPuncFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            paragraphs = text.split('\n')
            max_word_count = 0
            for paragraph in paragraphs:
                if len(paragraph.strip()) == 0:
                    continue
                sentences = re.split("[–.!?,;•/|…]", paragraph)
                for sentence in sentences:
                    words = sentence.split()
                    word_count = len(words)
                    if word_count > max_word_count:
                        max_word_count = word_count

            valid_checks.append(int(max_word_count) <= self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class SpecialCharacterFilter(TextFilter):
    # check whether content has special characters.
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'SpecialCharacterFilter'

    def filter_func(self, dataset):
        valid_checks = []
        speclai_character = [
            r"u200e",
            r"&#247;|\? :",
            r"[�□]|\{\/U\}",
            r"U\+26[0-F][0-D]|U\+273[3-4]|U\+1F[3-6][0-4][0-F]|U\+1F6[8-F][0-F]"
        ]
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            has_special_character = any(re.search(pattern, text) for pattern in speclai_character)
            valid_checks.append(not has_special_character)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class WatermarkFilter(TextFilter):
    # check whether content has watermarks.
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'WatermarkFilter'
        self.watermarks = args_dict.get('watermarks')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            content = data.get(dataset.keys)
            matches = re.search('|'.join(self.watermarks), content)
            valid_checks.append(matches is None)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class MeanWordLengthFilter(TextFilter):
    # check whether the mean length of word in [3, 10] 
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'MeanWordLengthFilter'
        self.min_length = args_dict.get('min_length')
        self.max_length = args_dict.get('max_length')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            normalized_content = normalize(text)
            normalized_words = tuple(normalized_content.split())
            num_normalized_words = len(normalized_words)
            if num_normalized_words == 0:
                valid_checks.append(False)
                continue

            num_chars = float(sum(map(len, normalized_words)))
            mean_length = num_chars / num_normalized_words
            mean_length = round(mean_length, 2)
            valid_checks.append(self.min_length <= mean_length < self.max_length)

        return np.array(valid_checks, dtype=int)



@PROCESSOR_REGISTRY.register()
class StopWordFilter(TextFilter):
    # Check whether the ratio of stop words > 6%
    def __init__(self, args_dict: dict):
        import nltk
        super().__init__(args_dict)
        self.filter_name = 'StopWordFilter'
        self.use_tokenizer = args_dict.get('use_tokenizer')
        self.threshold = args_dict.get('threshold')
        nltk.data.path.append(args_dict.get('model_cache_dir'))
        nltk.download('stopwords', download_dir=args_dict.get('model_cache_dir'))

    def filter_func(self, dataset):
        from nltk.corpus import stopwords
        valid_checks = []
        stw = stopwords.words('english')
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys) 
            if self.use_tokenizer:
                words = word_tokenize(text.lower())
            else:
                words = text.lower().split()
            num_raw_words = len(words)
            num_stop_words = sum(
                map(lambda w: w in stw, words)
            )
            ratio = num_stop_words / num_raw_words if num_raw_words > 0 else 0
            valid_checks.append(ratio > self.threshold and num_stop_words > 2)

        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class CurlyBracketFilter(TextFilter):
    # check whether content contains curly brackets: { or }
    
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'CurlyBracketFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            num = text.count('{') + text.count('}')
            ratio = num / len(text) if len(text) !=0 else 0
            valid_checks.append(ratio < self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class CapitalWordsFilter(TextFilter):
    # check whether capital words ratio > 0.2
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'CapitalWordsFilter'
        self.threshold = args_dict.get('threshold')
        self.use_tokenizer = args_dict.get('use_tokenizer')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            if self.use_tokenizer:
                words = word_tokenize(text)
            else:
                words = text.split()
            num_words = len(words)
            num_caps_words = sum(map(str.isupper, words))
            ratio = num_caps_words / num_words
            valid_checks.append(ratio <= self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class LoremIpsumFilter(TextFilter):
    # check whether the ratio of lorem ipsum < 3e-08
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'LoremIpsumFilter'
        self.threshold = float(args_dict.get('threshold'))

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            normalized_content = normalize(text)
            num_normalized_content = len(normalized_content)
            SEARCH_REGEX = re.compile(r"lorem ipsum", re.IGNORECASE)
            num_occurrences = len(SEARCH_REGEX.findall(normalized_content))
            ratio = num_occurrences / num_normalized_content
            valid_checks.append(ratio <= self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class UniqueWordsFilter(TextFilter):
    # check whether the ratio of unique words > 0.1
    
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'UniqueWordsFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            normalized_text = normalize(text)
            normalized_words = tuple(normalized_text.split())
            num_normalized_words = len(normalized_words)
            if num_normalized_words == 0:
                valid_checks.append(False)
                continue
            num_words = num_normalized_words
            num_unique_words = len(set(normalized_words))
            ratio = num_unique_words / num_words
            valid_checks.append(ratio > self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class CharNumberFilter(TextFilter):
    # check whether the number of characters > 100

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'CharNumberFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            text = text.strip()
            text = text.replace(" ", "")
            text = text.replace("\n", "")
            text = text.replace("\t", "")
            num_char = len(text)
            valid_checks.append(num_char >= self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class LineStartWithBulletpointFilter(TextFilter):
    # check whether lines start with bullet points.
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'LineStartWithBulletpointFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        valid_checks = []
        key_list =  [
            "\u2022",  # bullet point
            "\u2023",  # triangular bullet point
            "\u25B6",  # black right pointing triangle
            "\u25C0",  # black left pointing triangle
            "\u25E6",  # white bullet point
            "\u25A0",  # black square
            "\u25A1",  # white square
            "\u25AA",  # black small square
            "\u25AB",  # white small square
            "\u2013",  # en dash
            ]
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            raw_lines: Tuple[TextSlice] = split_paragraphs(
                text=text, normalizer=lambda x: x, remove_empty=True
                )
            num_lines = len(raw_lines)
            if num_lines == 0:
                valid_checks.append(False)
                continue
            num_occurrences = sum([line.text.lstrip().startswith(tuple(key_list)) for line in raw_lines])
            ratio = num_occurrences / num_lines
            valid_checks.append(ratio <= self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class LineWithJavascriptFilter(TextFilter):
    # check whether lines contain the word 'Javascript'.
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'LineWithJavascriptFilter'
        self.threshold = args_dict.get('threshold')

    def filter_func(self, dataset):
        valid_checks = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            normalized_lines: Tuple[TextSlice] = split_paragraphs(
                text=text, normalizer=normalize, remove_empty=True
            )
            num_lines = len(normalized_lines)
            if num_lines == 0:
                valid_checks.append(False)
                continue
            num_occurrences = sum(['javascript' in line.text for line in normalized_lines])
            num_not_occur = num_lines - num_occurrences
            valid_checks.append(num_lines<=3 or num_not_occur >= self.threshold)
        return np.array(valid_checks, dtype=int)

@PROCESSOR_REGISTRY.register()
class BlocklistFilter(TextFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.language = args_dict['language']
        self.threshold = args_dict['threshold']
        self.use_tokenizer = args_dict['use_tokenizer']
        self.blocklist = self.load_blocklist()
        self.filter_name = 'BlocklistFilter'
    
    def load_blocklist(self):
        file_path = f"./dataflow/process/text/filters/blocklist/{self.language}.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            return set(line.strip().lower() for line in file if line.strip())

    def filter_func(self, dataset):
        filtered_results = []
        for data in tqdm(dataset, desc=f"Implementing {self.filter_name}"):
            text = data.get(dataset.keys)
            if self.use_tokenizer:
                text = word_tokenize(text.lower())
            else:
                text = text.lower().split()
            blocklist_count = sum(1 for word in text if word in self.blocklist)
            filtered_results.append(blocklist_count <= self.threshold)
        return np.array(filtered_results).astype(int)
