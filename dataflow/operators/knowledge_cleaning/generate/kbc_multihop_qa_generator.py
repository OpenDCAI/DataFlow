from dataflow.prompts.multihopqa import MultiHopQAGeneratorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import random
from typing import Any, Dict, List, Optional, Sequence
import json
from tqdm import tqdm
import re

@OPERATOR_REGISTRY.register()
class KBCMultiHopQAGenerator(OperatorABC):
    r"""A processor for generating multi-hop question-answer pairs from user
    data.

    This class handles the processing of text data to generate multi-hop
    question-answer pairs using either an AI model or rule-based approaches.
    It manages the entire pipeline from text preprocessing to dataset curation.
    """

    def __init__(self,
                 llm_serving: LLMServingABC,
                 seed: int = 0,
                 lang="en",
                 prompt_template = None
                 ):
        r"""Initialize the UserDataProcessor.

        Args:
            config (Optional[ProcessorConfig], optional): Configuration for
                data processing. (default: :obj:`None`)
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = MultiHopQAGeneratorPrompt()

    @staticmethod
    def get_desc(lang: str = "zh") -> tuple:
        """Returns a description of the processor's functionality.

        Args:
            lang (str, optional): Language for description ('zh' or 'en').

        Returns:
            tuple: Description strings in specified language, including format example
        """
        if lang == "zh":
            return (
                "MultiHopQAGenerator 是多跳问答对生成处理器，支持从文本中自动生成需要多步推理的问题与答案。",
                "处理流程包括：文本预处理、信息抽取、问题生成与回答生成，支持自定义语言模型后端和参数。",
                "输出格式如下：",
                "输入：\n"
                "text: <原始上下文文本>",
                "输出：\n"
                "{\n"
                "  \"text\": <处理后的文本字符串>,\n"
                "  \"qa_pairs\": [\n"
                "    {\n"
                "      \"question\": <字符串：生成的问题>,\n"
                "      \"reasoning_steps\": [\n"
                "        {\"step\": <推理过程的步骤 1>},\n"
                "        {\"step\": <步骤 2>} ...\n"
                "      ],\n"
                "      \"answer\": <字符串：最终答案>,\n"
                "      \"supporting_facts\": [<支持该答案的事实 1>, <事实 2>, ...],\n"
                "      \"type\": <可选：问题类型，如“生物学”、“历史”等>\n"
                "    },\n"
                "    ...\n"
                "  ],\n"
                "  \"metadata\": {\n"
                "    \"source\": <数据来源>,\n"
                "    \"timestamp\": <时间戳字符串>,\n"
                "    \"complexity\": <整数：问题复杂度标记>\n"
                "  }\n"
                "}"
            )
        else:
            return (
                "MultiHopQAGenerator is a processor for generating multi-hop question-answer pairs from raw text.",
                "It includes preprocessing, information extraction, and reasoning-based QA generation, with configurable LLM backends.",
                "Expected output format:",
                "Input:\n"
                "text: <raw input context>",
                "Output:\n"
                "{\n"
                "  \"text\": <processed input text>,\n"
                "  \"qa_pairs\": [\n"
                "    {\n"
                "      \"question\": <string: generated question>,\n"
                "      \"reasoning_steps\": [\n"
                "        {\"step\": <inference step 1>},\n"
                "        {\"step\": <inference step 2>} ...\n"
                "      ],\n"
                "      \"answer\": <string: final answer>,\n"
                "      \"supporting_facts\": [<fact 1>, <fact 2>, ...],\n"
                "      \"type\": <optional string: QA category>\n"
                "    },\n"
                "    ...\n"
                "  ],\n"
                "  \"metadata\": {\n"
                "    \"source\": <source string>,\n"
                "    \"timestamp\": <timestamp string>,\n"
                "    \"complexity\": <integer: reasoning complexity>\n"
                "  }\n"
                "}"
            )


        
    def process_text(
        self, text: str, source: str = "user_input"
    ) -> List[Dict[str, Any]]:
        r"""Process a single text to generate multi-hop QA pairs.

        Args:
            text (str): The input text to process.
            source (str, optional): Source identifier for the text.
                (default: :obj:`"user_input"`)

        Returns:
            List[Dict[str, Any]]: List of processed examples with QA pairs and
                metadata.
        """
        # Convert text to standard format
        raw_data = [
            {
                'text': text,
                'source': source,
            }
        ]

        # Construct examples
        constructor = ExampleConstructor(lang=self.lang, llm_serving=self.llm_serving)
        examples = constructor.construct_examples(raw_data)

        # Manage data
        # curator = DataCurator(self.config, self.rng)
        # final_dataset = curator.curate_dataset(examples)

        return examples

    def process_batch(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        r"""Process multiple texts in batch to generate multi-hop QA pairs.

        Args:
            texts (List[str]): List of input texts to process.
            sources (Optional[List[str]], optional): List of source
                identifiers. (default: :obj:`None`)

        Returns:
            List[Dict[str, Any]]: List of processed examples with QA pairs and
                metadata.

        Raises:
            ValueError: If length of sources doesn't match length of texts.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        raw_data = [
            {
                'text': text,
                'source': source,
            }
            for text, source in zip(texts, sources)
        ]

        # Construct examples
        constructor = ExampleConstructor(
            lang=self.lang, 
            llm_serving=self.llm_serving, 
            prompt_template = self.prompt_template
        )
        examples = constructor.construct_examples(raw_data)

        # # Manage data
        # curator = DataCurator(self.config, self.rng)
        # final_dataset = curator.curate_dataset(examples)

        return examples
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")
        
    def run(
            self,
            input_key:str='',
            output_key:str='',
            storage: DataFlowStorage=None,
    ):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        texts = dataframe[self.input_key].tolist()
        qa_pairs=self.process_batch(texts)
        dataframe[self.output_key] = qa_pairs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

        
class ExampleConstructor:
    r"""Constructs training examples from raw text data.

    This class handles the construction of training examples by preprocessing
    text, extracting information pairs, and generating question-answer pairs.
    """

    def __init__(
        self,
        lang: str = "en",
        llm_serving: LLMServingABC = None,
        min_text_length: int = 100,
        max_text_length: int = 200000,
        prompt_template = None
    ):
        r"""Initialize the ExampleConstructor.

        Args:
            config (ProcessorConfig): Configuration for example construction.
            multi_hop_agent (Optional[MultiHopGeneratorAgent], optional):
                Agent for generating multi-hop QA pairs. (default: :obj:`None`)
        """
        self.lang = lang
        self.llm_sering = llm_serving
        self.logger = get_logger()
        self.max_length = max_text_length
        self.min_length = min_text_length
        # self.prompt = MultiHopQAGeneratorPrompt(lang=self.lang)
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = MultiHopQAGeneratorPrompt()

    def construct_examples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        r"""Construct training examples from raw data.

        Args:
            raw_data (List[Dict[str, Any]]): List of raw data dictionaries
                containing text and metadata.

        Returns:
            List[Dict[str, Any]]: List of constructed examples with QA pairs
                and metadata.
        """
        self.logger.info("Starting to construct examples...")
        examples = []

        for data in tqdm(raw_data, desc="Constructing examples"):
            # 1. Text preprocessing
            processed_text = self._preprocess_text(data.get('text', ''))
            if not processed_text:
                example = {
                    # 'text': processed_text,
                    'qa_pairs': [],
                    'metadata': {
                        'source': data.get('source', 'unknown'),
                        'timestamp': data.get('timestamp', ''),
                        'complexity': 0,
                    },
                }
                examples.append(example)
                continue

            # 2. Generate key information pairs
            info_pairs = self._extract_info_pairs(processed_text)

            # 3. Construct question-answer pairs
            if(info_pairs):
                qa_pairs = self._generate_qa_pairs(info_pairs)
            else:
                qa_pairs = []

            # 4. Add metadata
            example = {
                # 'text': processed_text,
                'qa_pairs': qa_pairs,
                'metadata': {
                    'source': data.get('source', 'unknown'),
                    'timestamp': data.get('timestamp', ''),
                    'complexity': self._calculate_complexity(qa_pairs) if qa_pairs else 0,
                },
            }

            examples.append(example)

        self.logger.info(f"Successfully constructed {len(examples)} examples")
        return examples

    def _preprocess_text(self, text: str) -> str:
        r"""Preprocess input text for example construction.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Preprocessed text, or empty string if text fails quality
                checks.
        """
        if not isinstance(text, str):
            return ''

        # 1. Basic cleaning
        text = text.strip()

        # 2. Length check
        if (
            len(text) < self.min_length
            or len(text) > self.max_length
        ):
            self.logger.warning("text fail to pass length check.")
            return ''

        # 3. Quality check
        if not self._check_text_quality(text):
            self.logger.warning("text fail to pass quality check.")
            return ''

        return text

    def _calculate_special_char_ratio(self,text):
        # 中文字符的Unicode范围（基本汉字+扩展）
        chinese_ranges = [
            (0x4E00, 0x9FFF),    # 基本汉字
            (0x3400, 0x4DBF),    # 扩展A
            (0x20000, 0x2A6DF),  # 扩展B
            (0x2A700, 0x2B73F),  # 扩展C
            (0x2B740, 0x2B81F),  # 扩展D
            (0x2B820, 0x2CEAF)   # 扩展E
        ]
        
        special_count = 0
        for c in text:
            # 检查是否为中文、字母数字或空格
            is_chinese = any(start <= ord(c) <= end for start, end in chinese_ranges)
            if not (c.isalnum() or c.isspace() or is_chinese):
                special_count += 1
        
        return special_count / len(text) if text else 0
    
    def _check_text_quality(self, text: str) -> bool:
        r"""Check the quality of input text.

        Args:
            text (str): Text to check quality for.

        Returns:
            bool: True if text passes quality checks, False otherwise.
        """
        # 1. Basic quality check
        if (self.lang=="en" and text.count('.') < 2):  # Must have at least 2 sentences
            return False
        elif(self.lang in ["zh","ch"] and text.count("。") < 2):
            return False
        
        # 2. Special character ratio check
        special_char_ratio = self._calculate_special_char_ratio(text)
        if special_char_ratio > 0.3:  # No more than 30% special characters
            return False

        return True

    def _extract_info_pairs(self, text: str) -> List[Dict[str, Sequence[str]]]:
        r"""Extract information pairs and relationships from text.

        Args:
            text (str): Input text to extract information from.

        Returns:
            List[Dict[str, Sequence[str]]]: List of dictionaries containing
                premise, intermediate, conclusion, and related contexts.
        """
        # Split into sentences
        if(self.lang=="en"):
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        else:
            sentences = [s.strip() for s in text.split('。') if s.strip()]

        info_pairs = []

        # Extract combinations of multiple related sentences
        for i in range(len(sentences) - 2):
            if len(sentences[i]) > 10 and len(sentences[i + 1]) > 10:
                info_pairs.append(
                    {
                        'premise': sentences[i],
                        'intermediate': sentences[i + 1],
                        'conclusion': sentences[i + 2]
                        if i + 2 < len(sentences)
                        else '',
                        'related_contexts': [
                            s
                            for j, s in enumerate(sentences)
                            if j != i and j != i + 1 and len(s) > 10
                        ][:2],
                        # Limit to 2 additional related contexts
                    }
                )

        return info_pairs

    def _generate_qa_pairs(
        self, info_pairs: List[Dict[str, Sequence[str]]]
    ) -> List[Dict[str, str]]:
        r"""Generate multi-hop question-answer pairs from information pairs.

        Args:
            info_pairs (List[Dict[str, Sequence[str]]]): List of information
                pairs extracted from text.

        Returns:
            List[Dict[str, str]]: List of generated QA pairs.
        """
        user_inputs=[]
        for pair in info_pairs:
            # 1. Generate multi-hop question-answer pair using AI
            # Construct full context
            context = (
                f"{pair['premise']}. {pair['intermediate']}."
                f" {pair['conclusion']}"
            )
            user_inputs.append(self.prompt_template.build_prompt(context))

        sys_prompt=self.prompt_template.build_system_prompt()
        
        responses = self.llm_sering.generate_from_input(user_inputs=user_inputs,system_prompt=sys_prompt)
        qa_pairs=self._extract_qa_pairs(responses)

        return qa_pairs
    
    def _extract_qa_pairs(self, responses: List[str]) -> List[Dict[str, Any]]:
        """
        从原始响应中精确提取符合结构的QA对
        自动跳过非法JSON和干扰文本
        """
        qa_pairs = []
        for response in responses:
            # self.logger.info(f"generated qa: {response}")
            
            # 方法1：尝试直接解析整个响应为JSON
            try:
                qa_pair = json.loads(response)
                if isinstance(qa_pair, dict) and "question" in qa_pair:
                    qa_pairs.append(qa_pair)
                    continue
                elif isinstance(qa_pair, list):
                    for item in qa_pair:
                        if isinstance(item, dict) and "question" in item:
                            qa_pairs.append(item)
                    continue
            except json.JSONDecodeError:
                pass
            
            # 方法2：使用正则表达式查找所有JSON对象
            try:
                # 查找所有以 { 开始的JSON对象
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                
                # 更精确的模式，匹配完整的JSON对象
                brace_count = 0
                start_pos = -1
                json_objects = []
                
                for i, char in enumerate(response):
                    if char == '{':
                        if brace_count == 0:
                            start_pos = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_pos != -1:
                            json_str = response[start_pos:i+1]
                            json_objects.append(json_str)
                            start_pos = -1
                
                # 尝试解析找到的每个JSON字符串
                for json_str in json_objects:
                    try:
                        qa_pair = json.loads(json_str)
                        if (isinstance(qa_pair, dict) and \
                            "question" in qa_pair and \
                            "reasoning_steps" in qa_pair and \
                            "answer" in qa_pair and \
                            "supporting_facts" in qa_pair and \
                            "type" in qa_pair):
                            qa_pairs.append(qa_pair)
                            self.logger.info(f"Successfully extracted QA pair: {qa_pair['question']}")
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"Failed to parse JSON object: {json_str[:100]}... Error: {e}")
                        continue
                
                            # 对qa_pairs中重复的question进行去重
                if qa_pairs:
                    seen_questions = set()
                    unique_qa_pairs = []
                    
                    for qa_pair in qa_pairs:
                        question = qa_pair.get("question", "").strip().lower()
                        if question and question not in seen_questions:
                            seen_questions.add(question)
                            unique_qa_pairs.append(qa_pair)
                            self.logger.debug(f"Added unique question: {qa_pair['question']}")
                        else:
                            self.logger.debug(f"Skipped duplicate question: {qa_pair.get('question', 'N/A')}")
                    
                    qa_pairs = unique_qa_pairs
                    self.logger.info(f"After deduplication: {len(qa_pairs)} unique QA pairs")

                # 如果没有找到有效的JSON对象，记录警告
                if not json_objects:
                    self.logger.warning("No JSON objects found in model response.")
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse QA information from model response. Error: {e}")
        
        return qa_pairs
    
    def _calculate_complexity(self, qa_pairs: List[Dict[str, Any]]) -> float:
        r"""Calculate the complexity score for a set of QA pairs.

        Args:
            qa_pairs (List[Dict[str, Any]]): List of QA pairs to calculate
                complexity for.

        Returns:
            float: Complexity score between 0.0 and 1.0.
        """
        if not qa_pairs:
            return 0.0

        # Calculate complexity based on multiple factors
        complexities = []
        for qa in qa_pairs:
            # 1. Number of reasoning steps
            reasoning_steps_count = len(qa.get('reasoning_steps', []))

            # 2. Number of supporting facts
            supporting_facts_count = len(qa.get('supporting_facts', []))

            # 3. Question length
            question_length = len(qa.get('question', '').split())

            # 4. Answer length
            answer_length = len(qa.get('answer', '').split())

            # Calculate complexity of a single QA pair
            qa_complexity = (
                min(reasoning_steps_count / 3, 1.0)
                * 0.4  # Weight for reasoning steps
                + min(supporting_facts_count / 3, 1.0)
                * 0.3  # Weight for supporting facts
                + min(question_length / 20, 1.0)
                * 0.15  # Weight for question length
                + min(answer_length / 50, 1.0) * 0.15
                # Weight for answer length
            )

            complexities.append(qa_complexity)

        return sum(complexities) / len(complexities)


