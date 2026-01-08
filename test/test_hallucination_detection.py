"""
Tests for Hallucination Detection Operators.

Run with: pytest test/test_hallucination_detection.py -v
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch


class TestLongContextFilterOperator:
    """Tests for LongContextFilterOperator."""
    
    def test_filter_by_token_count(self):
        """Test filtering samples by token count."""
        from dataflow.operators.hallucination_detection import LongContextFilterOperator
        from dataflow.utils.storage import DataFlowStorage
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = lambda text, **kwargs: list(range(len(text.split())))
        
        # Create test data
        df = pd.DataFrame({
            "text": [
                " ".join(["word"] * 100),   # 100 tokens
                " ".join(["word"] * 500),   # 500 tokens
                " ".join(["word"] * 1000),  # 1000 tokens
            ]
        })
        
        # Create operator
        op = LongContextFilterOperator(
            tokenizer=mock_tokenizer,
            min_tokens=200,
            max_tokens=800,
            text_fields=["text"],
        )
        
        # Run
        storage = DataFlowStorage()
        storage.set("dataframe", df)
        op.run(storage, input_key="dataframe", output_key="filtered")
        
        # Check result
        result = storage.get("filtered")
        assert len(result) == 1  # Only the 500-token sample
        assert "num_tokens" in result.columns
    
    def test_multiple_text_fields(self):
        """Test filtering with multiple text fields."""
        from dataflow.operators.hallucination_detection import LongContextFilterOperator
        from dataflow.utils.storage import DataFlowStorage
        
        mock_tokenizer = Mock()
        mock_tokenizer.encode = lambda text, **kwargs: list(range(len(text.split())))
        
        df = pd.DataFrame({
            "prompt": [" ".join(["word"] * 100)],
            "answer": [" ".join(["word"] * 50)],
        })
        
        op = LongContextFilterOperator(
            tokenizer=mock_tokenizer,
            min_tokens=100,
            max_tokens=200,
            text_fields=["prompt", "answer"],
        )
        
        storage = DataFlowStorage()
        storage.set("dataframe", df)
        op.run(storage, input_key="dataframe", output_key="filtered")
        
        result = storage.get("filtered")
        assert len(result) == 1
        assert result["num_tokens"].iloc[0] == 150  # 100 + 50


class TestHallucinationInjectionOperator:
    """Tests for HallucinationInjectionOperator."""
    
    def test_injection_ratio(self):
        """Test that hallucination ratio is respected."""
        from dataflow.operators.hallucination_detection import HallucinationInjectionOperator
        from dataflow.utils.storage import DataFlowStorage
        
        # Create mock LLM serving
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="The capital is <hal>Berlin</hal>.")
        
        df = pd.DataFrame({
            "context": ["France is in Europe."] * 10,
            "answer": ["The capital is Paris."] * 10,
        })
        
        op = HallucinationInjectionOperator(
            llm_serving=mock_llm,
            hallucination_ratio=0.5,
            seed=42,
        )
        
        storage = DataFlowStorage()
        storage.set("dataframe", df)
        op.run(storage, input_key="dataframe", output_key="output")
        
        result = storage.get("output")
        n_hallucinated = result["has_hallucination"].sum()
        
        # Should be approximately 50% (±2 due to randomness)
        assert 3 <= n_hallucinated <= 7
    
    def test_parse_hal_tags(self):
        """Test parsing of <hal> tags."""
        from dataflow.operators.hallucination_detection import HallucinationInjectionOperator
        
        mock_llm = Mock()
        op = HallucinationInjectionOperator(llm_serving=mock_llm)
        
        text = "The capital is <hal>Berlin</hal>, a beautiful city."
        labels, clean = op._parse_hal_tags(text)
        
        assert clean == "The capital is Berlin, a beautiful city."
        assert len(labels) == 1
        assert labels[0]["text"] == "Berlin"
        assert labels[0]["start"] == 15
        assert labels[0]["end"] == 21


class TestSpanAnnotationOperator:
    """Tests for SpanAnnotationOperator."""
    
    def test_sentence_splitting(self):
        """Test sentence splitting."""
        from dataflow.operators.hallucination_detection import SpanAnnotationOperator
        
        with patch("dataflow.operators.hallucination_detection.generate.span_annotation.HAS_TRANSFORMERS", True):
            op = SpanAnnotationOperator.__new__(SpanAnnotationOperator)
            op.logger = Mock()
            
            text = "This is sentence one. This is sentence two! Is this three?"
            sentences = op._split_sentences(text)
            
            assert len(sentences) == 3
            assert sentences[0] == "This is sentence one."
            assert sentences[1] == "This is sentence two!"
            assert sentences[2] == "Is this three?"
    
    def test_position_finding(self):
        """Test finding sentence positions."""
        from dataflow.operators.hallucination_detection import SpanAnnotationOperator
        
        with patch("dataflow.operators.hallucination_detection.generate.span_annotation.HAS_TRANSFORMERS", True):
            op = SpanAnnotationOperator.__new__(SpanAnnotationOperator)
            op.logger = Mock()
            
            text = "The quick brown fox jumps."
            start, end = op._find_sentence_position(text, "brown fox")
            
            assert start == 10
            assert end == 19


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

