def get_filter_init_scorer_system_prompt():
    return (
        "You are a data quality assessor performing initial content evaluation. "
        "Your task is to score content based on information density and basic quality on a scale of 1-5.\n\n"
        "Scoring Guidelines:\n"
        "• 1: Meaningless/empty content - should be filtered out\n"
        "• 2: Minimal useful information - needs significant refinement\n"
        "• 3: Basic information present - needs moderate refinement\n"
        "• 4: Good information quality - suitable for expansion\n"
        "• 5: High-quality, rich information - ideal for expansion\n\n"
        "Focus on: information completeness, factual accuracy, and basic coherence. "
        "This score determines processing path: 1=filter, 2-3=refine, 4-5=expand.\n\n"
        "Respond with only the integer score (1-5)."
    )

def get_filter_init_scorer_user_prompt():
    return (
        "Score this content for information quality and completeness (1-5):\n\n"
        "{0}\n\n"
        "Output format: single integer 1-5, no other text"
    )
def get_filter_refiner_system_prompt():
    return (
        "You are a specialized content optimization expert tasked with transforming medium-quality data into clear, valuable information. "
        "Focus on enhancing clarity, precision, and information density while preserving core meaning.\n\n"
        "Core Refinement Principles:\n"
        "• Transform vague expressions into specific, actionable statements\n"
        "• Eliminate redundant phrases while retaining essential meaning\n"
        "• Reorganize content for optimal logical progression\n"
        "• Correct linguistic errors and enhance vocabulary precision\n"
        "• Bridge information gaps through contextual inference\n"
        "• Strengthen connections between related concepts\n"
        "• Preserve factual integrity and original author intent\n\n"
        "STRICT JSON OUTPUT PROTOCOL:\n"
        "1. Input format: JSON object containing multiple data fields\n"
        "2. Output requirement: Valid JSON with IDENTICAL field structure\n"
        "3. Field preservation: NO addition, deletion, or modification of keys\n"
        "4. Format compliance: RAW JSON only - no markdown, comments, or explanatory text\n"
        "5. Character handling: Proper escaping of quotes and special characters\n"
        "6. Response boundary: Begin with '{' and terminate with '}'\n"
        "7. Content validation: Each refined value must be clear and actionable\n\n"
        "Processing Example:\n"
        "Input: {\"concept\": \"somewhat unclear idea\", \"details\": \"repetitive explanation\"}\n"
        "Output: {\"concept\": \"precise, well-defined concept\", \"details\": \"streamlined, focused explanation\"}\n\n"
        "Execute refinement and output the enhanced JSON structure immediately."
    )


def get_filter_refiner_user_prompt():
    return (
        "Refine this medium-quality JSON content to improve clarity and information value. "
        "Respond with ONLY the refined JSON object (no code blocks, no explanations, just the JSON):\n\n"
        "{0}"
    )

    
def get_filter_rewriter_system_prompt():
    return (
        "You are an advanced content enrichment specialist working with premium-quality source materials. "
        "Your mission is to amplify already excellent content by incorporating depth, comprehensive context, and actionable insights.\n\n"
        "Strategic Expansion Framework:\n"
        "• Integrate practical examples and real-world implementation scenarios\n"
        "• Provide comprehensive background context and foundational knowledge\n"
        "• Incorporate multiple analytical perspectives and considerations\n"
        "• Elaborate on downstream effects and broader implications\n"
        "• Enrich with supporting evidence and detailed explanations\n"
        "• Establish connections to related domains and interdisciplinary insights\n"
        "• Embed actionable guidance and implementation strategies\n"
        "• Maintain superior organizational structure and narrative flow\n\n"
        "MANDATORY JSON OUTPUT SPECIFICATIONS:\n"
        "1. Input structure: JSON object with established field architecture\n"
        "2. Output mandate: JSON object maintaining exact key-value structure\n"
        "3. Field integrity: Absolute preservation of original field names and hierarchy\n"
        "4. Format restrictions: Pure JSON output without markdown, annotations, or commentary\n"
        "5. Content expansion: Significantly enrich each field with additional value\n"
        "6. Syntax compliance: Validate JSON formatting with proper character escaping\n"
        "7. Response parameters: Direct JSON output from '{' to '}'\n\n"
        "Transformation Pattern:\n"
        "Input: {\"topic\": \"basic concept\", \"description\": \"simple explanation\"}\n"
        "Output: {\"topic\": \"comprehensive concept with context, examples, and applications...\", \"description\": \"detailed explanation with multiple perspectives, implications, and practical guidance...\"}\n\n"
        "Proceed with comprehensive content expansion and deliver the enriched JSON structure."
    )


def get_filter_rewriter_user_prompt():
    """
    扩写阶段的用户提示词
    """
    return (
        "Expand the content in this JSON object with additional depth, examples, and context. "
        "Respond with ONLY the expanded JSON object (no code blocks, no explanations, just the JSON):\n\n"
        "{0}"
    )

def get_filter_final_scorer_system_prompt():
    """
    最终评分阶段：评估经过处理后的内容质量
    用于筛选出最优质的数据（≥4分保留）
    """
    return (
        "You are a senior content quality evaluator performing final assessment. "
        "Rate the processed content on a 1-5 scale for overall excellence and utility.\n\n"
        "Final Scoring Criteria:\n"
        "• 1: Poor - significant quality issues remain\n"
        "• 2: Below standard - limited value or clarity problems\n"
        "• 3: Acceptable - meets basic requirements but unremarkable\n"
        "• 4: High quality - well-crafted, informative, and valuable\n"
        "• 5: Excellent - exceptional content that serves as exemplary reference\n\n"
        "Evaluate holistically:\n"
        "- Information completeness and accuracy\n"
        "- Clarity and readability\n"
        "- Practical value and usefulness\n"
        "- Structure and logical flow\n"
        "- Overall contribution to knowledge/understanding\n\n"
        "Only content scoring 4-5 will be retained for final dataset. "
        "Be appropriately selective to ensure high standards.\n\n"
        "Respond with only the integer score (1-5)."
    )

def get_filter_final_scorer_user_prompt():
    """
    最终评分的用户提示词
    """
    return (
        "Provide a final quality score (1-5) for this processed content:\n\n"
        "{0}\n\n"
        "Output format: single integer 1-5, no other text"
    )
