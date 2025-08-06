import re
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class AdaptiveParameterSelector:
    def __init__(self):
        self.technical_pattern = re.compile(
            r'\b\d+\.\d+\b|'                      # decimal numbers
            r'\b[A-Z]{2,}\b|'                     # acronyms
            r'\([^)]*\)|'                         # parentheses
            r'\b(?:Figure|Table|Section)\s+\d+'   # document references
        )

    def get_optimal_parameters(
        self,
        char_count: int,
        page_count: int,
        content_analysis: Dict[str, Any],
        text: str = ""
    ) -> Dict[str, Any]:

        # --- 1. Detect technical content from text if not provided ---
        has_technical_content = content_analysis.get('has_technical_content', False)
        if not has_technical_content and text:
            matches = self.technical_pattern.findall(text)
            has_technical_content = bool(matches)

        avg_paragraph_length = content_analysis.get('avg_paragraph_length', 500)
        table_density = content_analysis.get('table_density', 0.0)

        # --- 2. Base parameter selection using char_count ---
        if char_count <= 100:
            base_chunk_size = 100
            base_overlap = 0
            base_k = 1
            score_threshold = 0.6
        elif char_count <= 500:
            base_chunk_size = 200
            base_overlap = 20
            base_k = 2
            score_threshold = 0.55
        elif char_count <= 2000:
            base_chunk_size = 400
            base_overlap = 50
            base_k = 3
            score_threshold = 0.5
        elif char_count <= 10000:
            base_chunk_size = 800
            base_overlap = 100
            base_k = 4
            score_threshold = 0.45
        elif char_count <= 30000:
            base_chunk_size = 1200
            base_overlap = 200
            base_k = 6
            score_threshold = 0.4
        elif char_count <= 100000:
            base_chunk_size = 1800
            base_overlap = 300
            base_k = 8
            score_threshold = 0.35
        elif char_count <= 300000:
            base_chunk_size = 2500
            base_overlap = 400
            base_k = 10
            score_threshold = 0.3
        elif char_count <= 1000000:
            base_chunk_size = 3000
            base_overlap = 500
            base_k = 12
            score_threshold = 0.25
        else:
            base_chunk_size = 3500
            base_overlap = 600
            base_k = 15
            score_threshold = 0.2

        # --- 3. Adjustments based on page_count and content ---
        if has_technical_content and page_count > 50:
            base_chunk_size = min(base_chunk_size * 1.1, 4000)
        if table_density > 0.1:
            base_chunk_size = min(base_chunk_size * 1.2, 4000)

        # --- 4. Semantic density adjustment ---
        if text:
            word_count = max(1, len(re.findall(r'\w+', text)))
            marker_pattern = r'\b(?:therefore|however|consequently)\b'
            marker_count = len(re.findall(marker_pattern, text, flags=re.IGNORECASE))
            semantic_density = marker_count / (word_count / 1000)
            if semantic_density > 2.5:
                base_chunk_size = min(int(base_chunk_size * 1.2), 4000)

        # --- 5. Bounds for safety ---
        base_chunk_size = max(100, min(base_chunk_size, 4000))
        base_overlap = max(0, min(base_overlap, base_chunk_size // 4))
        base_k = max(1, min(base_k, 15))
        score_threshold = max(0.2, min(score_threshold, 0.6))

        # --- 6. Optional LLM generation parameters ---
        temperature = 0.3
        top_p = 0.9
        max_tokens = 512

        optimal_params = {
            'chunk_size': int(base_chunk_size),
            'chunk_overlap': int(base_overlap),
            'retriever_k': int(base_k),
            'score_threshold': score_threshold,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'estimated_chars': char_count,
            'page_count': page_count,
            'has_technical_content': has_technical_content,
            'content_characteristics': {
                'avg_paragraph_length': avg_paragraph_length,
                'table_density': table_density,
                'semantic_density': round(semantic_density, 2) if text else None # type: ignore
            }
        }

        logger.info(f"Optimal parameters determined: {optimal_params}")
        return optimal_params