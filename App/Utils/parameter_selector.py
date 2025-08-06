from typing import Dict,Any
import re
class AdaptiveParameterSelector:
    def __init__(self):
        self.technical_pattern = re.compile(r'\b\d+\.\d+\b|\b[A-Z]{2,}\b|\([^)]*\)|\b(?:Figure|Table|Section)\s+\d+')

    def get_optimal_parameters(self, document_text: str) -> Dict[str, Any]:
        word_count = len(document_text.split())
        page_estimate = max(1, word_count // 250)
        technical_matches = len(self.technical_pattern.findall(document_text))
        has_technical_content = technical_matches > (word_count * 0.015)

        if page_estimate <= 10:
            chunk_size, chunk_overlap, retriever_k = 1200, 200, 6
            temperature = 0.3
            max_tokens = 400
            top_p = 0.95
        elif page_estimate <= 35:
            chunk_size, chunk_overlap, retriever_k = 1800, 300, 8
            temperature = 0.2
            max_tokens = 500
            top_p = 0.9
        elif page_estimate <= 100:
            chunk_size, chunk_overlap, retriever_k = 2500, 400, 10
            temperature = 0.1
            max_tokens = 600
            top_p = 0.85
        else:
            chunk_size, chunk_overlap, retriever_k = 3000, 500, 12
            temperature = 0.05
            max_tokens = 700
            top_p = 0.8

        if has_technical_content:
            chunk_size = int(chunk_size * 1.1)

        return {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'retriever_k': retriever_k,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,  # ✅ Add this line
            'estimated_pages': page_estimate,
            'has_technical_content': has_technical_content
        }


