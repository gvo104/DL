import json
        
def estimate_sentences(max_tokens: int, n_sections: int = 5) -> int:
    """
    Преобразует max_tokens → примерное число предложений на секцию
    """
    tokens_per_section = max_tokens // n_sections

    # 1 токен ~ 0.75 слова
    words = tokens_per_section * 0.75

    # 1 предложение ~ 15 слов
    sentences = int(words // 15)

    return max(1, sentences)