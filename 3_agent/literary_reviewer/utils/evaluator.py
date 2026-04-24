import json
from pathlib import Path
from llm import llm_call


def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def evaluate_answer(answer: str, context: str) -> dict:
    """
    Оценивает качество ответа по 5 критериям.
    Возвращает словарь с баллами 0-5 и комментарием.
    """
    prompt_template = load_txt("prompts/evaluator_prompt.txt")
    prompt = prompt_template.format(context=context, answer=answer)

    raw = llm_call(prompt)

    # Извлекаем JSON из ответа модели
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        json_str = raw[start:end]
        result = json.loads(json_str)
        # Валидация — убедимся, что все ключи на месте
        for key in ["correctness", "groundedness", "completeness", 
                     "coverage_of_required_fields", "source_consistency"]:
            if key not in result:
                result[key] = 0
        if "comment" not in result:
            result["comment"] = ""
        # Приводим к целым
        for key in ["correctness", "groundedness", "completeness",
                     "coverage_of_required_fields", "source_consistency"]:
            result[key] = int(result[key])
    except Exception as e:
        print(f"EVALUATOR PARSE ERROR: {e}")
        result = {
            "correctness": 0,
            "groundedness": 0,
            "completeness": 0,
            "coverage_of_required_fields": 0,
            "source_consistency": 0,
            "comment": f"Ошибка парсинга оценки. Raw: {raw[:200]}"
        }

    # Добавляем интегральную оценку
    result["rubric"] = round(
        (result["correctness"] + result["groundedness"] + result["completeness"] +
         result["coverage_of_required_fields"] + result["source_consistency"]) / 5, 2
    )

    return result