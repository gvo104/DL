"""
Main experiment runner.
Запускает baseline, agent, agent+evaluator на всех темах,
собирает метрики и логи.
"""
import time
import pandas as pd
from datetime import datetime

from baseline import run_baseline
from agent import run_agent
from llm import llm_call
from utils.evaluator import evaluate_answer

# 8 тем
TOPICS = [
    "Agentic AI for customer support",
    "Graph RAG for enterprise knowledge systems",
    "LLM evaluation and process-aware metrics",
    "Retrieval-Augmented Generation",
    "Prompt Engineering techniques",
    "Federated learning for healthcare",
    "Explainable AI in finance",
    "Multimodal transformers for video understanding"
]


def run_experiment(mode: str, topic: str, **kwargs) -> dict:
    """
    Запускает один прогон и возвращает словарь с метриками.
    mode: "baseline" | "agent" | "agent_evaluator"
    """
    t0 = time.time()

    if mode == "baseline":
        result = run_baseline(topic, llm_call)
        answer = result["answer"]
        n_papers = result.get("n_papers", 0)
        n_steps = 1
        # контекст: Wikipedia + CrossRef + заголовки OpenAlex
        wiki = result.get("wiki", "")
        papers_text = result.get("papers_text", "")
        context = wiki + "\n\n" + papers_text if (wiki or papers_text) else ""

    elif mode in ("agent", "agent_evaluator"):
        per_page = kwargs.get("per_page", 5)
        max_steps = kwargs.get("max_steps", 6)
        state = run_agent(topic, llm_call, max_steps=max_steps, per_page=per_page)
        answer = state.final_answer
        n_papers = len(state.sources.get("openalex_papers", []))
        n_steps = state.step_id
        # контекст: Wikipedia + аннотации OpenAlex
        wiki = state.sources.get("wikipedia", "")
        abstracts_text = state.sources.get("abstracts_text", "")
        context = wiki + "\n\n" + abstracts_text if (wiki or abstracts_text) else ""
        # трасса уже сохранена внутри run_agent, повторно не сохраняем

    else:
        raise ValueError(f"Unknown mode: {mode}")

    latency = round(time.time() - t0, 2)

    # Оценка evaluator'ом (для всех режимов, где есть контекст)
    scores = {}
    if answer and context:
        scores = evaluate_answer(answer, context)
    else:
        scores = {
            "correctness": 0,
            "groundedness": 0,
            "completeness": 0,
            "coverage_of_required_fields": 0,
            "source_consistency": 0,
            "rubric": 0,
            "comment": ""
        }

    return {
        "topic": topic,
        "mode": mode,
        "per_page": kwargs.get("per_page", 5),
        "max_steps": kwargs.get("max_steps", 6),
        "n_steps": n_steps,
        "n_papers": n_papers,
        "latency": latency,
        **{k: scores.get(k, 0) for k in ["correctness", "groundedness", "completeness",
                                           "coverage_of_required_fields", "source_consistency", "rubric"]},
        "comment": scores.get("comment", "")
    }


def main():
    all_results = []

    # Эксперименты 1–3: baseline, agent, agent+evaluator на всех 8 темах
    for topic in TOPICS:
        print(f"\n{'='*50}")
        print(f"TOPIC: {topic}")

        for mode in ["baseline", "agent", "agent_evaluator"]:
            print(f"  Mode: {mode}...")
            row = run_experiment(mode, topic, per_page=5, max_steps=6)
            all_results.append(row)
            print(f"    rubric={row['rubric']}, latency={row['latency']}s, steps={row['n_steps']}")

    # Эксперимент 4: сравнение top-3, top-5, top-8 на подмножестве тем (первые 4)
    for per_page in [3, 5, 8]:
        for topic in TOPICS[:4]:
            print(f"\n  Agent per_page={per_page}, topic={topic}...")
            row = run_experiment("agent", topic, per_page=per_page, max_steps=6)
            all_results.append(row)

    # Эксперимент 5: ограничение max_steps (4, 6, 8) на подмножестве тем
    for max_steps in [4, 6, 8]:
        for topic in TOPICS[:4]:
            print(f"\n  Agent max_steps={max_steps}, topic={topic}...")
            row = run_experiment("agent", topic, per_page=5, max_steps=max_steps)
            all_results.append(row)

    # Сохраняем результаты
    df = pd.DataFrame(all_results)
    df.to_csv("experiment_results.csv", index=False)
    print("\nResults saved to experiment_results.csv")
    print("\nMean metrics by mode:")
    print(df.groupby("mode")[["rubric", "latency", "n_steps"]].mean())


if __name__ == "__main__":
    main()