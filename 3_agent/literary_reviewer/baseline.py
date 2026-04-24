from datetime import datetime
import time

from utils.llm_runtime import ensure_ollama_running, call_llm
from utils.tools import search_wikipedia, search_openalex
from utils.prompt_builder import build_prompt
from utils.text_stats import estimate_sentences
from utils.logger import RunLogger
from config import MAX_TOKENS


# -------------------------
# utils
# -------------------------

def new_run_id():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_papers(papers):
    """
    Форматирует список публикаций OpenAlex в текстовый блок
    """
    lines = []

    for p in papers:
        title = p.get("display_name", "Unknown")
        year = p.get("publication_year", "n/a")
        lines.append(f"- {title} ({year})")

    return "\n".join(lines)


# -------------------------
# baseline
# -------------------------

def run_baseline(topic: str, llm_call=call_llm, mode: str = "baseline"):
    """
    Baseline pipeline:
    - Wikipedia context
    - OpenAlex retrieval
    - single-pass LLM generation
    - full structured logging per run
    """

    logger = RunLogger()
    logger.start_run(topic, mode=mode)

    run_start = time.time()

    # -------------------------
    # 1. OLLAMA CHECK
    # -------------------------
    ensure_ollama_running(logger)

    # -------------------------
    # 2. WIKIPEDIA
    # -------------------------
    t0 = time.time()
    wiki = search_wikipedia(topic)

    logger.log({
        "event": "wikipedia",
        "time_sec": round(time.time() - t0, 4),
        "size": len(wiki)
    })

    # -------------------------
    # 3. OPENALEX
    # -------------------------
    t0 = time.time()
    papers = search_openalex(topic, per_page=5)

    logger.log({
        "event": "openalex",
        "time_sec": round(time.time() - t0, 4),
        "n_papers": len(papers)
    })

    # -------------------------
    # 4. FORMAT PAPERS
    # -------------------------
    papers_text = format_papers(papers)

    # -------------------------
    # 5. PROMPT BUILDING
    # -------------------------
    n_sentences = estimate_sentences(MAX_TOKENS)

    prompt = build_prompt(topic, wiki, papers_text, n_sentences)

    logger.log({
        "event": "prompt_ready",
        "prompt_preview": prompt[:1200]
    })

    logger.save_prompt(prompt)

    # -------------------------
    # 6. LLM GENERATION
    # -------------------------
    t0 = time.time()
    answer = llm_call(prompt)

    logger.log({
        "event": "llm_generation_time",
        "time_sec": round(time.time() - t0, 4),
        "answer_preview": answer[:800]
    })

    logger.save_answer(answer)

    # -------------------------
    # 7. META + TRACE SAVE
    # -------------------------
    logger.save_trace()

    logger.save_meta({
        "topic": topic,
        "mode": mode,
        "n_papers": len(papers),
        "wiki_size": len(wiki),
        "total_time_sec": round(time.time() - run_start, 4)
    })

    # -------------------------
    # RETURN
    # -------------------------
    return {
        "topic": topic,
        "answer": answer,
        "n_papers": len(papers)
    }