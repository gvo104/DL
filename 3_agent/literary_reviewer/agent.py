import time
from datetime import datetime
from utils.tools import search_wikipedia, search_openalex, invert_abstract
from utils.state import AgentState, log_step, save_trace
from utils.prompt_builder import build_prompt_agent
from utils.logger import RunLogger


def run_agent(topic: str, llm_call, max_steps: int = 6, per_page: int = 5) -> AgentState:
    """
    Agent pipeline:
    - Wikipedia context
    - OpenAlex retrieval + извлечение аннотаций через invert_abstract
    - Генерация финального ответа
    - Полное логирование: AgentState + RunLogger (как в baseline)
    """
    state = AgentState(topic=topic)
    logger = RunLogger()
    logger.start_run(topic, mode="agent")
    run_start = time.time()

    # 1. Wikipedia
    t0 = time.time()
    wiki = search_wikipedia(topic)
    logger.log({
        "event": "wikipedia",
        "time_sec": round(time.time() - t0, 4),
        "size": len(wiki),
        "preview": wiki[:300] if wiki else "(empty)"
    })
    log_step(state, "wikipedia", {"query": topic}, wiki[:300] if wiki else "(empty)")
    state.sources["wikipedia"] = wiki if wiki else "[Нет информации]"

    # 2. OpenAlex
    t0 = time.time()
    papers = search_openalex(topic, per_page=per_page)
    logger.log({
        "event": "openalex",
        "time_sec": round(time.time() - t0, 4),
        "n_papers": len(papers),
        "preview": [p.get("display_name", "?")[:100] for p in papers[:3]]
    })
    log_step(state, "openalex", {"query": topic, "per_page": per_page},
             f"Найдено {len(papers)} работ")
    state.sources["openalex_papers"] = [p.get("display_name", "?") for p in papers]

    # 3. Извлечение аннотаций
    abstracts_list = []
    for p in papers[:per_page]:
        inv = p.get("abstract_inverted_index")
        title = p.get("display_name", "Без названия")
        year = p.get("publication_year", "n/a")
        if inv:
            t0 = time.time()
            text = invert_abstract(inv)
            logger.log({
                "event": "invert_abstract",
                "time_sec": round(time.time() - t0, 4),
                "paper": title,
                "size": len(text)
            })
            abstracts_list.append(f"{title} ({year}): {text}")
            log_step(state, "invert_abstract", {"paper": title}, text[:300])
        else:
            abstracts_list.append(f"{title} ({year}): [аннотация отсутствует]")
            log_step(state, "invert_abstract", {"paper": title}, "no abstract")

    abstracts_text = "\n\n".join(abstracts_list)
    state.sources["abstracts_text"] = abstracts_text 

    # 4. Промпт и генерация
    prompt = build_prompt_agent(topic, wiki, abstracts_text)
    logger.log({
        "event": "prompt_ready",
        "prompt_preview": prompt[:1200]
    })
    logger.save_prompt(prompt)

    t0 = time.time()
    answer = llm_call(prompt)
    logger.log({
        "event": "llm_generation_time",
        "time_sec": round(time.time() - t0, 4),
        "answer_preview": answer[:800]
    })
    logger.save_answer(answer)

    log_step(state, "generate_answer", {"prompt_preview": prompt[:300]}, answer[:800])

    state.final_answer = answer
    state.status = "completed"
    state.stop_reason = f"Выполнены все шаги (max_steps={max_steps})"

    # Сохраняем всё
    logger.save_trace()
    logger.save_meta({
        "topic": topic,
        "mode": "agent",
        "n_papers": len(papers),
        "n_steps": state.step_id,
        "wiki_size": len(wiki),
        "abstracts_total_size": len(abstracts_text),
        "total_time_sec": round(time.time() - run_start, 4)
    })

    # Сохраняем трассу агента в ту же папку
    save_trace(state, f"{logger.run_path}/agent_trace.json")

    return state