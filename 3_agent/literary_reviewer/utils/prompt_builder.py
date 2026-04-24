from pathlib import Path


def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(topic: str, wiki: str, papers_text: str, n_sentences: int) -> str:
    system_prompt = load_txt("prompts/system_prompt.txt")
    task_prompt = load_txt("prompts/task_prompt.txt")
    few_shots = load_txt("prompts/few_shots.txt")

    task_prompt = task_prompt.format(
        topic=topic,
        wiki=wiki,
        papers_text=papers_text,
        n_sentences=n_sentences
    )

    return "\n\n".join([
    "### SYSTEM",
    system_prompt,
    
    "### TASK",
    task_prompt,

    "### FEW-SHOTS",
    "Примеры не для копирования:",
    few_shots,
    ])