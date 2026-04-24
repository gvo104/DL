import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentState:
    topic: str
    objective: str = "Generate structured scientific overview"
    step_id: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    sources: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    final_answer: str = ""
    status: str = "in_progress"
    stop_reason: str = ""


def log_step(state: AgentState, action: str, payload: dict, result: str):
    """
    Фиксирует шаг в history агента.
    """
    state.step_id += 1
    state.history.append({
        "step_id": state.step_id,
        "action": action,
        "payload": payload,
        "result": result[:300]
    })


def save_trace(state: AgentState, path: str):
    """
    Сохраняет полную трассу агента в JSON-файл.
    """
    data = {
        "topic": state.topic,
        "objective": state.objective,
        "status": state.status,
        "stop_reason": state.stop_reason,
        "final_answer": state.final_answer,
        "sources": state.sources,
        "history": state.history
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)