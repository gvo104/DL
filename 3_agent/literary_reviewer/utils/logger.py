import os
import json
from datetime import datetime


import os
import json
from datetime import datetime


class RunLogger:
    def __init__(self, base_dir="logs"):
        self.base_dir = base_dir
        self.trace = []

        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_path = None

    def start_run(self, topic: str, mode: str = "baseline"):
        safe_topic = topic.replace(" ", "_").lower()[:50]
        self.run_path = os.path.join(
            self.base_dir,
            f"{self.run_id}_{mode}_{safe_topic}"
        )

        os.makedirs(self.run_path, exist_ok=True)

    def log(self, event: dict):
        event = dict(event)
        event["timestamp"] = datetime.now().isoformat()
        self.trace.append(event)

    def save_prompt(self, prompt: str):
        with open(os.path.join(self.run_path, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

    def save_answer(self, answer: str):
        with open(os.path.join(self.run_path, "answer.txt"), "w", encoding="utf-8") as f:
            f.write(answer)

    def save_trace(self):
        with open(os.path.join(self.run_path, "trace.json"), "w", encoding="utf-8") as f:
            json.dump(self.trace, f, ensure_ascii=False, indent=2)

    def save_meta(self, meta: dict):
        with open(os.path.join(self.run_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
    def save_context(self, context: str):
        with open(os.path.join(self.run_path, "context.txt"), "w", encoding="utf-8") as f:
            f.write(context)