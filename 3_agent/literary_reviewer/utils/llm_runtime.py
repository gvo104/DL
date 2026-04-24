import subprocess
import requests
import time

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


# -------------------------
# CHECK / START OLLAMA
# -------------------------
def ensure_ollama_running(logger=None):
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if r.status_code == 200:
            if logger:
                logger.log({"event": "ollama_status", "status": "running"})
            return True
    except:
        pass

    if logger:
        logger.log({"event": "ollama_status", "status": "not_running"})

    # попытка запуска
    subprocess.Popen(["ollama", "serve"])
    time.sleep(3)

    return True


# -------------------------
# LLM CALL + TIMING
# -------------------------
def call_llm(prompt: str, logger=None, model="qwen2.5:7b"):
    start = time.time()

    if logger:
        logger.log({"event": "llm_prompt", "prompt": prompt[:1500]})

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        r.raise_for_status()
        result = r.json()["response"]

    except Exception as e:
        result = f"LLM_ERROR: {str(e)}"

    if logger:
        logger.log({
            "event": "llm_response",
            "time_sec": time.time() - start,
            "response_preview": result[:500]
        })

    return result