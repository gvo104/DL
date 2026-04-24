import requests
from config import OLLAMA_URL, MODEL_NAME, TEMPERATURE


def llm_call(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE
                }
            },
            timeout=60
        )

        response.raise_for_status()
        return response.json()["response"]

    except Exception as e:
        return f"LLM_ERROR: {str(e)}"