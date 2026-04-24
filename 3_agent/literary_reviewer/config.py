MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# сетевые настройки
TIMEOUT = 60

# генерация
MAX_TOKENS = 1000
TEMPERATURE = 0.2

# приложение
APP_NAME = "literary-reviewer/0.1"

# headers для внешних API
WIKIPEDIA_HEADERS = {
    "User-Agent": f"{APP_NAME}"
}