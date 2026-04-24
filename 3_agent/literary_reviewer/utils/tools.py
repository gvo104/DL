import requests
from urllib.parse import quote

from config import TIMEOUT, WIKIPEDIA_HEADERS


def search_wikipedia(query: str) -> str:
    """
    Возвращает краткое описание (summary) из Wikipedia
    """

    try:
        # -------------------------
        # 1. SEARCH
        # -------------------------
        search_url = "https://en.wikipedia.org/w/api.php"

        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        r = requests.get(
            search_url,
            params=search_params,
            headers=WIKIPEDIA_HEADERS,
            timeout=TIMEOUT
        )
        r.raise_for_status()

        results = r.json().get("query", {}).get("search", [])
        if not results:
            return ""

        title = results[0].get("title")
        if not title:
            return ""

        # -------------------------
        # 2. SUMMARY
        # -------------------------
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"

        r2 = requests.get(
            summary_url,
            headers=WIKIPEDIA_HEADERS,
            timeout=TIMEOUT
        )
        r2.raise_for_status()

        extract = r2.json().get("extract", "")

        return extract or ""

    except Exception as e:
        print("WIKI ERROR:", e)
        return ""


def search_openalex(query: str, per_page: int = 5):
    """
    Получает список публикаций из OpenAlex
    """

    try:
        url = "https://api.openalex.org/works"

        params = {
            "search": query,
            "per-page": per_page,
            "select": "display_name,publication_year"
        }

        r = requests.get(
            url,
            params=params,
            timeout=TIMEOUT
        )
        r.raise_for_status()

        return r.json().get("results", [])

    except Exception as e:
        print("OPENALEX ERROR:", e)
        return []