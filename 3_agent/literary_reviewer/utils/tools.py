import requests
import time
import re
from urllib.parse import quote

from config import TIMEOUT, WIKIPEDIA_HEADERS, APP_NAME


def search_wikipedia(query: str) -> str:
    """
    Поиск по Wikipedia с несколькими стратегиями + проверка релевантности.
    """
    strategies = [
        {"srsearch": query, "srwhat": "title"},
        {"srsearch": query, "srwhat": "text"},
    ]

    keywords = [w for w in query.split() if len(w) >= 5]
    for kw in keywords[:5]:
        strategies.append({"srsearch": kw, "srwhat": "text"})

    try:
        search_url = "https://en.wikipedia.org/w/api.php"

        for params in strategies[:7]:
            search_params = {
                "action": "query",
                "list": "search",
                "format": "json",
                **params
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
                continue

            # Проверяем несколько результатов, а не только первый
            for result in results[:3]:
                title = result.get("title", "")
                if not title:
                    continue

                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
                r2 = requests.get(
                    summary_url,
                    headers=WIKIPEDIA_HEADERS,
                    timeout=TIMEOUT
                )
                r2.raise_for_status()

                extract = r2.json().get("extract", "")
                if not extract:
                    continue

                # ПРОВЕРКА РЕЛЕВАНТНОСТИ
                query_words = [w.lower() for w in query.split() if len(w) >= 4]
                text_to_check = (title + " " + extract[:500]).lower()
                
                matches = sum(1 for w in query_words if w in text_to_check)
                
                # Повышенный порог: нужно ≥2 совпадений ИЛИ ≥60% слов
                if len(query_words) >= 2:
                    min_matches = max(2, int(len(query_words) * 0.6))
                else:
                    min_matches = 1
                
                # Дополнительно: проверяем, что это не явно чужая тема
                irrelevant_domains = [
                    "alternative education", "narrative evaluation",
                    "vibe coding", "music", "film", "sports"
                ]
                is_irrelevant = any(domain in text_to_check for domain in irrelevant_domains)
                
                if matches >= min_matches and not is_irrelevant:
                    return extract

        return ""

    except Exception as e:
        print(f"WIKI ERROR ({query}): {e}")
        return ""


def search_crossref(query: str) -> str:
    """
    Поиск научных статей через CrossRef API (публичный, без ключа).
    Возвращает заголовки и abstracts релевантных статей.
    """
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": 3
        }
        headers = {
            "User-Agent": f"{APP_NAME} (mailto:dev@{APP_NAME})"
        }

        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=TIMEOUT
        )
        r.raise_for_status()

        items = r.json().get("message", {}).get("items", [])
        if not items:
            return ""

        context_parts = []
        for item in items:
            title = item.get("title", [""])[0] if item.get("title") else "Без названия"
            abstract = item.get("abstract", "")
            if abstract:
                # Очистка: удаляем ВСЁ, что похоже на HTML/XML
                # 1. JATS-теги (научная разметка)
                abstract = re.sub(r'</?jats:[^>]+>', ' ', abstract)
                abstract = re.sub(r'</?jats[^>]*>', ' ', abstract)
                
                # 2. Двойное экранирование
                abstract = re.sub(r'&lt;/?\w+&gt;', ' ', abstract)
                
                # 3. Обычные теги
                abstract = re.sub(r'</?\w+>', ' ', abstract)
                abstract = re.sub(r'<\s*>', ' ', abstract)
                
                # 4. Декодируем HTML entities
                abstract = abstract.replace('&amp;nbsp;', ' ')
                abstract = abstract.replace('&amp;lt;', '')
                abstract = abstract.replace('&amp;gt;', '')
                abstract = abstract.replace('&amp;amp;', '&')
                abstract = abstract.replace('&amp;quot;', '"')
                abstract = abstract.replace('&nbsp;', ' ')
                abstract = abstract.replace('&lt;', '')
                abstract = abstract.replace('&gt;', '')
                abstract = abstract.replace('&amp;', '&')
                abstract = abstract.replace('&quot;', '"')
                
                # 5. Чистим пробелы
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                
                # Чистим заголовок тоже
                title = re.sub(r'</?jats:[^>]+>', '', title)
                title = re.sub(r'&lt;/?\w+&gt;', '', title)
                title = re.sub(r'</?\w+>', '', title)
                title = title.replace('&amp;nbsp;', ' ')
                title = re.sub(r'\s+', ' ', title).strip()
                
                context_parts.append(f"{title}: {abstract}")

        return "\n\n".join(context_parts) if context_parts else ""

    except Exception as e:
        print(f"CROSSREF ERROR: {e}")
        return ""


def get_enriched_context(query: str) -> str:
    """
    Собирает контекст из Wikipedia + CrossRef.
    Возвращает комбинированный текст или сообщение об отсутствии данных.
    """
    contexts = []

    # Wikipedia
    wiki = search_wikipedia(query)
    if wiki:
        contexts.append(("Wikipedia", wiki))

    # CrossRef
    cr = search_crossref(query)
    if cr:
        contexts.append(("CrossRef", cr))

    if not contexts:
        return "[Нет релевантной информации в доступных источниках]"

    parts = []
    for source, text in contexts:
        parts.append(f"=== {source} ===\n{text}")

    return "\n\n".join(parts)


def search_openalex(query: str, per_page: int = 5):
    """
    Получает список публикаций из OpenAlex.
    """
    try:
        url = "https://api.openalex.org/works"

        params = {
            "search": query,
            "per-page": per_page,
            "select": "display_name,publication_year"
        }

        headers = {
            "User-Agent": f"{APP_NAME}"
        }

        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=TIMEOUT
        )
        r.raise_for_status()

        return r.json().get("results", [])

    except Exception as e:
        print(f"OPENALEX ERROR ({query}): {e}")
        return []