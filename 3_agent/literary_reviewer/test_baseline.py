from baseline import run_baseline
from llm import llm_call

topics = [
    "Agentic AI for customer support",
    "Graph RAG for enterprise knowledge systems",
    "LLM evaluation and process-aware metrics"
]


def test():
    for t in topics:
        print("\n========================")
        print("TOPIC:", t)

        res = run_baseline(t, llm_call)

        print("PAPERS:", res["n_papers"])
        print("ANSWER (first 500 chars):")
        print(res["answer"][:10000])


if __name__ == "__main__":
    test()