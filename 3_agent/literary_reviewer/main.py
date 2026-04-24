from baseline import run_baseline
from llm import llm_call
from utils.io import save_result


def main():
    topic = "Graph RAG for enterprise knowledge systems"

    result = run_baseline(topic, llm_call)

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    save_result(result, "baseline_result.json")


if __name__ == "__main__":
    main()