from agent import run_agent
from llm import llm_call

topics = [
    "Agentic AI for customer support",
    "Graph RAG for enterprise knowledge systems",
    "LLM evaluation and process-aware metrics",
    "Retrieval-Augmented Generation",
    "Prompt Engineering techniques",
    "Federated learning for healthcare",
    "Explainable AI in finance",
    "Multimodal transformers for video understanding"
]


def test():
    for t in topics:
        print("\n========================")
        print("TOPIC:", t)

        state = run_agent(t, llm_call, max_steps=6, per_page=5)

        print("STEPS:", state.step_id)
        print("STATUS:", state.status)
        print("ANSWER (first 500 chars):")
        print(state.final_answer[:500])
        print()


if __name__ == "__main__":
    test()