import os
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from query import answer

test_set = [
    {"question": "How many calories are in one roti?", "ground_truth": "Approximately 120 kcal"},
    {"question": "Which has more fiber, rajma or chole?", "ground_truth": "Chole has more fiber, about 10g, vs rajma's 9g"},
    {"question": "How much protein is in 100g of paneer?", "ground_truth": "18g protein"},
    {"question": "What is idli made from?", "ground_truth": "Fermented rice and urad dal batter, steamed"},
]

def run_eval():
    judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"]))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))

    rows = []
    for t in test_set:
        ans, ctx = answer(t["question"])
        rows.append({
            "question": t["question"],
            "answer": ans,
            "contexts": ctx,
            "ground_truth": t["ground_truth"],
        })

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=2, timeout=180),
    )
    print(result)
    return result

if __name__ == "__main__":
    run_eval()
