import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
_db = Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def answer(question: str, k: int = 4):
    ctx_docs = _db.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in ctx_docs)
    resp = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer using only the context above. If the context doesn't contain the answer, say so."
        }]
    )
    return resp.choices[0].message.content, [d.page_content for d in ctx_docs]

if __name__ == "__main__":
    q = "your test question here"
    ans, ctx = answer(q)
    print("ANSWER:\n", ans)
