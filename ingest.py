from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_index(data_dir="data/", persist_dir="./chroma_db"):
    docs = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} docs into {persist_dir}")

if __name__ == "__main__":
    build_index()
