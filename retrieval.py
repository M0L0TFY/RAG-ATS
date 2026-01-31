import os
from models import QueryResponse
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
collection = client["RAG-ATS"]["resume_vectors"]

llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="vector_index"
)

def query_resumes(query: str) -> QueryResponse:
    docs = vector_store.similarity_search(query, k=3)
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"Context: {context}\nQuery: {query}"
    response = llm.invoke(prompt)
    return QueryResponse(
        query=query,
        answer=response,
        sources=[os.path.basename(s) for s in sources]
    )