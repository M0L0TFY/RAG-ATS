import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = "RAG-ATS"
COLLECTION_NAME = "resume_vectors"

def create_search_index(collection, index_name):
    search_index_model = SearchIndexModel(
        definition = {
        "fields": [
            {
                "numDimensions": 384,
                "path": "embedding",
                "similarity": "cosine",
                "type": "vector"
            }
        ]
    },
    name = index_name,
    type = "vectorSearch"
    )
    collection.create_search_index(model=search_index_model)

def load_resumes(directory_path):
    """
    Document(
        page_content="Page 1 contents of pdf",
        metadata={
            'source': 'folder_path/candidate1.pdf', 
            'page': 0
        }
    )
    """
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()
    
def document_to_chunks(data):
    """
    Many of Document() structure but with overlapping
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(data)

def embed_and_store(chunks):
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
    collection = db[COLLECTION_NAME]
    
    create_search_index(collection, "vector_index")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection=collection,
        index_name="vector_index"
    )

def run_ingestion_pipeline(directory_path):
    if not os.path.exists(directory_path):
        return {"message": "Folder not found."}

    #load_resumes
    raw_docs = load_resumes(directory_path)
    if not raw_docs:
        return {"message": "No PDFs found"}

    #document_to_chunks
    processed_chunks = document_to_chunks(raw_docs)

    #embed_and_store
    embed_and_store(processed_chunks)

if __name__ == "__main__":
    run_ingestion_pipeline()
    