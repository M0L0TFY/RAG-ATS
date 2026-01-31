import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from ingestion import run_ingestion_pipeline
from retrieval import query_resumes
from models import QueryResponse, UploadResponse

app = FastAPI()

@app.post("/upload", response_model=UploadResponse)
async def upload_resumes(files: list[UploadFile] = File(...)):
    temp_dir = f"temp_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    try:
        # load files to temp file
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        run_ingestion_pipeline(temp_dir)
        return UploadResponse(message="Successfully processed resumes.")
    except Exception as e: return UploadResponse(message=str(e))
    finally: shutil.rmtree(temp_dir) # clean up temp dir

@app.get("/query", response_model=QueryResponse)
async def search(query: str):
    return query_resumes(query)