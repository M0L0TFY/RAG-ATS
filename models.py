from pydantic import BaseModel

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

class UploadResponse(BaseModel):
    message: str