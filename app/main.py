from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.db import initialize_db_connection, get_db_service
from app.models import QueryRequest
from app.services import chat, Data_Read_Load

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
origins = ["http://127.0.0.1:8000", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {"message": "Hello World!"}


@app.post("/search")
async def search(query: QueryRequest, db_service=Depends(get_db_service)):
    """
    API to search data from the Pinecone index using a query.
    """
    if not db_service["initialized"]:
        raise HTTPException(status_code=500, detail="Database not initialized.")
    return {"response": await chat(query.query, db_service["index"], db_service["embedding_model"])}


@app.post("/upload")
async def upload(file: UploadFile = File(...), db_service=Depends(get_db_service)):
    """
    API to upload a file, process it, and upsert the content into Pinecone.
    """
    if not db_service["initialized"]:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    file_content = await file.read()
    await Data_Read_Load(file_content, file.filename, db_service["index"], db_service["embedding_model"])
    return {"message": "Document processed and indexed successfully."}


@app.on_event("startup")
async def startup_event():
    """
    Initialize the database connection during application startup.
    """
    await initialize_db_connection()


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup resources on shutdown.
    """
    print("Shutting down database connections...")
