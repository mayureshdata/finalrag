import os
import asyncio
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Body
from dotenv import load_dotenv
load_dotenv()

db_status = {"initialized": False, "index": None, "embedding_model": None, "pc": None}

# print(os.getenv("PINECONE_API_KEY"))
async def initialize_db_connection():
    """
    Initialize the Pinecone database connection during application startup.
    """
    try:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical")

        if not pinecone_key or not openai_key:
            raise EnvironmentError("Missing API keys for Pinecone or OpenAI.")

        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_key)
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        
        # Check or create Pinecone index
        if pinecone_index_name in pc.list_indexes().names():
            print(f"Index '{pinecone_index_name}' exists.")
        else:
            pc.create_index(name=pinecone_index_name, dimension=1536, metric="cosine", spec=spec)
            while not pc.describe_index(pinecone_index_name).status["ready"]:
                await asyncio.sleep(1)
            print(f"Index '{pinecone_index_name}' created.")

        # Set up the global db status
        db_status.update({
            "initialized": True,
            "index": pc.Index(pinecone_index_name),
            "embedding_model": OpenAIEmbeddings(api_key=openai_key),
            "pc": pc,
        })
        # print("Database connection initialized:", db_status)
        print("Database connection initialized successfully.")
    except Exception as e:
        db_status["initialized"] = False
        print(f"Database initialization failed: {e}")
        raise e


def get_db_service():
    """
    Dependency to retrieve the database service for each request.
    """
    if not db_status["initialized"]:
        raise HTTPException(status_code=500, detail="Database not initialized.")
    return db_status # old working for search
    # return {
    #     "index": db_status["index"],
    #     "embedding_model": db_status["embedding_model"],
    # }
