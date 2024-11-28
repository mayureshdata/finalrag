import openai
from fastapi import HTTPException
from typing import Any, Dict, List
import logging
from openai import AsyncOpenAI
from langchain_community.document_loaders import PyPDFLoader,CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from app.db import get_db_service,db_status
client = AsyncOpenAI()
import tempfile
import os
from typing import Any

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# print('db_status',db_status)
# index= pc.Index(pinecone_index_name),
# embedding_model= OpenAIEmbeddings(api_key=openai_key)
# Accessing index and embedding_model
# index = db_status["index"]
# embedding_model = db_status["embedding_model"]
# print('index',index)
# print('embedding_model',embedding_model)

async def chat(query: str, index: Any, embedding_model: Any, model_name: str = "gpt-4") -> str:
    """
    Processes a user query, retrieves relevant documents from Pinecone, and generates a response.
    """
    try:
        print('index',index)
        print('embedding_model',embedding_model)
        query_embedding = embedding_model.embed_query(query)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        matched_docs = [match["metadata"] for match in results.get("matches", [])]

        if matched_docs:
            document_summary = "\n\n".join(
                [f"Document {i+1}:\n{doc['page_content']}" for i, doc in enumerate(matched_docs)]
            )
        else:
            document_summary = "No relevant documents found."

        prompt = (
            f"The user has asked: '{query}'. Below are relevant documents:\n\n"
            f"{document_summary}\n\n"
            "Please generate an accurate response to the query."
        )

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.5,
        )
        # Assuming response has a method to get the first choice
        return response.choices[0].message.content
        # return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")


# async def Data_Read_Load(file_content: bytes, filename: str, index: Any):
#     """
#     Processes the uploaded file and upserts its content into Pinecone.
#     """
#     try:
#         # Example of processing (convert file content to text embeddings)
#         # Add actual implementation details here
#         print(f"Processing file: {filename}")
#         print(f"File content length: {len(file_content)} bytes")
#         # TODO: Add embedding logic here
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

async def extract_and_process_pdf(file_path: str):
    """
    Extract, clean, and chunk text content from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of cleaned and chunked documents ready for embedding and upserting.
    """
    try:
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print("Extracted content successfully from PDF!")

        # Clean and preprocess content (e.g., remove unnecessary whitespace or special characters)
        for doc in documents:
            # Basic cleaning of content
            doc.page_content = clean_text(doc.page_content)
        
        # Split documents into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunked_documents = text_splitter.split_documents(documents)

        print(f"Chunked {len(documents)} pages into {len(chunked_documents)} chunks.")
        return chunked_documents

    except Exception as e:
        print(f"An error occurred while processing the PDF: {str(e)}")
        return None

def clean_text(text: str) -> str:
    """
    Clean the input text by removing unwanted characters, extra spaces, and other noise.

    Args:
        text (str): The raw text to clean.

    Returns:
        str: The cleaned text.
    """
    import re
    # Remove extra spaces, newlines, and non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)  # Keep basic punctuation
    return text


def pine(docs,index: Any, embedding_model: Any):
    """
    Process and upsert documents into Pinecone.

    Args:
        docs: List of document objects to be upserted.
        index: Pinecone index object.
        embedding_model: OpenAIEmbeddings instance.
    """
    try:
        # index, embedding_model = get_db_service()
        # db_service = get_db_service()  # This will raise an error if not initialized

        # index = db_service["index"]
        # embedding_model = db_service["embedding_model"]
        print('index',index)
        print('embedding_model',embedding_model)
        for doc in docs:
            print("Processing document:", doc)

            # Convert page content to embedding
            embedding = embedding_model.embed_query(doc.page_content)

            # Prepare data for upsert
            upsert_data = {
                "id": f"{doc.metadata['row']}",  # Unique ID for each entry
                "values": embedding,
                "metadata": {**doc.metadata, "page_content": doc.page_content}
            }

            # Upsert into Pinecone
            index.upsert(vectors=[upsert_data])

        return index
    except Exception as e:
        raise RuntimeError(f"Error while processing documents for Pinecone: {str(e)}")


# def PineForPdf(docsofpdf,index: Any, embedding_model: Any, model_name: str = "gpt-4"):
def PineForPdf(docsofpdf,index: Any,embedding_model: Any):
    """
    Process and upsert PDF content into Pinecone.

    Args:
        docsofpdf: List of documents extracted and chunked from PDF.
        index: Pinecone index object.
        embedding_model: OpenAIEmbeddings instance.
    """
    try:
        # index, embedding_model = get_db_service()
        # db_service = get_db_service()  # This will raise an error if not initialized

        # index = db_service["index"]
        # embedding_model = db_service["embedding_model"]
        print('index',index)
        print('embedding_model',embedding_model)
        for doc in docsofpdf:
            print("Processing PDF document:", doc)
            
            # Convert page content to embedding
            embedding = embedding_model.embed_query(doc.page_content)
            
            # Prepare data for upsert
            upsert_data = {
                "id": f"{doc.metadata.get('page', 'unknown')}-{hash(doc.page_content)}",  # unique ID for each entry
                "values": embedding,
                "metadata": {**doc.metadata, "page_content": doc.page_content}
            }
            
            # Upsert into Pinecone
            index.upsert(vectors=[upsert_data])
        
        print("All documents upserted successfully!")
    except Exception as e:
        raise RuntimeError(f"Error while processing PDFs for Pinecone: {str(e)}")



async def Data_Read_Load(file_content: bytes, filename: str, index: Any,embedding_model: Any,):
    """
    Processes the uploaded file and upserts its content into Pinecone.
    """
    try:
        # Determine the file extension based on filename for loader selection
        file_ext = filename.rsplit('.', 1)[-1].lower()

        # Use a temporary file to store file_content
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Process the file based on its type (CSV or PDF)
        if file_ext == 'csv':
            loader = CSVLoader(temp_file_path)  # Assuming CSVLoader accepts a file path
            documents = loader.load()
            index = pine(documents,index,embedding_model)  # Assuming `pine()` is the function for indexing
            print("########################")
            print(documents)
            print("########################")
            return index
        elif file_ext == 'pdf':
            # Extract and process the PDF content asynchronously
            documents = await extract_and_process_pdf(temp_file_path)
            if documents is None:
                raise HTTPException(status_code=500, detail="PDF extraction failed")
            index = PineForPdf(documents,index,embedding_model)  # Assuming `PineForPdf()` is the indexing function
            print("###########docsofpdf#############")
            print('in the Data_Read_Load\n', documents[:2])
            print("############docsofpdf############")
            return index
        else:
            raise ValueError("Unsupported file type")

    except Exception as e:
        # Handle errors during file processing
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

    finally:
        # Clean up the temporary file after loading
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

'''
def Data_Read_Load(file,filename):
    
   # Determine the file extension based on filename for loader selection
    file_ext = filename.rsplit('.', 1)[-1].lower()

    # Use a temporary file to store file_content
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(file)
        temp_file_path = temp_file.name

    try:
        # Load the file content based on type
        if file_ext == 'csv':
            loader = CSVLoader(temp_file_path)  # Assuming CSVLoader accepts a file path
            documents = loader.load()
            index = pine(documents)
            print("########################")
            print(documents)
            print("########################")
            return index
        elif file_ext == 'pdf':
            documents = extract_pdf_content_Normalpdf(temp_file_path)
            index = PineForPdf(documents)
            # index = PineForPdf(docsofpdf)
            print("###########docsofpdf#############")
            print('in the Data_Read_Load\n',documents[:2])
            print("############docsofpdf############")
            
            # return index
            # load , chunk and index the content from pdf 
            # loader =PyMuPDFLoader(temp_file_path)  # Assuming PDFLoader accepts a file path
            # docsofpdf = loader.load()
            # index = PineForPdf(docsofpdf)
            # print("###########docsofpdf#############")
            # print(docsofpdf[:2])
            # print("############docsofpdf############")
            # return index
            
        else:
            raise ValueError("Unsupported file type")

        

    finally:
        # Clean up the temporary file after loading
        os.remove(temp_file_path)
    

 
    # pass
    # age_87_documents = [doc for doc in documents if 'Age: 87' in doc.page_content]
  

'''