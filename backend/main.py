import os
import logging
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Any
import aiofiles

from utils import (
    extract_text_from_pdf,
    generate_fluent_answer,
    clean_text,
    chunk_context,
    generate_embeddings,
    retrieve_relevant_chunks,
    t5_model,
    tokenizer,
    device
)

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("app.log")
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# database setup
DATABASE_URL = "sqlite:///./qa_api.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    original_filename = Column(String, unique=False, index=True)
    content = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# FastAPI initialization
app = FastAPI(
    title="T5-Based Question Answering API",
    description="API for uploading PDFs and asking questions using a fine-tuned T5 model.",
    version="1.0.0"
)

# CORS configuration 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

document_embeddings = {}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from pydantic import BaseModel

class QuestionRequest(BaseModel):
    document_id: int
    question: str

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    document_id: int
    question: str
    timestamp: str
    model_answers: List[Dict[str, Any]]

class DocumentResponse(BaseModel):
    id: int
    filename: str
    upload_date: str

# Api endpoint to upload a Pdf
@app.post("/upload_pdf/", response_model=DocumentResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload and process a PDF file."""
    try:
        logger.info(f"Receiving file upload: {file.filename}")

        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Uploaded file is not a PDF: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Generating a unique filename to prevent conflicts
        unique_filename = f"{datetime.utcnow().timestamp()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)

        # Saving the uploaded file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)
        logger.info(f"File saved successfully at: {file_path}")

        # Extracting text from Pdf
        try:
            text_content = extract_text_from_pdf(file_path)
        except ValueError as ve:
            logger.error(f"PDF extraction error: {str(ve)}")
            # Remove the file if extraction fails
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Unexpected error during PDF extraction: {str(e)}")
            os.remove(file_path)
            raise HTTPException(status_code=500, detail="Error extracting text from PDF")

        # creating a new document entry
        document = Document(
            filename=unique_filename,
            original_filename=file.filename,
            content=text_content,
            upload_date=datetime.utcnow()
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"Document processed and saved with ID: {document.id}")

        # generate and store embeddings for the document
        chunks = chunk_context(text_content)
        if chunks:
            embeddings = generate_embeddings(chunks)
            document_embeddings[document.id] = {
                "chunks": chunks,
                "embeddings": embeddings
            }
            logger.info(f"Embeddings generated and stored for document ID: {document.id}")
        else:
            logger.warning(f"No chunks generated for document ID: {document.id}")

        return {
            "id": document.id,
            "filename": document.original_filename,
            "upload_date": document.upload_date.isoformat()
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error during file upload")

# Api endpoint to ask a question
@app.post("/ask_question/", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Generate an answer for a question using the fine-tuned T5 model."""
    logger.info(f"Processing question for document {request.document_id}: {request.question}")

    try:
        # Retrieving the document from the database
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            logger.warning(f"Document not found with ID: {request.document_id}")
            raise HTTPException(status_code=404, detail="Document not found")

        if request.document_id not in document_embeddings:
            logger.info(f"Generating embeddings for document ID: {request.document_id}")
            chunks = chunk_context(document.content)
            if not chunks:
                logger.error("No chunks available after chunking the document.")
                raise HTTPException(status_code=500, detail="No context available to process the question.")
            embeddings = generate_embeddings(chunks)
            document_embeddings[request.document_id] = {
                "chunks": chunks,
                "embeddings": embeddings
            }
            logger.info(f"Embeddings generated and stored for document ID: {request.document_id}")

        # retrieving the relevant chunks based on the question
        chunks = document_embeddings[request.document_id]["chunks"]
        embeddings = document_embeddings[request.document_id]["embeddings"]
        relevant_chunks = retrieve_relevant_chunks(request.question, chunks, embeddings, top_k=3)

        if not relevant_chunks:
            logger.warning("No relevant chunks retrieved for the question.")
            return {
                "answer": "I'm sorry, but I couldn't find relevant information to answer your question.",
                "confidence": 0.0,
                "document_id": request.document_id,
                "question": request.question,
                "timestamp": datetime.utcnow().isoformat(),
                "model_answers": []
            }

        # generating answer using the T5 model
        answer = generate_fluent_answer(request.question, relevant_chunks)
        confidence = 1.0  

        return {
            "answer": answer,
            "confidence": confidence,
            "document_id": request.document_id,
            "question": request.question,
            "timestamp": datetime.utcnow().isoformat(),
            "model_answers": [
                {
                    "model": "t5_qa_model",
                    "answer": answer,
                    "confidence": confidence
                }
            ]
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing question")

# Api endpoint to list documents
@app.get("/documents/", response_model=List[DocumentResponse])
async def get_documents(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Retrieve a list of all uploaded documents with pagination."""
    try:
        documents = db.query(Document).order_by(Document.upload_date.desc()).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(documents)} documents.")
        return [
            {
                "id": doc.id,
                "filename": doc.original_filename,
                "upload_date": doc.upload_date.isoformat()
            }
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching documents")

# Api endpoint to delete a document
@app.delete("/document/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)) -> Dict[str, str]:
    """Delete a document and its associated file."""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.warning(f"Attempted to delete non-existent document ID: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete the file
        file_path = os.path.join(UPLOAD_DIRECTORY, document.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file at: {file_path}")
        else:
            logger.warning(f"File not found at: {file_path}")

        # delete the embeddings if present
        if document_id in document_embeddings:
            del document_embeddings[document_id]
            logger.info(f"Deleted embeddings for document ID: {document_id}")

        # delete the database record
        db.delete(document)
        db.commit()
        logger.info(f"Deleted document ID: {document_id} from database.")

        return {"message": "Document deleted successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting document")

@app.get("/health/")
async def health_check() -> Dict[str, Any]:
    """Check API health status."""
    try:
        # checking if the model is loaded and operational
        model_status = "healthy" if t5_model else "unhealthy"

        uploads_exists = os.path.exists(UPLOAD_DIRECTORY) and os.path.isdir(UPLOAD_DIRECTORY)
        uploads_status = "healthy" if uploads_exists else "unhealthy"

        # checking database connection by querying a simple statement
        db = SessionLocal()
        try:
            db.execute("SELECT 1")
            db_status = "connected"
        except Exception:
            db_status = "disconnected"
        finally:
            db.close()

        overall_status = "healthy" if model_status == "healthy" and uploads_status == "healthy" and db_status == "connected" else "unhealthy"

        return {
            "status": overall_status,
            "database": db_status,
            "model": model_status,
            "uploads_directory": uploads_status,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
