# utils.py

import logging
import os
from typing import List, Tuple
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
import fitz  # PyMuPDF
import unicodedata
import re
from sentence_transformers import SentenceTransformer, util

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers if they don't exist
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("utils.log")
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Paths to the trained models
T5_MODEL_DIR = './t5_qa_model'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Efficient and effective for semantic similarity

# Load the tokenizer and T5 model
try:
    logger.info(f"Loading T5 tokenizer from {T5_MODEL_DIR}")
    tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_DIR)
    logger.info(f"Loading T5 model from {T5_MODEL_DIR}")
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_DIR)
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_model.to(device)
    logger.info(f"T5 model loaded successfully on device {device}")
except Exception as e:
    logger.error(f"Error loading T5 model: {str(e)}")
    raise e

# Load the sentence transformer model for embeddings
try:
    logger.info(f"Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_model.to(device)
    logger.info("SentenceTransformer model loaded successfully")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    raise e

def extract_text_from_pdf(file_path: str) -> str:
    """
    Enhanced PDF text extraction with better formatting preservation.
    """
    try:
        with fitz.open(file_path) as doc:
            text_blocks = []
            for page_num, page in enumerate(doc, start=1):
                # Extract text with blocks to preserve structure
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4]
                    # Clean up the text block
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    if text:
                        text_blocks.append(text)
        
        # Join blocks with proper spacing
        text = '\n'.join(text_blocks)
        
        if not text.strip():
            logger.error(f"No text found in PDF: {file_path}")
            raise ValueError("No text content found in PDF")
        
        # Clean the extracted text
        text = clean_text(text)
        
        logger.info(f"Successfully extracted text from PDF: {file_path}")
        return text
            
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{file_path}': {str(e)}")
        raise e

def clean_text(text: str) -> str:
    """
    Enhanced text cleaning and normalization.
    """
    try:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
        
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        
        # Remove unnecessary punctuation
        text = re.sub(r'([.,!?])([.,!?])+', r'\1', text)
        
        # Fix common OCR issues
        text = re.sub(r'[‐‑‒–—―]', '-', text)  # Normalize different types of hyphens
        text = re.sub(r'[\u201c\u201d]', '"', text)  # Normalize quotes
        
        # Remove repeated punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        
        cleaned = text.strip()
        logger.debug(f"Cleaned text length: {len(cleaned)}")
        return cleaned
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text

def chunk_context(context: str, max_chunk_size: int = 512) -> List[str]:
    """
    Split context into smaller, meaningful chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', context)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_embeddings(chunks: List[str]) -> torch.Tensor:
    """
    Generate sentence embeddings for each chunk.
    """
    try:
        logger.info("Generating embeddings for context chunks")
        embeddings = embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
        logger.info("Embeddings generated successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise e

def retrieve_relevant_chunks(question: str, chunks: List[str], embeddings: torch.Tensor, top_k: int = 3) -> List[str]:
    """
    Retrieve the top_k most relevant context chunks based on semantic similarity.
    """
    try:
        logger.info(f"Retrieving top {top_k} relevant chunks for the question")
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
        available_chunks = len(chunks)
        effective_k = min(top_k, available_chunks)  # Ensure k does not exceed available chunks

        if effective_k == 0:
            logger.warning("No chunks available to retrieve.")
            return []
        
        top_results = torch.topk(cos_scores, k=effective_k)
        
        relevant_chunks = [chunks[idx] for idx in top_results.indices]
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {str(e)}")
        raise e

def post_process_answer(answer: str) -> str:
    """
    Enhanced post-processing for cleaner, more natural answers.
    """
    try:
        # Remove common unwanted patterns
        answer = re.sub(r'ID:?\s*\d+', '', answer)
        answer = re.sub(r'Objective:.*', '', answer)
        answer = re.sub(r'^\s*[-•]\s*', '', answer)  # Remove leading bullets
        
        # Clean up whitespace
        answer = ' '.join(answer.split())
        
        # Remove incomplete sentences at the end
        if '.' in answer:
            sentences = answer.split('.')
            complete_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 2]
            if complete_sentences:
                answer = '. '.join(complete_sentences) + '.'
        
        # Ensure proper capitalization
        if answer and answer[0].isalpha():
            answer = answer[0].upper() + answer[1:]
        
        # Add proper ending punctuation
        if not any(answer.endswith(p) for p in '.!?'):
            answer += '.'
            
        # Clean up any remaining artifacts
        answer = re.sub(r'\s+([.,!?])', r'\1', answer)
        
        # Format names more cleanly
        answer = re.sub(r'([A-Z][a-z]+)\s*([A-Z][a-z]+)', r'\1 \2', answer)
        
        # Remove any remaining unwanted patterns
        answer = re.sub(r'\b(ID|Submitted by|Name|Roll|Number):\s*', '', answer, flags=re.IGNORECASE)
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"Error in post-processing answer: {str(e)}")
        return answer

def generate_fluent_answer(question: str, relevant_chunks: List[str]) -> str:
    """
    Generate a fluent, well-formed answer to the given question based on the most relevant context chunks.
    """
    try:
        logger.info(f"Generating answer for question: {question}")
        
        if not relevant_chunks:
            logger.warning("No relevant chunks found for the question.")
            return "I'm sorry, but I couldn't find relevant information to answer your question."
        
        # Combine the relevant chunks into a single context
        combined_context = ' '.join(relevant_chunks)
        
        # Format input with explicit instruction
        input_text = (
            f"Answer the following question based on the given context. "
            f"Provide a clear and concise answer. "
            f"Question: {question} "
            f"Context: {combined_context}"
        )
        
        # Tokenize with optimized settings
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding='max_length'
        ).to(device)
        
        # Generate answer with refined parameters
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs,
                max_length=150,        # Shorter but more focused
                min_length=20,         # Ensure minimum length
                num_beams=5,
                length_penalty=1.0,    # Balanced length penalty
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.6,       # Reduced for more focused answers
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.5,
                # Prevent unwanted tokens
                bad_words_ids=[[tokenizer.encode(word)[0]] for word in ['objective', 'id', 'ID']]
            )
        
        # Decode and post-process
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = post_process_answer(answer)
        
        # Retry if answer is too short
        if len(answer.split()) < 5:
            logger.info("Answer too short, retrying generation")
            return generate_fluent_answer(question, relevant_chunks)
            
        logger.info(f"Generated answer: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "I apologize, but I couldn't generate an accurate answer to that question."
