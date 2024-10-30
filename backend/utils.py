import logging
import os
from typing import List, Tuple
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
import fitz 
import unicodedata
import re
from sentence_transformers import SentenceTransformer, util


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# creating handlers if they don't exist
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

# path to the trained model
T5_MODEL_DIR = './t5_qa_model'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  

# loading the tokenizer and T5 model 
try:
    logger.info(f"Loading T5 tokenizer from {T5_MODEL_DIR}")
    tokenizer = T5TokenizerFast.from_pretrained(T5_MODEL_DIR)
    logger.info(f"Loading T5 model from {T5_MODEL_DIR}")
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_model.to(device)
    logger.info(f"T5 model loaded successfully on device {device}")
except Exception as e:
    logger.error(f"Error loading T5 model: {str(e)}")
    raise e


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
          
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4]
                   
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    if text:
                        text_blocks.append(text)
        
       
        text = '\n'.join(text_blocks)
        
        if not text.strip():
            logger.error(f"No text found in PDF: {file_path}")
            raise ValueError("No text content found in PDF")
        
        # cleaning the extracted text
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
        
        text = unicodedata.normalize('NFKD', text)
        
        # Removing control characters if there are any
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
        
        
        text = ' '.join(text.split())
        
        # Removing unnecessary punctuation if present
        text = re.sub(r'([.,!?])([.,!?])+', r'\1', text)
        
        
        text = re.sub(r'[‐‑‒–—―]', '-', text)  
        text = re.sub(r'[\u201c\u201d]', '"', text)  
        
       
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        
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
        effective_k = min(top_k, available_chunks)  

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
        
        answer = re.sub(r'ID:?\s*\d+', '', answer)
        answer = re.sub(r'Objective:.*', '', answer)
        answer = re.sub(r'^\s*[-•]\s*', '', answer) 
        
        # cleaning up whitespace
        answer = ' '.join(answer.split())
        
        # Removing incomplete sentences at the end
        if '.' in answer:
            sentences = answer.split('.')
            complete_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 2]
            if complete_sentences:
                answer = '. '.join(complete_sentences) + '.'
        
      
        if answer and answer[0].isalpha():
            answer = answer[0].upper() + answer[1:]
        
       
        if not any(answer.endswith(p) for p in '.!?'):
            answer += '.'
            
       
        answer = re.sub(r'\s+([.,!?])', r'\1', answer)
        
     
        answer = re.sub(r'([A-Z][a-z]+)\s*([A-Z][a-z]+)', r'\1 \2', answer)
        
       
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
        
        # combining the relevant chunks into a single context
        combined_context = ' '.join(relevant_chunks)
        
       
        input_text = (
            f"Answer the following question based on the given context. "
            f"Provide a clear and concise answer. "
            f"Question: {question} "
            f"Context: {combined_context}"
        )
        
        # optimizing
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding='max_length'
        ).to(device)
        
        # Generating answer with refined parameters
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs,
                max_length=150,       
                min_length=20,     
                num_beams=5,
                length_penalty=1.0,    
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.6,      
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.5,
                bad_words_ids=[[tokenizer.encode(word)[0]] for word in ['objective', 'id', 'ID']]
            )
        
       
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = post_process_answer(answer)
        
    
        if len(answer.split()) < 5:
            logger.info("Answer too short, retrying generation")
            return generate_fluent_answer(question, relevant_chunks)
            
        logger.info(f"Generated answer: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "I apologize, but I couldn't generate an accurate answer to that question."
