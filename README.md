# T5-Based Question Answering API

A robust and scalable solution for extracting and answering questions based on the content of uploaded PDF documents. This API leverages the power of a fine-tuned T5 model and advanced semantic search techniques to provide accurate and contextually relevant answers to user queries.

## Features

- **PDF Upload and Processing:** Easily upload PDF documents for text extraction and processing
- **Semantic Search:** Utilizes sentence embeddings for retrieving relevant document sections
- **Advanced Answer Generation:** Generates clear answers using a fine-tuned T5 model
- **Document Management:** List, view, and delete uploaded documents
- **Health Monitoring:** Built-in health check endpoint
- **Comprehensive Logging:** Detailed logging for monitoring and debugging

## Technologies Used

- Python 3.8+
- FastAPI
- Transformers (Hugging Face)
- Sentence-Transformers
- PyMuPDF (fitz)
- SQLAlchemy
- SQLite
- Uvicorn
- FAISS

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/t5-qa-api.git
cd t5-qa-api
```

### 2. Create and Activate a Virtual Environment

Using venv:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using conda:
```bash
conda create -n t5_qa_env python=3.8
conda activate t5_qa_env
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the T5 Model

Place your fine-tuned T5 model in the `backend/t5_qa_model/` directory with all necessary files:
- config.json
- pytorch_model.bin
- tokenizer_config.json
- special_tokens_map.json
- vocab.json

### 5. Create Necessary Directories

```bash
mkdir uploads
mkdir logs
```

### 6. Start the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

### 1. Upload a PDF

```http
POST /upload_pdf/
```

**Request:**
- Form Data: `file` (PDF file)

**Response:**
```json
{
  "id": 7,
  "filename": "document.pdf",
  "upload_date": "2024-10-30T09:30:00.123456"
}
```

### 2. Ask a Question

```http
POST /ask_question/
```

**Request:**
```json
{
  "document_id": 7,
  "question": "What is the name of the person?"
}
```

**Response:**
```json
{
  "answer": "The person's name is John Doe.",
  "confidence": 1.0,
  "document_id": 7,
  "question": "What is the name of the person?",
  "timestamp": "2024-10-30T09:31:48.480000",
  "model_answers": [
    {
      "model": "t5_qa_model",
      "answer": "The person's name is John Doe.",
      "confidence": 1.0
    }
  ]
}
```

### 3. List Documents

```http
GET /documents/?skip=0&limit=10
```

### 4. Delete Document

```http
DELETE /document/{document_id}
```

### 5. Health Check

```http
GET /health/
```

## Directory Structure

```
t5-qa-api/
├── README.md
├── requirements.txt
├── main.py
├── utils.py
├── .gitignore
├── backend/
│   └── t5_qa_model/
├── uploads/
├── logs/
└── venv/
```

## Error Handling

The API implements comprehensive error handling:
- 400 Bad Request: Invalid inputs
- 404 Not Found: Resource not found
- 500 Internal Server Error: Server-side errors

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeatureName`
3. Commit changes: `git commit -m "Add Your Feature Description"`
4. Push to branch: `git push origin feature/YourFeatureName`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgements

- Hugging Face for Transformers
- FastAPI
- Sentence-Transformers
- PyMuPDF
- FAISS

## Contact

For questions or issues, please open an issue on the GitHub repository.
