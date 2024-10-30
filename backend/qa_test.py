from utils import generate_answer, extract_text_from_pdf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_with_sample_pdf(pdf_path: str):
    """Test QA system with a sample PDF"""
    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"\nExtracted {len(text)} characters from PDF")
        
        test_questions = [
            "What is the email address?",
            "What is the phone number?",
            "What is the person's name?",
            "What are the technical skills?",
            "What is the education background?"
        ]
        
        print("\nTesting Questions:")
        for question in test_questions:
            result = generate_answer(text, question)
            print(f"\nQ: {question}")
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_pdf_path = "path/to/your/test.pdf"
    test_with_sample_pdf(test_pdf_path)