import requests
import os

def test_upload():
    # Create a test PDF file
    with open("test.pdf", "w") as f:
        f.write("Test PDF content")

    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health")
        print("Health check response:", health_response.json())

        # Test file upload
        files = {
            'file': ('test.pdf', open('test.pdf', 'rb'), 'application/pdf')
        }
        
        response = requests.post(
            'http://localhost:8000/upload_pdf/',
            files=files
        )
        
        print("\nUpload response:", response.json() if response.ok else response.text)
        print("Status code:", response.status_code)
        
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Clean up test file
        if os.path.exists("test.pdf"):
            os.remove("test.pdf")

if __name__ == "__main__":
    test_upload()