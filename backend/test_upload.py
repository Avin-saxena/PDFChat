import requests
import os

def test_upload():
 
    with open("test.pdf", "w") as f:
        f.write("Test PDF content")

    try:
      
        health_response = requests.get("http://localhost:8000/health")
        print("Health check response:", health_response.json())

       
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
        if os.path.exists("test.pdf"):
            os.remove("test.pdf")

if __name__ == "__main__":
    test_upload()