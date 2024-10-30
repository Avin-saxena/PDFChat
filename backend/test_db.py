import sqlite3
from database import SessionLocal
from models import Document
import os

def test_database():
    print("Testing database connectivity...")
    

    try:
        conn = sqlite3.connect('documents.db')
        cursor = conn.cursor()
        
   
        cursor.execute("PRAGMA table_info(documents)")
        columns = cursor.fetchall()
        
        print("\nTable structure:")
        for col in columns:
            print(f"Column: {col[1]}, Type: {col[2]}, Nullable: {not col[3]}")
            
        conn.close()
        print("\nSQLite connection test: Success")
        
    except Exception as e:
        print(f"SQLite connection test failed: {e}")
        return False

    
    try:
        db = SessionLocal()
      
        test_doc = Document(
            filename="test.pdf",
            original_filename="test_original.pdf"
        )
        db.add(test_doc)
        db.commit()
        
      
        db.delete(test_doc)
        db.commit()
        
        db.close()
        print("SQLAlchemy connection test: Success")
        return True
        
    except Exception as e:
        print(f"SQLAlchemy connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_database()