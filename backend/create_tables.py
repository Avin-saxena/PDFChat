import os
import sqlite3
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from models import Document
from database import Base, engine

def drop_all_tables():
    db_path = "documents.db"
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print("Removed existing database file")
    except Exception as e:
        print(f"Error removing database: {e}")

def create_tables():
    try:
        drop_all_tables()
        
        Base.metadata.create_all(bind=engine)
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if 'documents' in tables:
            columns = [col['name'] for col in inspector.get_columns('documents')]
            expected_columns = ['id', 'filename', 'original_filename', 'content', 'upload_date']
            
            missing_columns = set(expected_columns) - set(columns)
            if missing_columns:
                print(f"Warning: Missing columns: {missing_columns}")
            else:
                print("Table 'documents' created successfully with all required columns:")
                print("Columns:", columns)
        else:
            print("Error: Table 'documents' was not created")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

if __name__ == "__main__":
    if create_tables():
        print("Database initialized successfully!")
    else:
        print("Failed to initialize database")