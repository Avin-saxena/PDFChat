import os
import sqlite3
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database():
    db_path = Path('documents.db')
    
    # Try to remove existing database
    if db_path.exists():
        try:
            # Try to close any existing connections
            try:
                conn = sqlite3.connect(str(db_path))
                conn.close()
            except:
                pass
            
            os.remove(db_path)
            logger.info("Removed existing database")
            time.sleep(1)  # Wait a bit to ensure file is released
        except Exception as e:
            logger.error(f"Error removing database: {e}")
            return False

    # Create new database with correct schema
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create the documents table with all required columns
        cursor.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename VARCHAR(255) NOT NULL UNIQUE,
            original_filename VARCHAR(255) NOT NULL,
            content TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        
        # Verify the schema
        cursor.execute("PRAGMA table_info(documents)")
        columns = cursor.fetchall()
        
        logger.info("\nCreated table with columns:")
        for col in columns:
            logger.info(f"Column: {col[1]}, Type: {col[2]}, Nullable: {not col[3]}")
        
        conn.close()
        logger.info("Database created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

if __name__ == "__main__":
    try:
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        if reset_database():
            print("\nDatabase reset completed successfully!")
        else:
            print("\nDatabase reset failed!")
            exit(1)
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)