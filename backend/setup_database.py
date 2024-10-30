import os
import sys
import time
import sqlite3
from pathlib import Path
import psutil

def kill_python_processes():
    """Kill all Python processes except the current one"""
    current_pid = os.getpid()
    killed = False
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.pid != current_pid and 'python' in proc.name().lower():
                proc.kill()
                killed = True
                print(f"Killed process: {proc.name()} (PID: {proc.pid})")
        except:
            continue
    
    if killed:
        time.sleep(2)  

def setup_database():
    db_path = Path('documents.db')
    
    kill_python_processes()
    
    if db_path.exists():
        try:
            try:
                temp_conn = sqlite3.connect(str(db_path))
                temp_conn.close()
            except:
                pass
            
            os.remove(db_path)
            print("Successfully removed old database")
        except Exception as e:
            print(f"Warning: Could not remove old database: {e}")
            backup_path = f"documents_backup_{int(time.time())}.db"
            try:
                os.rename(db_path, backup_path)
                print(f"Renamed old database to {backup_path}")
            except Exception as e:
                print(f"Error: Could not rename database: {e}")
                return False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
      
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            original_filename TEXT NOT NULL,
            content TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        
        print("Successfully created new database with correct schema!")
        
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(documents)")
        columns = {col[1] for col in cursor.fetchall()}
        conn.close()
        
        expected_columns = {'id', 'filename', 'original_filename', 'content', 'upload_date'}
        if not expected_columns.issubset(columns):
            missing = expected_columns - columns
            print(f"Error: Missing columns: {missing}")
            return False
            
        print("\nDatabase schema verified successfully!")
        print("Available columns:", columns)
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Installing required package: psutil")
        os.system(f"{sys.executable} -m pip install psutil")
        import psutil

    print("Starting database setup...")
    
    if setup_database():
        print("\nDatabase setup completed successfully!")
    else:
        print("\nDatabase setup failed!")
        sys.exit(1)