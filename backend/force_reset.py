import os
import sys
import time
import psutil
import sqlite3
from pathlib import Path

def kill_python_processes():
    """Kill all Python processes except the current one"""
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Skip the current process
            if proc.pid == current_pid:
                continue
                
            # Check if it's a Python process
            process_name = proc.name().lower()
            if 'python' in process_name or 'uvicorn' in process_name:
                print(f"Terminating process: {process_name} (PID: {proc.pid})")
                proc.terminate()
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def force_reset_database():
    db_path = Path('./documents.db')
    
    print("Starting database reset process...")
    
    # 1. Kill any Python processes
    print("Terminating Python processes...")
    kill_python_processes()
    
    # Wait for processes to terminate
    print("Waiting for processes to terminate...")
    time.sleep(3)
    
    # 2. Try to remove the database file
    if db_path.exists():
        try:
            # Try to close any SQLite connections
            try:
                conn = sqlite3.connect(str(db_path))
                conn.close()
            except:
                pass
            
            print("Attempting to delete database file...")
            for attempt in range(5):
                try:
                    os.remove(db_path)
                    print("Successfully deleted old database file!")
                    break
                except PermissionError:
                    if attempt < 4:
                        print(f"Attempt {attempt + 1}: File still locked, waiting...")
                        time.sleep(2)
                    else:
                        print("Could not delete database file after 5 attempts")
                        return False
                except FileNotFoundError:
                    print("Database file already deleted")
                    break
        except Exception as e:
            print(f"Error removing database: {e}")
            return False

    # 3. Create new database
    try:
        print("Creating new database...")
        # Import here to avoid import issues
        from database import Base, engine
        Base.metadata.create_all(bind=engine)
        print("Successfully created new database!")
        
        # Verify the database was created
        if db_path.exists():
            print("Database file exists and is ready to use!")
            return True
        else:
            print("Error: Database file was not created!")
            return False
            
    except Exception as e:
        print(f"Error creating new database: {e}")
        return False

if __name__ == "__main__":
    try:
        print("Starting forced database reset...")
        if force_reset_database():
            print("Database reset completed successfully!")
            sys.exit(0)
        else:
            print("Failed to reset database.")
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)