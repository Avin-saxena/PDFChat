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
            if proc.pid == current_pid:
                continue
                
            process_name = proc.name().lower()
            if 'python' in process_name or 'uvicorn' in process_name:
                print(f"Terminating process: {process_name} (PID: {proc.pid})")
                proc.terminate()
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def force_reset_database():
    db_path = Path('./documents.db')
    
    print("Starting database reset process...")
    
    print("Terminating Python processes...")
    kill_python_processes()
    
    print("Waiting for processes to terminate...")
    time.sleep(3)
    
    if db_path.exists():
        try:
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
    try:
        print("Creating new database...")
        from database import Base, engine
        Base.metadata.create_all(bind=engine)
        print("Successfully created new database!")
        
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