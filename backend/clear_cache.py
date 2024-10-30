import os
import shutil
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_cache():
    """Clear model cache and temporary files"""
    try:
        logger.info("Starting cache cleanup...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

        cache_paths = [
            os.path.expanduser("~/.cache/huggingface/"),
            os.path.expanduser("~/.cache/torch/"),
            "./__pycache__",
            "./vectors"
        ]

        for path in cache_paths:
            path = Path(path)
            if path.exists():
                try:
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
                    logger.info(f"Cleared cache at: {path}")
                except Exception as e:
                    logger.warning(f"Could not clear {path}: {str(e)}")

        for root, dirs, files in os.walk("."):
            for name in files:
                if name.endswith(".pyc") or name.endswith(".pyo"):
                    os.remove(os.path.join(root, name))
                    logger.info(f"Removed cache file: {name}")

        logger.info("Cache cleanup completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during cache cleanup: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Clear model and system cache')
    parser.add_argument('--force', action='store_true', help='Force clear all cache files')
    args = parser.parse_args()

    print("Starting cache cleanup...")
    if clear_cache():
        print("Cache cleared successfully!")
    else:
        print("Error clearing cache. Check the logs.")