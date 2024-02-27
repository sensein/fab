import os

class VeryBaseService:
    def __init__(self):
        pass

    def get_cache_directory(self, cache_dir_path = "__cache__"):
        # Check if the cache directory already exists
        if not os.path.exists(cache_dir_path):
            # If not, create the directory (including any necessary parent directories)
            os.makedirs(cache_dir_path, exist_ok=True)
        return os.path.abspath(cache_dir_path)
