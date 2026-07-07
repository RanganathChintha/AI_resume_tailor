import os
import json
import hashlib
import glob
from PyPDF2 import PdfReader
from app.core.config import settings

class ResumeCacheManager:
    def _get_directory_hash(self):
        """Generates a hash based on the file names and modification times."""
        hasher = hashlib.md5()
        pdf_files = sorted(glob.glob(os.path.join(settings.DATA_DIR, "*.pdf")))
        
        for file_path in pdf_files:
            hasher.update(file_path.encode('utf-8'))
            mtime = os.path.getmtime(file_path)
            hasher.update(str(mtime).encode('utf-8'))
            
        return hasher.hexdigest()

    def _extract_text_from_pdfs(self):
        """Reads all PDFs in the data folder and extracts text."""
        all_text = ""
        pdf_files = glob.glob(os.path.join(settings.DATA_DIR, "*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in the '{settings.DATA_DIR}' directory.")
            
        for file_path in pdf_files:
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    all_text += extracted + "\n"
        return all_text

    def get_resume_data(self):
        """Returns resume data from cache if unchanged, otherwise parses PDFs."""
        current_hash = self._get_directory_hash()
        
        if os.path.exists(settings.CACHE_FILE):
            with open(settings.CACHE_FILE, 'r', encoding='utf-8') as f:
                try:
                    cache_data = json.load(f)
                    if cache_data.get("hash") == current_hash:
                        print("[INFO] No changes in data folder. Loading resume from cache...")
                        return cache_data.get("text")
                except json.JSONDecodeError:
                    pass

        print("[INFO] Changes detected or no cache found. Parsing PDFs...")
        text_data = self._extract_text_from_pdfs()
        
        with open(settings.CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({"hash": current_hash, "text": text_data}, f, indent=4)
            
        return text_data