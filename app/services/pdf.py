import os

import shutil

from fastapi import UploadFile

import pymupdf


def save_pdf(file: UploadFile, upload_dir: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    
    if not file.filename:
        raise ValueError("Uploaded file must have a filename.")

    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path


def extract_document(file_path: str) -> list[dict]:
    document = pymupdf.open(file_path)
    
    try:
        return document.to_markdown(page_chunks=True)
    
    finally:
        document.close()