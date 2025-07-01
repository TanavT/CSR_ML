import json
import os
import fitz  # PyMuPDF
from textwrap import wrap

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text_with_context(text: str, max_tokens: int = 300, overlap: int = 50):
    """Chunk text with context-aware paragraph windows."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers library is required for token-aware chunking")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # 1. Paragraph splitting
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    for i in range(len(paragraphs)):
        # Current + optional context before/after
        context_window = [
            paragraphs[i - 1] if i - 1 >= 0 else "",
            paragraphs[i],
            paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
        ]
        combined = " ".join([p for p in context_window if p])

        # Tokenize and truncate if too long
        tokens = tokenizer.tokenize(combined)
        if len(tokens) > max_tokens:
            token_ids = tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
            combined = tokenizer.decode(token_ids, skip_special_tokens=True)

        chunks.append(combined)

    return chunks

def parse_pdf_folder_with_chunking(input_folder):
    all_chunks = []
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(input_folder, fname)
            try:
                full_text = extract_text_from_pdf(path)
                chunks = chunk_text_with_context(full_text)
                for chunk in chunks:
                    all_chunks.append((fname, chunk))  # store filename with each chunk
            except Exception as e:
                print(f"Failed to parse {fname}: {e}")
    return all_chunks
