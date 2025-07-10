import json
import os
import fitz  # PyMuPDF
from textwrap import wrap
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer



# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     # print(f"text: {text}")
#     print("Done Extracting")
#     return text


def extract_text_from_pdf_and_chunk_docling(pdf_path):
    converter = DocumentConverter()
    model_id = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    max_tokens = 150
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True
    )
    document = DocumentConverter().convert(source=pdf_path).document
    chunk_iter = chunker.chunk(dl_doc=document)
    chunks = []
    for i, chunk in enumerate(chunk_iter):
        # print(f"=== {i} ===")
        # print(f"chunk.text:\n{f'{chunk.text[:300]}…'!r}")
        enriched_text = format_chunk_with_section(chunker.contextualize(chunk=chunk))
        # print(f"chunker.contextualize(chunk):\n{f'{enriched_text[:300]}…'!r}")
        chunks.append(enriched_text)
    return chunks

def format_chunk_with_section(chunk: str) -> str:
    lines = chunk.strip().split('\n', 1)
    if len(lines) == 2:
        section_title = lines[0].strip()
        content = lines[1].strip()
        return f"[Title of section: {section_title}]\n{content}"
    else:
        # just in case, shouldn't trigger
        return f"[Title of section: {lines[0].strip()}]"

# def chunk_text_with_context(text: str, max_tokens: int = 300):
#     """deprecated: Chunk text with context-aware paragraph windows."""
#     try:
#         from transformers import AutoTokenizer
#     except ImportError:
#         raise RuntimeError("transformers library is required for token-aware chunking")
#
#     tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#
#     paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
#     chunks = []
#
#     for i in range(len(paragraphs)):
#         context_window = [
#             paragraphs[i - 1] if i - 1 >= 0 else "",
#             paragraphs[i],
#             paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
#         ]
#         combined = " ".join([p for p in context_window if p])
#
#         tokens = tokenizer.tokenize(combined)
#         if len(tokens) > max_tokens:
#             token_ids = tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
#             combined = tokenizer.decode(token_ids, skip_special_tokens=True)
#
#         chunks.append(combined)
#     print(chunks)
#     return chunks


def parse_pdf_folder_with_chunking(input_folder):
    all_chunks = []
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(input_folder, fname)
            try:
                chunks = extract_text_from_pdf_and_chunk_docling(path)
                for chunk in chunks:
                    all_chunks.append(chunk)  # store filename with each chunk
            except Exception as e:
                print(f"Failed to parse {fname}: {e}")
    return all_chunks
