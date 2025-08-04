import os
import torch
from parsers.clinicaltrials_parser import parse_pdf_folder_with_chunking
from embeddings.embedding_index import EmbeddingIndexer
from qa_model.local_llm import LocalLLM
import re


def cut_runoff_questions(answer:str) -> str:
    match = re.search(r"\b\w+:", answer)
    if match:
        return answer[:match.start()].strip()
    return answer.strip()


def main():
    # print("CUDA available:", torch.cuda.is_available())
    # if torch.cuda.is_available():
    #     print("Device name:", torch.cuda.get_device_name(0))
    #     print("Device count:", torch.cuda.device_count())
    input_folder = os.path.join(os.getcwd(), "data/clinical_trials_pdfs")
    index_folder = os.path.join(os.getcwd(), "data/index")
    os.makedirs(index_folder, exist_ok=True)

    print("Parsing clinical study reports...")
    texts = parse_pdf_folder_with_chunking(input_folder)

    print(f"Building embedding index on {len(texts)} chunks...")
    indexer = EmbeddingIndexer()
    indexer.build_index(texts)

    print("Loading model...")
    llm = LocalLLM()

    query = ("What are the inclusion criteria")
    print(f"Searching for relevant chunks for query: {query}")
    results = indexer.search(query, top_k=5)
    print("Generating answer...")

    combined_context = "\n\n".join([r[0] for r in results])
    full_prompt = (
        f"You are a helpful assistant. Based on the following context, answer the *single* question briefly and do not include any additional questions.\n\n"
        f"{combined_context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    # print("\n--- Results with Cosine Similarity---")
    # print(results)
    print("\n--- Context Given ---")
    print(combined_context)
    answer = llm.answer(full_prompt)
    answer = cut_runoff_questions(answer)
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
