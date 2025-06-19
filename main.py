import os
from parsers.clinicaltrials_parser import parse_clinical_trials_json_folder
from embeddings.embedding_index import EmbeddingIndexer
from qa_model.local_llm import LocalLLM

def main():
    input_folder = os.path.join(os.getcwd(), "data/clinical_trials_json")
    index_folder = os.path.join(os.getcwd(), "data/index")
    os.makedirs(index_folder, exist_ok=True)

    print("Parsing clinical trial JSON files...")
    documents = parse_clinical_trials_json_folder(input_folder)
    filenames, texts = zip(*documents)

    print(f"Building embedding index on {len(texts)} documents...")
    indexer = EmbeddingIndexer()
    indexer.build_index(texts)
    indexer.save_index(os.path.join(index_folder, "faiss.index"), os.path.join(index_folder, "texts.pkl"))

    print("Loading model...")
    llm = LocalLLM()

    query = "What is the objective of the study?"
    print(f"Searching for relevant documents for query: {query}")
    results = indexer.search(query, top_k=3)
    print("Generating answer...")

    combined_context = "\n\n".join([r[0] for r in results])
    # print(combined_context)
    full_prompt = (
        "Use the following context to answer the question.\n\n"
        f"{combined_context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    answer = llm.answer(full_prompt)

    print("\n--- Answer ---")
    print(answer)

if __name__ == "__main__":
    main()
