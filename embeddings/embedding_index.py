from sentence_transformers import SentenceTransformer
import faiss
import pickle

class EmbeddingIndexer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []  # will store tuples (filename, chunk_text)

    def build_index(self, texts_with_metadata):
        self.texts = texts_with_metadata
        embeddings = self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, top_k)
        results = [(self.texts[i], float(distances[0][idx])) for idx, i in enumerate(indices[0])]
        return results

    # def save_index(self, index_path, texts_path):
    #     faiss.write_index(self.index, index_path)
    #     with open(texts_path, "wb") as f:
    #         pickle.dump(self.texts, f)
    #
    # def load_index(self, index_path, texts_path):
    #     self.index = faiss.read_index(index_path)
    #     with open(texts_path, "rb") as f:
    #         self.texts = pickle.load(f)

