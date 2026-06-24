import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def run_embed():
    # load the chunks we saved in ingest.py
    with open("chunks.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    print("Loaded", len(all_chunks), "chunks")

    # load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded")

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print("Embeddings shape:", embeddings.shape)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Vectors in index:", index.ntotal)

    faiss.write_index(index, "faiss_index.bin")
    print("Index saved to faiss_index.bin")

if __name__ == "__main__":
    run_embed()