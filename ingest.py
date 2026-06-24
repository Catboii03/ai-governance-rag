import os
import json
import kagglehub
from chunking import hybrid_chunk

def run_ingest():
    # locate the data
    dataset_path = kagglehub.dataset_download("umerhaddii/ai-governance-documents-data")
    agora_path = os.path.join(dataset_path, "agora")
    fulltext_path = os.path.join(agora_path, "fulltext")
    fulltext_files = os.listdir(fulltext_path)

    # chunk every document and tag each chunk with its source
    all_chunks = []
    for filename in fulltext_files:
        agora_id = os.path.splitext(filename)[0]
        file_path = os.path.join(fulltext_path, filename)
        text = open(file_path, encoding="utf-8").read()
        chunks = hybrid_chunk(text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "agora_id": agora_id})

    print("Total chunks created:", len(all_chunks))

    # save to disk for the next stage
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    print("Saved", len(all_chunks), "chunks to chunks.json")

if __name__ == "__main__":
    run_ingest()