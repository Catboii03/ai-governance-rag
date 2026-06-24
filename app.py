import os                                  # to check whether output files exist

from ingest import run_ingest              # one-time chunking step
from embed import run_embed                # one-time embedding step

# build chunks.json only if it's missing (first run)
if not os.path.exists("chunks.json"):
    print("First run: building chunks...")
    run_ingest()

# build the FAISS index only if it's missing (first run)
if not os.path.exists("faiss_index.bin"):
    print("First run: building embeddings index...")
    run_embed()

# import here (not at the top) so rag.py's model loading only fires
# after the data files are guaranteed to exist
from rag import run_app
run_app()