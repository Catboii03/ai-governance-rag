# AI Governance RAG

A question-answering system that answers questions about AI governance laws and policies, grounded in real government documents. Ask a question in plain language and get an answer backed by the source documents it came from.

## Overview

This project is a Retrieval-Augmented Generation (RAG) system built over the AGORA dataset — a collection of 646 AI governance documents (laws, regulations, and policies) from Georgetown CSET's Emerging Technology Observatory. Instead of searching by keywords, it understands the *meaning* of a question, finds the most relevant sections of the documents, and uses a local language model to write an answer grounded in those sections.

## How It Works

The system runs as a pipeline:

1. **Chunking** — each document is split into smaller pieces, by section where possible and by fixed size otherwise.
2. **Embedding** — each chunk is converted into a vector (a numeric representation of its meaning) using a sentence-transformer model.
3. **Indexing** — all chunk vectors are stored in a FAISS index for fast similarity search.
4. **Retrieval** — a user's question is embedded the same way, and the index returns the most relevant chunks.
5. **Generation** — the retrieved chunks and the question are passed to a local LLM, which writes an answer grounded in those chunks and cites the source documents.

## Key Design Decisions

**Hybrid chunking (section-based + fixed-size).**
I split documents by section so chunks don't get cut mid-sentence and each piece stays a coherent unit. To handle the different ways documents label their sections, the splitter recognises several common markers — SECTION, SEC., Article, Chapter, and Clause — each only treated as a real section break when followed by a number, so casual mentions in the text don't trigger false splits. But since the documents come from many sources with different formats, not all have clean section markers — so I added a fixed-size fallback with overlap. The overlap prevents ideas that cross a boundary from being lost, and the fallback keeps any document from producing one huge unusable chunk. I chose which markers to support by auditing how often each appeared across the corpus, rather than guessing.

**Semantic search instead of keyword matching.**
Keyword matching only finds exact word matches, which can miss relevant documents or return unrelated ones when the wording differs. Semantic search matches *meaning*, so a question about "AI identifying people from camera footage" can still find facial-recognition law even though it shares no keywords with it.

**Local, free model instead of a paid API.**
The system uses a local model so it runs entirely on your own device, needs no internet connection, and requires no paid subscription or API key. This also keeps the project reproducible and avoids the risk of committing API keys.

**Grounding and citations.**
The model is instructed to answer *only* from the retrieved context, and to cite the source document numbers it used. If the answer isn't in the retrieved documents, it's told to say so. This keeps answers tied to real sources instead of straying into made-up information.

**4-bit quantization.**
The language model is loaded in 4-bit, compressing it to roughly 2–3 GB so it fits on GPUs with 6 GB of VRAM or less. Without this, the model is too large to fit fully on a typical GPU and spills over to the CPU, which is much slower.

## How to Run

**Requirements:** Python 3.12, an NVIDIA GPU (for 4-bit quantization), and a free Kaggle account (for the dataset, downloaded automatically on first run).

1. Install dependencies:
```
   pip install -r requirements.txt
```
   Note: PyTorch here is the CUDA 13.0 build. If needed, install it with:
```
   pip install torch --index-url https://download.pytorch.org/whl/cu130
```

2. Set your Kaggle API token (needed only on the first run to download the dataset):
```
   $env:KAGGLE_API_TOKEN = "your_token_here"
```

3. Run the app:
```
   python app.py
```
   On the first run, the system downloads the dataset, chunks it, and builds the search index (this takes a few minutes). Every run after that skips straight to the question prompt. Type a question, or type `quit` to exit.

## Example Queries

**In-corpus question:**
> Q: What rules apply to facial recognition?
> A: Returns a set of rules grounded in the relevant facial-recognition law, including notice requirements, restrictions on use, and policy requirements, citing the source document.

**Semantic match (different words, same meaning):**
> Q: Can companies be punished for misusing AI?
> A: Correctly retrieves a provision about suspending or terminating AI services for violations — even though the document never uses the words "punished" or "misusing."

**Out-of-scope question:**
> Q: What is the capital of France?
> A: States that the capital is Paris, but notes this was not found in the provided documents.

## Limitations & Future Work

- **Out-of-scope handling is imperfect.** For general-knowledge questions, the model sometimes supplies its own answer alongside the "not in the documents" note rather than fully declining. A stronger prompt or a retrieval-confidence threshold (skipping generation when the closest chunk is too far) would fix this.
- **Semantic-only retrieval.** Adding keyword (BM25) search alongside semantic search — a hybrid approach — would improve precision for exact-term queries.
- **Generation latency.** Answers take roughly 40 seconds on a 6 GB GPU. A smaller model or shorter answer length would speed this up — a speed/quality tradeoff.

## Use of AI Tools

AI assistance was used for conceptual reference, explanation, and debugging guidance. The system design, decisions, and implementation are my own.