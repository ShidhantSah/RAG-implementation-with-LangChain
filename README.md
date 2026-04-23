# RAG Implementation with LangChain 

A working implementation of Retrieval-Augmented Generation (RAG) using LangChain, Llama (via Groq), and ChromaDB. Load in your PDFs, ask questions about them, and get answers grounded in what's actually in the documents — not hallucinated from training data.

Built as a Jupyter notebook so every step of the pipeline is visible and easy to follow.

---

## What it does

You drop in a PDF, it gets chunked and embedded into ChromaDB, and when you ask a question, the most relevant chunks are retrieved and handed to Llama as context. The model answers based on your document, not its general knowledge.

It's intentionally kept minimal — no heavy abstractions — so you can see exactly what's happening at each stage of the RAG pipeline.

---

## How it's built

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Llama (via Groq API) |
| Embeddings | HuggingFace / LangChain Embeddings |
| Vector Store | ChromaDB |
| Document Type | PDFs |
| Interface | Jupyter Notebook |

The pipeline follows the standard RAG flow:

```
PDF → Chunking → Embedding → ChromaDB
                                  ↓
         Query → Retrieval → Context + Query → Llama (Groq) → Answer
```

---

## Getting started

### Prerequisites

```bash
pip install langchain langchain-community langchain-groq chromadb pypdf sentence-transformers jupyter
```

### API key setup

You'll need a Groq API key. Get one free at [console.groq.com](https://console.groq.com). Then either create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

Or set it directly in the notebook:

```python
import os
os.environ["GROQ_API_KEY"] = "your_api_key_here"
```

### Running the notebook

```bash
git clone https://github.com/ShidhantSah/RAG-implementation-with-LangChain.git
cd RAG-implementation-with-LangChain
jupyter notebook
```

Open the notebook, point it to your PDF, and run the cells top to bottom.

---

## Why Groq + Llama?

Groq runs Llama inference insanely fast — responses that would take seconds on a typical API come back almost instantly. And unlike OpenAI, you don't need a credit card to get started; the free tier is generous enough for building and testing a full RAG pipeline.

It's a genuinely good combo for local/experimental RAG work.

---

## Notes

- ChromaDB persists embeddings locally — no external database or cloud setup needed.
- Chunk size and overlap have a surprisingly large effect on answer quality. If answers feel off, that's usually the first thing to tune.
- Groq's free tier has rate limits, so if you're loading very large PDFs with many chunks, you might hit them during the embedding or retrieval phase.
- The pipeline works with any PDF — research papers, documentation, books, reports — just swap the file path.

---

## Why RAG

Fine-tuning is expensive and overkill for most use cases. Stuffing entire documents into the prompt hits context limits fast. RAG is the practical middle ground — make the LLM aware of your specific documents without retraining anything. Building it from scratch rather than using a prebuilt wrapper makes it obvious why each component exists and where things break.

---

## License

MIT — use it however you like.
