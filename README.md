# OpsPilot

An agentic enterprise support & operations copilot. It answers questions over a document
knowledge base **with citations**, takes multi-step actions across connected systems, and
has guardrails that block prompt injection and require human approval before any write action.

> Portfolio project for an AI Developer role. Built in deliberate, learnable phases.

## Status

**Phase 0 — Environment setup: complete.**
- Python 3.11 virtual environment + Git
- Local GPU embeddings (`all-MiniLM-L6-v2` on CUDA) verified
- Gemini API (`google-genai` SDK) verified

See [NOTES.md](NOTES.md) for the learning log.

## Tech stack

Python 3.11 · Gemini (generation) · sentence-transformers on GPU (embeddings) · Chroma
(vector DB) · LangGraph/LangChain (orchestration) · FastAPI · Streamlit · Docker.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows PowerShell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install python-dotenv google-genai sentence-transformers
cp .env.example .env                    # then add your GEMINI_API_KEY
```
