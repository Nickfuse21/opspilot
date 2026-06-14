"""Central settings for OpsPilot. Every other module imports from here,
so changing a model name, path, or chunk size only happens in one place."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env so secrets (like GEMINI_API_KEY) are available via os.getenv.
load_dotenv()

# --- Paths -------------------------------------------------------------------
# This file lives at src/opspilot/config.py, so the project root is 3 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"  # source documents
CHROMA_DIR = PROJECT_ROOT / "chroma"                           # vector DB on disk
CHROMA_COLLECTION = "opspilot_kb"                              # name of the collection

# --- Secrets (loaded from .env, never hardcoded) -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Models ------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # local embeddings (Phase 0)
EMBEDDING_DEVICE = "cuda"                      # run embeddings on the GPU
GEMINI_MODEL = "gemini-flash-latest"           # used for final answers
GEMINI_MODEL_LITE = "gemini-flash-lite-latest" # cheaper model for simple steps

# --- Chunking knobs (explained in Step 4) ------------------------------------
CHUNK_SIZE = 500     # max characters per chunk
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks

# --- Retrieval ---------------------------------------------------------------
TOP_K = 3            # how many chunks to fetch per question
