# OpsPilot — Learning Log

## Phase 0 — Environment setup (2026-06-14)

- **Virtual environments** give each project its own isolated "toolbox" of packages, so
  different projects can't break each other's dependency versions. Created with
  `python -m venv .venv`, activated with `.\.venv\Scripts\Activate.ps1`.
- **`.gitignore` gotcha:** comments must be on their own line. A trailing comment after a
  pattern (`.env  # secret`) makes Git treat the *whole line* as a literal filename, so the
  pattern silently fails to ignore anything. Always verify with `git check-ignore <file>`.
- **Secrets** live in `.env` (gitignored, real key) with a committed `.env.example` template.
  Loaded into the program with `python-dotenv`'s `load_dotenv()` + `os.getenv()` — never
  hardcoded.
- **PyTorch has two builds:** plain `pip install torch` can give the CPU-only version. To use
  the GPU you must install from PyTorch's CUDA index
  (`--index-url https://download.pytorch.org/whl/cu121`). The `+cu121` suffix in the version
  string confirms the GPU build. Verified with `torch.cuda.is_available()`.
- **Embeddings** turn text into a list of numbers (a 384-dim vector for `all-MiniLM-L6-v2`)
  where similar *meaning* → nearby vectors. This is the foundation of RAG retrieval. Running
  the model locally on the GPU (`device="cuda"`) costs zero API calls.
- **SDK currency matters:** `google-generativeai` is deprecated; switched to the supported
  `google-genai` SDK (one `Client` object, model name passed per-call).
