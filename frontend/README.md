## Frontend (planned)

This folder is reserved for a future **frontend** (web UI / dashboard) that will sit on top of the RAG backend.

Suggested approach (when you implement it):

- Initialize a JS/TS framework here (for example):
  - `npx create-next-app@latest .` (Next.js)
  - or `npm create vite@latest .` (Vite + React/Vue/Svelte)
- Talk to the Python backend via an HTTP/REST or GraphQL API that wraps the RAG pipeline implemented in the root Python files.
- Keep build artefacts (`.next/`, `dist/`, `.vite/`, etc.) out of git – they are already covered in the root `.gitignore`.

For now this is just a placeholder to keep the **repo modular** (backend + future frontend) for GitHub.

