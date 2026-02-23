## Contributing

### Setup

```bash
pip install -e ".[dev]"
```

### Code style

- Keep modules small and composable.
- Avoid hard-coding provider behavior in core pipeline code; use provider interfaces.
- Prefer adding new providers under `backend/providers/`.

### Running

```bash
python main.py ingest --reindex
python main.py query "test question"
```

