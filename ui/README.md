# PaperLens UI

This is a small demo UI for the existing FastAPI retrieval service. It does not
modify the current API code.

## Run

Start the retrieval API from the project root:

```bash
uvicorn app.main:app --reload --port 8000
```

Then start the UI server:

```bash
uvicorn ui.server:app --reload --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

If the retrieval API runs somewhere else, set:

```bash
PAPER_API_BASE_URL=http://127.0.0.1:8001 uvicorn ui.server:app --reload --port 7860
```
