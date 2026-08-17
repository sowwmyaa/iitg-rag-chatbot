# Backend : IITG RAG Chatbot API

FastAPI backend powering a conversational RAG chatbot for IIT Guwahati. Answers are grounded in the institute's own documents (PDFs, images, tables) via retrieval-augmented generation.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Orchestration | LangChain + LangGraph |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Content moderation | Groq (`openai/gpt-oss-safeguard-20b`) |
| Embeddings | HuggingFace sentence-transformers |
| Vector store | ChromaDB |
| PDF text extraction | PyPDF2 |
| PDF table extraction | pdfplumber |
| PDF image extraction | PyMuPDF |
| Image captioning | Moondream2 (local, ingest-time only) |

## Project Structure

```
backend/
├── main.py                    FastAPI app-— the deployed server
├── vectorize_documents.py     Builds the vector DB from data/
├── vision_captioner.py        Moondream2 image captioning (ingest-time only)
├── requirements.txt           Deployed server dependencies
├── requirements-vision.txt    Local-only deps for image/table ingestion
├── runtime.txt                Pins Python version for deployment
├── config.json.example        Template for local API key config
├── data/                      Source documents (PDFs, .txt, images)
└── vector_db_dir/             Persisted Chroma vector database
```

## How It Works

1. **Ingestion** (`vectorize_documents.py`, run locally): reads every file in `data/`, extracts text (PyPDF2), tables (pdfplumber → Markdown), and images (PyMuPDF → captioned by Moondream2), then embeds everything into `vector_db_dir/`.
2. **Serving** (`main.py`, what's deployed): on each chat request, a LangGraph pipeline runs:
   - **Moderate** - classifies the message as allowed, abusive, or off-topic
   - **Condense** - rewrites follow-up questions into standalone ones using chat history
   - **Retrieve** - semantic search against ChromaDB
   - **Generate** - Groq LLM answers using retrieved context

The deployed server never runs Moondream2/PyMuPDF/pdfplumber - those only run locally when you rebuild the vector DB.

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Create `config.json` from the example and add your Groq API key:
```json
{ "GROQ_API_KEY": "gsk_your_key_here" }
```

## Building the Vector Database

Basic (PDFs and .txt only):
```bash
python vectorize_documents.py
```

To also process images and extract tables/embedded images from PDFs:
```bash
pip install -r requirements-vision.txt
python vectorize_documents.py
```

Delete `vector_db_dir/` before rerunning if you've changed the contents of `data/`.

## Running Locally

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Check it's alive: `http://localhost:8000/api/health`

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/about` | Bot metadata for the frontend sidebar |
| POST | `/api/chat` | Send a message, get a response |
| POST | `/api/session/reset` | Clear a session's conversation history |

## Deployment (Render)

- **Root Directory:** `backend`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variables:**
  - `GROQ_API_KEY` - your Groq API key
  - `CORS_ORIGINS` - your deployed frontend's URL (e.g. `https://yourapp.vercel.app`)
- `runtime.txt` pins Python to 3.11 - newer versions (3.13+) fail to build some dependencies from source.

## Notes

- `vector_db_dir/` is committed to this repo so the deployed backend has working data without needing a build-time ingestion step.
- Conversation memory is in-process (LangGraph's `MemorySaver`) - it resets on server restart and doesn't persist across deployments. This is intentional; no user data is stored long-term.
