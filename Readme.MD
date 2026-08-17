# IITG RAG Chatbot

A conversational AI assistant for IIT Guwahati, answering questions using the institute's own documents - admissions info, academic policies, campus services, timetables, and more.

## What it does

Ask a question, get an answer grounded in real IITG documents - not a generic AI response. Understands follow-up questions, filters abusive or off-topic messages, and can read content from PDFs, tables, and images.

## Structure

```
├── backend/     FastAPI + LangChain/LangGraph + ChromaDB + Groq - the API
└── frontend/    React (Vite) — the chat UI
```

See `backend/README.md` for backend details.

## Tech Stack

**Backend:** FastAPI, LangChain, LangGraph, ChromaDB, Groq LLMs, HuggingFace embeddings, PyMuPDF, pdfplumber, Moondream2
**Frontend:** React, Vite

## Running Locally

Two terminals - backend first, then frontend.

```bash
# Terminal 1 - backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

# Terminal 2 - frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Deployment

- **Backend** → [Render](https://render.com) (root: `backend`, see `backend/README.md` for exact build/start commands and required environment variables)
- **Frontend** → [Vercel](https://vercel.com) (root: `frontend`, set `VITE_API_URL` to the deployed backend's URL)

Once both are live, the frontend calls the backend over a plain REST API — no other setup needed.

## Live Demo

_Add your deployed URL here once live._
