# College RAG Chatbot — FastAPI backend (React frontend calls this)
# Same RAG pipeline as the original Streamlit app: Chroma + HuggingFace embeddings + Groq LLM.
# Conversational logic runs as a small LangGraph: condense question -> retrieve -> generate.
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import os
import json
import uuid
from typing import Annotated
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
working_dir = os.path.dirname(os.path.realpath(__file__))


def load_groq_api_key():
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key
    config_path = f"{working_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)["GROQ_API_KEY"]
    raise RuntimeError(
        "GROQ_API_KEY not found. Set it as an environment variable or add config.json."
    )


GROQ_API_KEY = load_groq_api_key()
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

MODEL_NAME = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Prompts (unchanged from the original app)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You are a **specialized AI assistant** dedicated exclusively to **IIT Guwahati** and its services. Your responses must be **accurate, concise, and strictly based on IIT Guwahati's verified data**.

Your goals:

1. Quickly understand the user's needs with **minimal follow-up questions**.
2. Provide **clear, concise, helpful answers** using IIT Guwahati data.
3. Suggest **relevant IIT Guwahati services** when appropriate.
4. Maintain a **warm, professional, and empathetic tone**.

---

### **INTERACTION GUIDELINES**

#### **PHASE 1 - Fast Intake (Always Do First)**

Before giving detailed answers, ask the **fewest possible follow-up questions** to collect essential info (aim for 1-3 total).

**Stop asking once you have enough info to answer effectively.**

---

#### **PHASE 2 - RESPOND USING USER DATA + IIT Guwahati DATA**

* Use the user data + IIT Guwahati data only.
* Deliver clear, short, and high-value responses.
* Use bullet points for readability.
* Add **1-3 relevant emojis** to support tone.

---

#### **PHASE 3 - RELATED SERVICE SUGGESTIONS**

After the main answer, suggest **1-2 IIT Guwahati services** that match the user's needs.

---

### **CONTEXT & RESPONSE RULES**

1. If provided context contains relevant IIT Guwahati info -> build on it.
2. If context is empty or irrelevant -> politely inform the user you can only discuss IIT Guwahati topics.
3. Always answer using **verified IIT Guwahati data** only.

---

### **TONE RULES**

* Warm, empathetic, supportive.
* Professional but friendly.
* Short, concise, and informative.
* Fact-based.
"""

DEFAULT_NEGATIVE_PROMPT = """
- Do **NOT** provide any information that is **not supported by verified IIT Guwahati data** or the provided system context.
- Do **NOT** imply you are an **employee, representative, agent, or official spokesperson** of IIT Guwahati.
- Do **NOT** fabricate or invent IIT Guwahati **services, features, pricing, policies, internal processes, or proprietary details**.
- Do **NOT** offer **legal, financial, medical, or other unrelated professional advice** outside IIT Guwahati's domain.
- Do **NOT** respond to topics **outside IIT Guwahati's scope**; instead, politely state that the relevant data is not available.
- Do **NOT** guess or assume **confidential, internal, or sensitive business information** about IIT Guwahati.
- Do **NOT** generate speculative, generic, or hypothetical business advice that is **not grounded in IIT Guwahati's verified information**.
- Do **NOT** use, cite, or reference **external sources, external knowledge, or outside databases** beyond the authorized IIT Guwahati context.
- Do **NOT** insert personal opinions, assumptions, unfounded claims, or subjective judgments.
- Do **NOT** mislead the user with unsupported or speculative responses.
- Do **NOT** use an unprofessional, casual, or overly familiar tone; maintain professionalism at all times.
"""


# Groq's model for policy-based content classification
MODERATION_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

MODERATION_POLICY = """You are a content moderator for a college assistant chatbot scoped to IIT Guwahati.

Classify the user's message into exactly one category:
- BLOCK_ABUSE: contains profanity, harassment, hate speech, or abusive language.
- BLOCK_OFFTOPIC: not related to IIT Guwahati academics, admissions, campus life,
  facilities, policies, or student services (e.g. "where is my water bottle",
  general trivia, requests unrelated to a college).
- ALLOW: a reasonable question about IIT Guwahati, or a normal conversational
  message (greetings, thanks, clarifying follow-ups) that doesn't need to
  mention IIT Guwahati by name to be legitimate.

Reply with exactly one word: BLOCK_ABUSE, BLOCK_OFFTOPIC, or ALLOW. Nothing else."""


def moderate_message(question: str) -> str:
    """Returns 'BLOCK_ABUSE', 'BLOCK_OFFTOPIC', or 'ALLOW'."""
    llm = ChatGroq(model=MODERATION_MODEL_NAME, temperature=0)
    result = llm.invoke([
        {"role": "system", "content": MODERATION_POLICY},
        {"role": "user", "content": question},
    ])
    verdict = result.content.strip().upper()
    if "BLOCK_ABUSE" in verdict:
        return "BLOCK_ABUSE"
    if "BLOCK_OFFTOPIC" in verdict:
        return "BLOCK_OFFTOPIC"
    return "ALLOW"  # fail open on unexpected output — don't block on a parsing miss


MODERATION_MESSAGES = {
    "BLOCK_ABUSE": "Let's keep the conversation respectful. Please rephrase your question, and I'll be happy to help with anything related to IIT Guwahati.",
    "BLOCK_OFFTOPIC": "It seems you may be asking questions outside my context, please ask questions related to IIT Guwahati only.",
}


# ---------------------------------------------------------------------------
# RAG setup — vectorstore loaded once at startup
# ---------------------------------------------------------------------------
def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


VECTORSTORE = None


# ---------------------------------------------------------------------------
# Conversational RAG graph — condense question -> retrieve -> generate
#
# Why a graph instead of one big chain: each step is now a plain, readable
# function you can independently test, tweak, or extend later (e.g. add a
# routing step before "condense", or a reranking step after "retrieve")
# without restructuring everything else.
#
# History persistence: LangGraph's checkpointer stores the full message
# list per `thread_id` (we use the frontend's session_id) automatically —
# no manual SESSIONS dict to keep in sync. Using MemorySaver here (in
# process memory) to keep dependencies minimal; swapping in a persistent
# checkpointer (e.g. SqliteSaver) later is a one-line change if you want
# conversations to survive a server restart.
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    standalone_question: str
    context: str
    moderation_verdict: str


def moderate_node(state: ChatState):
    latest_question = state["messages"][-1].content
    verdict = moderate_message(latest_question)
    return {"moderation_verdict": verdict}


def moderation_router(state: ChatState) -> str:
    """Blocked messages skip straight to a canned response; allowed ones proceed to retrieval."""
    return "blocked" if state["moderation_verdict"] != "ALLOW" else "condense"


def blocked_response_node(state: ChatState):
    message = MODERATION_MESSAGES.get(
        state["moderation_verdict"], MODERATION_MESSAGES["BLOCK_OFFTOPIC"])
    return {"messages": [AIMessage(content=message)]}


def _format_history(messages) -> str:
    lines = []
    for m in messages:
        speaker = "Human" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def condense_question_node(state: ChatState):
    messages = state["messages"]
    latest_question = messages[-1].content

    # No prior turns -> nothing to resolve, use the question as-is.
    if len(messages) <= 1:
        return {"standalone_question": latest_question}

    history_text = _format_history(messages[:-1])
    rewrite_prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up question as a standalone question that includes any context "
        "it implicitly refers to (e.g. resolve 'it', 'that', pronouns, or "
        "shortened references). If the question is already standalone, return "
        "it unchanged. Reply with only the rewritten question, nothing else.\n\n"
        f"Chat History:\n{history_text}\n\n"
        f"Follow-up question: {latest_question}\n"
        "Standalone question:"
    )
    llm = ChatGroq(model=MODEL_NAME, temperature=0)
    rewritten = llm.invoke(rewrite_prompt).content.strip()
    return {"standalone_question": rewritten or latest_question}


def retrieve_node(state: ChatState):
    retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["standalone_question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context}


def generate_node(state: ChatState):
    llm = ChatGroq(model=MODEL_NAME, temperature=0)
    history_text = _format_history(state["messages"][:-1])
    latest_question = state["messages"][-1].content

    prompt = f"""{DEFAULT_SYSTEM_PROMPT}

{DEFAULT_NEGATIVE_PROMPT}

Context:
{state['context']}

Chat History:
{history_text}

Question: {latest_question}

Answer:"""

    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("moderate", moderate_node)
    builder.add_node("blocked", blocked_response_node)
    builder.add_node("condense", condense_question_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "moderate")
    builder.add_conditional_edges("moderate", moderation_router, {
        "blocked": "blocked",
        "condense": "condense",
    })
    builder.add_edge("blocked", END)
    builder.add_edge("condense", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=MemorySaver())


GRAPH = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="College RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    global VECTORSTORE, GRAPH
    VECTORSTORE = setup_vectorstore()
    GRAPH = build_graph()


class MessageRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/about")
async def about():
    return {
        "title": "IIT Guwahati Chatbot",
        "description": "An AI-powered chatbot designed to provide answers related to IIT Guwahati.",
        "goals": [
            "Student Support",
            "Admissions Guidance",
            "Academic Information",
            "Campus Services",
            "Program Details",
            "Accessibility",
        ],
        "purpose": (
            "Designed as a seamless, user-friendly entry point to IIT Guwahati's support "
            "system, this chatbot helps students and prospective students easily access "
            "accurate information without confusion or hesitation. Whether users have "
            "questions about admissions, academic programs, campus facilities, student "
            "services, or general assistance, the chatbot provides clear explanations, "
            "reliable guidance, and context-aware responses powered by IIT Guwahati's "
            "verified knowledge base."
        ),
        "values": [
            "Student-Centered", "Accessibility", "Accuracy", "Transparency",
            "Professionalism", "Inclusivity", "Excellence", "Support",
            "Integrity", "Continuous Improvement",
        ],
    }


@app.post("/api/chat")
async def chatbot(request: MessageRequest):
    if VECTORSTORE is None or GRAPH is None:
        raise HTTPException(status_code=503, detail="Not ready yet.")

    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=400, detail="message must not be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    config = {"configurable": {"thread_id": session_id}}
    result = GRAPH.invoke(
        {"messages": [HumanMessage(content=message)]}, config=config)
    answer = result["messages"][-1].content

    return {"response": answer, "session_id": session_id}


@app.post("/api/session/reset")
async def reset_session(session_id: str):
    # Session state now lives in the graph's checkpointer, keyed by thread_id.
    # The frontend already generates a brand-new session_id for "new
    # conversation", so the old thread is simply abandoned — nothing to clean
    # up explicitly with MemorySaver (it's process memory, freed on restart).
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
