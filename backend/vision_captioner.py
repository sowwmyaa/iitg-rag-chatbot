# Local image captioning using Moondream2 — CPU-friendly, no GPU required.
#
# IMPORTANT: this module is only imported by vectorize_documents.py at ingest
# time. It is never imported by main.py, so the deployed FastAPI backend
# never needs torch/transformers/Moondream at all — the deployed app only
# ever sees plain text (the captions this script generates), retrieved from
# Chroma exactly like PDF/text chunks. No architecture change to serving.
#
# First run will download the model (~3.7GB) — this can take a few minutes.
# Every run after that loads from the local HuggingFace cache and is fast.

from PIL import Image

MODEL_ID = "vikhyatk/moondream2"
# Pinned to a specific revision on purpose: Moondream updates frequently and
# some newer revisions pull in extra system dependencies (e.g. pyvips, which
# needs a system libvips install on Windows). This revision is confirmed to
# work with just `pip install transformers einops torch pillow`.
REVISION = "2024-04-02"

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(
            f"Loading Moondream2 (revision {REVISION})... first run downloads the model, may take a few minutes.")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            revision=REVISION,
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        print("Moondream2 loaded.")
    return _model, _tokenizer


def caption_image(image_path: str) -> str:
    """
    Return a detailed text description of the image, suitable for embedding
    and retrieval alongside PDF/text chunks.

    Note on limits: Moondream2 is a small (~1.9B param) general captioning
    model. It's reliable for describing scenes, diagrams, and general
    content, but is NOT guaranteed to accurately transcribe dense small text
    (e.g. a packed timetable grid or tiny labels on a campus map photo).
    For those specific cases, verify the output quality before trusting it
    in the knowledge base — an API-based vision model (Groq/Gemini) may be
    more reliable for text-heavy images. See vectorize_documents.py output
    for each image's caption so you can spot-check before relying on it.
    """
    model, tokenizer = _load_model()
    image = Image.open(image_path).convert("RGB")
    enc_image = model.encode_image(image)

    prompt = (
        "Describe this image in detail for someone who cannot see it. "
        "If it contains any text, tables, schedules, lists, or labeled "
        "diagrams, transcribe the visible text as precisely as possible, "
        "preserving row/column structure or labels where relevant."
    )
    description = model.answer_question(enc_image, prompt, tokenizer)
    return description.strip()
