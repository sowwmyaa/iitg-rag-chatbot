# Builds the vector database from everything in data/: PDFs, .txt files, and
# standalone images (.jpg/.jpeg/.png).
#
# For PDFs specifically, three things happen:
# 1. PyPDF2 extracts the normal running text (as before).
# 2. pdfplumber detects table structures and converts each to a clean
#    Markdown table, fixing PyPDF2's scrambled column order for native-text
#    tables.
# 3. PyMuPDF extracts embedded IMAGES (diagrams, scanned tables, photos) and
#    captions each with Moondream2 (see vision_captioner.py).
# All three get merged into one Document per PDF, then chunked/embedded
# exactly as before — no changes needed anywhere else in the app.
#
# Run this any time you add/remove files in data/. It rebuilds vector_db_dir
# from scratch each time (see README_RUN_DEPLOY.md for why).

import io
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

PDF_EXTENSIONS = (".pdf",)
TEXT_EXTENSIONS = (".txt",)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Skip embedded images smaller than this — PDFs are full of tiny decorative
# icons, bullet dots, and logos that aren't worth captioning and just add
# noise to the knowledge base.
MIN_EMBEDDED_IMAGE_SIZE = 100  # pixels, width and height


def extract_pdf_tables(file_path: str):
    """
    Detect table structures in a PDF using pdfplumber and convert each to a
    clean Markdown table. Returns a list of (page_number, markdown_string)
    tuples. Returns an empty list (with a warning) if pdfplumber isn't
    installed — PDF text extraction still works fine without it, you just
    keep PyPDF2's scrambled column order for tables.
    """
    try:
        import pdfplumber
    except ImportError:
        print("  (pdfplumber not installed — skipping table extraction. "
              "Run: pip install -r requirements-vision.txt)")
        return []

    results = []
    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception as e:
                print(
                    f"    Could not scan page {page_index + 1} for tables: {e}")
                continue

            for table in tables:
                if not table or len(table) < 2:
                    # skip empty or single-row "tables" (usually false positives)
                    continue

                # Clean cell values: pdfplumber returns None for empty cells
                rows = [[(cell or "").strip().replace("\n", " ")
                         for cell in row] for row in table]

                header, body_rows = rows[0], rows[1:]
                md_lines = [
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(["---"] * len(header)) + " |",
                ]
                for row in body_rows:
                    # Pad/truncate rows that don't match header length (ragged tables happen)
                    row = (row + [""] * len(header))[:len(header)]
                    md_lines.append("| " + " | ".join(row) + " |")

                results.append((page_index + 1, "\n".join(md_lines)))

    return results


def extract_pdf_images(file_path: str):
    """
    Extract embedded images from a PDF, page by page, using PyMuPDF.
    Returns a list of (page_number, PIL.Image) tuples. Returns an empty
    list (with a warning) if PyMuPDF isn't installed — PDF text extraction
    still works fine without it, you just lose embedded-image captioning.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError:
        print("  (PyMuPDF/Pillow not installed — skipping embedded image extraction. "
              "Run: pip install -r requirements-vision.txt)")
        return []

    results = []
    doc = fitz.open(file_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                pil_image = Image.open(io.BytesIO(base_image["image"]))
                if pil_image.width < MIN_EMBEDDED_IMAGE_SIZE or pil_image.height < MIN_EMBEDDED_IMAGE_SIZE:
                    continue  # skip tiny icons/logos
                results.append((page_index + 1, pil_image.convert("RGB")))
            except Exception as e:
                print(
                    f"    Could not extract an image on page {page_index + 1}: {e}")
                continue
    doc.close()
    return results


def load_pdf_document(file_path: str, filename: str) -> Document:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    # Detect tables and convert to clean Markdown (fixes PyPDF2's scrambled
    # column order for native-text tables)
    tables = extract_pdf_tables(file_path)
    if tables:
        print(f"  Found {len(tables)} table(s)")
        for page_num, markdown_table in tables:
            text += f"\n\n[Table on page {page_num}]\n{markdown_table}"

    # Extract and caption any embedded images (diagrams, scanned tables, photos)
    embedded_images = extract_pdf_images(file_path)
    if embedded_images:
        # lazy import — only needed if images found
        from vision_captioner import caption_pil_image
        print(
            f"  Found {len(embedded_images)} embedded image(s), captioning...")
        for page_num, pil_image in embedded_images:
            try:
                caption = caption_pil_image(
                    pil_image, source_hint=f"an image from page {page_num} of a document"
                )
                print(
                    f"    Page {page_num} image: {caption[:150]}{'...' if len(caption) > 150 else ''}")
                text += f"\n\n[Image on page {page_num}]: {caption}"
            except Exception as e:
                print(
                    f"    Could not caption an image on page {page_num}: {e}")
                continue

    return Document(page_content=text, metadata={"source": filename, "type": "pdf"})


def load_text_document(file_path: str, filename: str) -> Document:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
    return Document(page_content=text, metadata={"source": filename, "type": "txt"})


def load_image_document(file_path: str, filename: str) -> Document:
    # lazy import — only needed if an image file exists
    from vision_captioner import caption_image

    caption = caption_image(file_path)
    print(f"  Caption: {caption[:200]}{'...' if len(caption) > 200 else ''}")
    return Document(page_content=caption, metadata={"source": filename, "type": "image"})


def load_all_documents(directory: str):
    """Load and process every supported file in the given directory."""
    documents = []
    all_files = sorted(os.listdir(directory))

    for filename in all_files:
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext in PDF_EXTENSIONS:
                print(f"Processing PDF: {filename}...")
                documents.append(load_pdf_document(file_path, filename))
                print(f"  Done: {filename}")

            elif ext in TEXT_EXTENSIONS:
                print(f"Processing text file: {filename}...")
                documents.append(load_text_document(file_path, filename))
                print(f"  Done: {filename}")

            elif ext in IMAGE_EXTENSIONS:
                print(f"Processing image: {filename}...")
                documents.append(load_image_document(file_path, filename))
                print(f"  Done: {filename}")

            else:
                # silently skip unsupported files (.DS_Store, .gitkeep, etc.)
                continue

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            continue

    return documents


def main():
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF, .txt, and image files here.")
        return

    if not os.path.exists("vector_db_dir"):
        os.makedirs("vector_db_dir")
        print("Created 'vector_db_dir' directory for storing vectorized documents.")

    try:
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings()

        print("Loading and processing files from 'data'...")
        documents = load_all_documents("data")

        if not documents:
            print(
                "No documents were successfully processed. Please check your files in 'data/'.")
            return

        print(f"Successfully loaded {len(documents)} documents")

        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        text_chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(text_chunks)} chunks")

        print("Creating vector database...")
        Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )

        print("Successfully vectorized and stored documents in 'vector_db_dir'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure all required dependencies are installed:")
        print("pip install -r requirements.txt")
        print("For image support, also: pip install -r requirements-vision.txt")


if __name__ == "__main__":
    main()
