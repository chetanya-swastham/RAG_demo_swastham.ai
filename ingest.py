"""
ingest.py — Ingestion pipeline for Swastham Forward-Translation RAG.

Reads a document (PDF or Markdown), chunks it with section-aware splitting,
computes embeddings via sentence-transformers, and stores them in a FAISS index.

Usage:
    python ingest.py --source documentRAG.md
    python ingest.py --source obesity_guide.pdf
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import faiss
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> str:
    """Extract full text from a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_markdown(filepath: str) -> str:
    """Read raw markdown text."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(filepath: str) -> str:
    """Auto-detect format and extract text."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in (".md", ".markdown", ".txt"):
        return extract_text_from_markdown(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# Section-Aware Chunking
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
# PDF text has no '#' markers; numbered section titles like "12. Origins and Philosophy"
# or "5a. Comorbidity Deep Dive: PCOS" are used as level-3 headings instead.
NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}[a-z]?\.)\s+([A-Z][^\n]{3,90})$", re.MULTILINE)

# Sections the corpus itself marks as safety-critical (documentRAG.md Section 36)
SAFETY_SECTIONS = {"15", "18", "28"}
SAFETY_KEYWORDS = ["sulfonylurea", "glimepiride", "take insulin", "on insulin",
                   "pregnan", "breastfeed", "eating disorder", "hypoglycaemia"]

# Maximum chunk size in characters (roughly 250–350 words)
CHUNK_MAX_SIZE = 1400
CHUNK_OVERLAP_CHARS = 150


def _split_by_sections(text: str) -> list[dict]:
    """Split text into sections using markdown headings as boundaries.
    
    Returns a list of dicts:  {"heading": str, "body": str, "level": int}
    """
    matches = list(HEADING_RE.finditer(text))
    numbered = not matches
    if numbered:
        matches = list(NUMBERED_HEADING_RE.finditer(text))
    if not matches:
        # No headings found — treat entire text as one section
        return [{"heading": "", "body": text.strip(), "level": 0}]

    sections = []
    for i, m in enumerate(matches):
        level = 3 if numbered else len(m.group(1))
        heading = f"{m.group(1)} {m.group(2).strip()}" if numbered else m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Keep empty-body headings (e.g. "## PART III") so the heading hierarchy stays correct
        sections.append({"heading": heading, "body": body, "level": level})
    return sections


def _split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
    """Split a long text block into overlapping pieces on sentence boundaries."""
    if len(text) <= max_size:
        return [text]

    # Try to split on paragraph breaks first, then sentences
    chunks = []
    current_start = 0
    while current_start < len(text):
        end = min(current_start + max_size, len(text))
        if end < len(text):
            # Try to find a paragraph break
            para_break = text.rfind("\n\n", current_start, end)
            if para_break > current_start + max_size // 3:
                end = para_break
            else:
                # Fall back to sentence boundary
                sent_break = text.rfind(". ", current_start, end)
                if sent_break > current_start + max_size // 3:
                    end = sent_break + 1
        chunk = text[current_start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        current_start = max(end - overlap, current_start + 1)
    return chunks


def chunk_document(text: str) -> list[dict]:
    """Chunk document into retrieval-friendly pieces.
    
    Each chunk is a dict: {"id": int, "heading": str, "text": str}
    The heading provides section context for each chunk.
    """
    sections = _split_by_sections(text)
    chunks = []
    chunk_id = 0

    # Build a heading hierarchy tracker
    heading_stack = []  # list of (level, heading)

    for section in sections:
        level = section["level"]
        heading = section["heading"]

        # Update heading stack
        if level > 0:
            # Remove headings at same or deeper level
            heading_stack = [(l, h) for l, h in heading_stack if l < level]
            heading_stack.append((level, heading))

        # Build full context heading
        # (the level-1 document title is skipped — it is identical for every chunk)
        context_heading = " > ".join(h for l, h in heading_stack if l > 1) or heading

        # Split body if too long
        body = section["body"]
        if not body:
            continue
        pieces = _split_long_text(body, CHUNK_MAX_SIZE, CHUNK_OVERLAP_CHARS)

        for piece in pieces:
            chunks.append({
                "id": chunk_id,
                "heading": context_heading,
                "text": piece,
                "safety_critical": _is_safety_critical(heading, piece),
            })
            chunk_id += 1

    return chunks


def _is_safety_critical(heading: str, text: str) -> bool:
    """Tag chunks carrying medication / pregnancy / child / eating-disorder cautions."""
    section = re.match(r"(\d{1,2})[a-z]?\.", heading)
    if section and section.group(1) in SAFETY_SECTIONS:
        return True
    lower = text.lower()
    return any(k in lower for k in SAFETY_KEYWORDS)


# ---------------------------------------------------------------------------
# Embedding & Index Building
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2") -> tuple:
    """Compute embeddings and build a FAISS index.
    
    Returns (faiss_index, embeddings_array).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    # Embed "heading: text" so the heading context influences retrieval
    texts_to_embed = [
        f"{c['heading']}: {c['text']}" if c["heading"] else c["text"]
        for c in chunks
    ]

    print(f"Computing embeddings for {len(texts_to_embed)} chunks ...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on L2-normalized = cosine
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dimension {dim}")
    return index, embeddings


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "data"
INDEX_META_FILE = "index_meta.json"


def save_artifacts(chunks: list[dict], index: faiss.Index, output_dir: Path = OUTPUT_DIR,
                   model_name: str = "all-MiniLM-L6-v2"):
    """Save chunks JSON, FAISS index and the embedding model name to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.json"
    index_path = output_dir / "faiss_index.bin"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks -> {chunks_path}")

    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index -> {index_path}")

    # Retrieval must embed queries with the same model, so record which one built the index
    meta = {"embedding_model": model_name, "chunks": len(chunks), "dimension": index.d}
    (output_dir / INDEX_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Stored canonical answers were built from the old chunks — discard them
    (output_dir / "answer_store.json").unlink(missing_ok=True)


def load_artifacts(data_dir: Path = OUTPUT_DIR) -> tuple:
    """Load persisted chunks and FAISS index."""
    chunks_path = data_dir / "chunks.json"
    index_path = data_dir / "faiss_index.bin"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index(str(index_path))
    return chunks, index


def load_index_meta(data_dir: Path = OUTPUT_DIR) -> dict:
    """Metadata written by save_artifacts ({} for an index built before it existed)."""
    try:
        return json.loads((data_dir / INDEX_META_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Retrieval helper (used by rag_pipeline.py)
# ---------------------------------------------------------------------------

_model_cache = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load (once) and return the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def retrieve(query: str, chunks: list[dict], index: faiss.Index,
             top_k: int = 5, model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """Retrieve top-k chunks most similar to the query."""
    model = get_embedding_model(model_name)

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG index.")
    parser.add_argument("--source", required=True, help="Path to PDF or Markdown file")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                        help="Embedding model name (default: EMBEDDING_MODEL from .env)")
    parser.add_argument("--output-dir", default=None, help="Output directory for index and chunks")
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"Error: file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print(f"Extracting text from: {args.source}")
    text = extract_text(args.source)
    print(f"Extracted {len(text)} characters")

    chunks = chunk_document(text)
    print(f"Created {len(chunks)} chunks")

    index, _ = build_index(chunks, model_name=args.model)
    save_artifacts(chunks, index, output_dir=out_dir, model_name=args.model)
    print("Done!")


if __name__ == "__main__":
    main()
