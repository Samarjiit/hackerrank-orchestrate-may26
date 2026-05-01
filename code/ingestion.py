from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from tqdm import tqdm

# ChromaDB 1.x moved embedding functions; try both import paths
try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as _STEF
except ImportError:
    try:
        from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
            SentenceTransformerEmbeddingFunction as _STEF,
        )
    except ImportError:
        from chromadb.utils import embedding_functions as _ef  # type: ignore
        _STEF = _ef.SentenceTransformerEmbeddingFunction  # type: ignore

import config


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ParsedDoc:
    content: str
    title: str
    company: str
    category: str
    source: str
    path: str


@dataclass
class Chunk:
    content: str
    title: str
    company: str
    category: str
    source: str
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.chunk_id:
            h = hashlib.md5(f"{self.company}:{self.source}:{self.content[:80]}".encode()).hexdigest()[:12]
            self.chunk_id = h


# ── Markdown helpers ───────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")
_URL_RE = re.compile(r"\[([^\]]+)\]\(https?://[^\)]+\)")
_BARE_URL_RE = re.compile(r"https?://\S+")
_HEADING_RE = re.compile(r"\n(?=#{1,4} )")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (meta, body)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text = m.group(1)
    body = text[m.end():]
    meta: dict = {}
    for line in fm_text.splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"')
    return meta, body


def _clean_markdown(text: str) -> str:
    """Remove images, convert linked text, strip bare URLs."""
    text = _IMAGE_RE.sub("", text)
    text = _URL_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_company(path: Path) -> str:
    """Detect company from file path."""
    for part in path.parts:
        low = part.lower()
        if low == "hackerrank":
            return "hackerrank"
        if low == "claude":
            return "claude"
        if low == "visa":
            return "visa"
    return "unknown"


def _detect_category(path: Path, company: str) -> str:
    """Extract category from path relative to company directory."""
    parts = path.parts
    try:
        # Find the index of the company directory
        company_idx = next(
            i for i, p in enumerate(parts) if p.lower() == company
        )
        # Category is everything between company dir and filename
        cat_parts = parts[company_idx + 1 : -1]
        return "/".join(cat_parts) if cat_parts else "general"
    except StopIteration:
        return "general"


# ── Document loading ───────────────────────────────────────────────────────────

def load_document(path: Path) -> Optional[ParsedDoc]:
    """Load and parse a single markdown file."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    meta, body = _parse_frontmatter(raw)
    body = _clean_markdown(body)

    if len(body.strip()) < 50:
        return None

    company = _detect_company(path)
    category = _detect_category(path, company)
    title = meta.get("title", path.stem.replace("-", " ").title())
    source = meta.get("source_url", str(path))

    return ParsedDoc(
        content=body,
        title=title,
        company=company,
        category=category,
        source=source,
        path=str(path),
    )


def load_corpus(data_dir: Path) -> list[ParsedDoc]:
    """Walk the data directory and load all markdown documents."""
    docs: list[ParsedDoc] = []
    md_files = list(data_dir.rglob("*.md"))

    for md_path in md_files:
        # Skip index files — they are tables of contents, not support articles
        if md_path.stem == "index":
            continue
        doc = load_document(md_path)
        if doc:
            docs.append(doc)

    return docs


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_document(doc: ParsedDoc, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split a parsed document into overlapping chunks anchored at headings."""
    sections = _HEADING_RE.split(doc.content)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush(buf: list[str]) -> None:
        text = "\n\n".join(buf).strip()
        if len(text) >= 80:
            chunks.append(
                Chunk(
                    content=text,
                    title=doc.title,
                    company=doc.company,
                    category=doc.category,
                    source=doc.source,
                )
            )

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= chunk_size:
            if buffer_len + len(section) > chunk_size and buffer:
                flush(buffer)
                # Keep last item as overlap context
                if buffer and len(buffer[-1]) <= overlap:
                    buffer = [buffer[-1]]
                    buffer_len = len(buffer[-1])
                else:
                    buffer = []
                    buffer_len = 0
            buffer.append(section)
            buffer_len += len(section)
        else:
            # Large section: split by paragraph
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
            for para in paragraphs:
                if buffer_len + len(para) > chunk_size and buffer:
                    flush(buffer)
                    buffer = []
                    buffer_len = 0
                buffer.append(para)
                buffer_len += len(para)

    if buffer:
        flush(buffer)

    return chunks


# ── ChromaDB helpers ───────────────────────────────────────────────────────────

def get_chroma_collection(
    chroma_dir: Path = config.CHROMA_DIR,
    embedding_model: str = config.EMBEDDING_MODEL,
    collection_name: str = config.CHROMA_COLLECTION,
) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = _STEF(model_name=embedding_model)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def is_corpus_indexed(collection: chromadb.Collection) -> bool:
    """Return True if the collection already contains documents."""
    return collection.count() > 0


def index_corpus(
    data_dir: Path = config.DATA_DIR,
    chroma_dir: Path = config.CHROMA_DIR,
    embedding_model: str = config.EMBEDDING_MODEL,
    collection_name: str = config.CHROMA_COLLECTION,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    batch_size: int = 64,
    force: bool = False,
) -> chromadb.Collection:
    """
    Load, chunk, and index the entire support corpus into ChromaDB.
    Skips indexing if the collection already contains data (unless force=True).
    """
    collection = get_chroma_collection(chroma_dir, embedding_model, collection_name)

    if not force and is_corpus_indexed(collection):
        return collection

    if force:
        # Drop and recreate
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = get_chroma_collection(chroma_dir, embedding_model, collection_name)

    print("Loading support corpus from disk …")
    docs = load_corpus(data_dir)
    print(f"  → {len(docs)} documents loaded")

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))

    print(f"  → {len(all_chunks)} chunks created")

    # De-duplicate by chunk_id
    seen: set[str] = set()
    unique_chunks: list[Chunk] = []
    for c in all_chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique_chunks.append(c)

    print(f"  → {len(unique_chunks)} unique chunks after deduplication")
    print("Indexing into ChromaDB …")

    for i in tqdm(range(0, len(unique_chunks), batch_size), desc="Embedding"):
        batch = unique_chunks[i : i + batch_size]
        collection.add(
            documents=[c.content for c in batch],
            metadatas=[
                {
                    "company": c.company,
                    "category": c.category,
                    "title": c.title,
                    "source": c.source,
                }
                for c in batch
            ],
            ids=[c.chunk_id for c in batch],
        )

    print(f"  ✓ Indexed {len(unique_chunks)} chunks into '{collection_name}'")
    return collection
