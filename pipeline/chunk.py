"""
Chunking Logic
===============
Takes scraped page data and splits it into chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter with section-aware splitting.
Preserves heading metadata (H1, H2, H3 context) per chunk.

Usage:
    from pipeline.chunk import chunk_pages
    chunks = chunk_pages(pages_data)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 600       # ~600 tokens (approximated as characters * 0.75)
CHUNK_OVERLAP = 100    # 100 token overlap
CHAR_MULTIPLIER = 4    # Rough chars-per-token estimate

# We configure the splitter in characters (chunk_size * multiplier)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE * CHAR_MULTIPLIER,         # ~2400 characters ≈ 600 tokens
    chunk_overlap=CHUNK_OVERLAP * CHAR_MULTIPLIER,   # ~400 characters ≈ 100 tokens
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    is_separator_regex=False,
)


# ---------------------------------------------------------------------------
# Core chunking function
# ---------------------------------------------------------------------------
def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Take a list of scraped page dicts and produce a flat list of chunks.

    Input format (from scraper):
        {
            "page_title": "...",
            "source_url": "...",
            "source_type": "handbook" | "direction",
            "sections": [
                {"heading_level": int, "title": "...", "text": "..."},
                ...
            ]
        }

    Output: list of chunk dicts:
        {
            "chunk_text": "...",
            "source_url": "...",
            "section_title": "...",
            "page_title": "...",
            "source_type": "handbook" | "direction",
        }
    """
    all_chunks = []

    for page in pages:
        page_title = page.get("page_title", "")
        source_url = page.get("source_url", "")
        source_type = page.get("source_type", "handbook")
        sections = page.get("sections", [])

        for section in sections:
            section_title = section.get("title", page_title)
            section_text = section.get("text", "").strip()

            if not section_text or len(section_text) < 30:
                continue

            # Prepend heading context to each chunk for better semantic search
            heading_prefix = f"[{page_title}] {section_title}\n\n"

            # Split section text into chunks
            text_chunks = TEXT_SPLITTER.split_text(section_text)

            for chunk_text in text_chunks:
                # Add heading context to the beginning of each chunk
                enriched_text = heading_prefix + chunk_text

                all_chunks.append({
                    "chunk_text": enriched_text,
                    "source_url": source_url,
                    "section_title": section_title,
                    "page_title": page_title,
                    "source_type": source_type,
                })

    print(f"✓ Chunking complete: {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


def chunk_single_page(page: dict) -> list[dict]:
    """Convenience function to chunk a single page dict."""
    return chunk_pages([page])
