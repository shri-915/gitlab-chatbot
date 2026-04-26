"""
Ingestion Orchestrator
=======================
Reads scraped data from JSON, chunks it, embeds it, and upserts to Supabase.
Uses chunk text hash for deduplication — safe to run multiple times.

Usage:
    python pipeline/ingest.py
"""

import hashlib
import json
import os
import sys
import time

from dotenv import load_dotenv
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.chunk import chunk_pages
from pipeline.embed import embed_chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRAPED_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scraped_data.json"
)
SUPABASE_TABLE = "gitlab_chunks"
UPSERT_BATCH_SIZE = 50  # Rows per upsert batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_chunk_hash(text: str) -> str:
    """Compute SHA-256 hash of chunk text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_scraped_data(filepath: str = None) -> list[dict]:
    """Load scraped data from the JSON file."""
    path = filepath or SCRAPED_DATA_FILE
    if not os.path.exists(path):
        print(f"✗ Scraped data file not found: {path}")
        print("  Run the scraper first: python scraper/scrape.py")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} pages from {path}")
    return data


def _get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        print("✗ SUPABASE_URL and SUPABASE_KEY must be set.")
        print("  Set them in .env or as environment variables.")
        sys.exit(1)

    return create_client(url, key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
)
def _upsert_batch(client: Client, rows: list[dict]):
    """Upsert a batch of rows to Supabase, with retry logic."""
    client.table(SUPABASE_TABLE).upsert(
        rows,
        on_conflict="chunk_hash",  # Deduplication key
    ).execute()


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def run_ingestion(scraped_file: str = None):
    """
    Full ingestion pipeline:
    1. Load scraped data from JSON
    2. Chunk all pages
    3. Embed all chunks
    4. Upsert to Supabase with deduplication
    """
    print("=" * 60)
    print("GitLab Handbook Assistant — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load scraped data
    print("\n📄 Step 1: Loading scraped data...")
    pages = _load_scraped_data(scraped_file)

    # Step 2: Chunk pages
    print("\n✂️  Step 2: Chunking pages...")
    chunks = chunk_pages(pages)

    if not chunks:
        print("✗ No chunks produced. Check scraped data quality.")
        sys.exit(1)

    # Step 3: Embed chunks
    print("\n🧠 Step 3: Embedding chunks...")
    chunks = embed_chunks(chunks)

    # Step 4: Upsert to Supabase
    print("\n💾 Step 4: Upserting to Supabase...")
    client = _get_supabase_client()

    # Prepare rows for upsert
    rows = []
    skipped = 0
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding or all(v == 0.0 for v in embedding):
            skipped += 1
            continue

        rows.append({
            "chunk_text": chunk["chunk_text"],
            "embedding": embedding,
            "source_url": chunk["source_url"],
            "section_title": chunk["section_title"],
            "page_title": chunk["page_title"],
            "source_type": chunk["source_type"],
            "chunk_hash": _compute_chunk_hash(chunk["chunk_text"]),
        })

    print(f"  Prepared {len(rows)} rows for upsert ({skipped} skipped due to missing embeddings)")

    # Upsert in batches
    total_upserted = 0
    for batch_start in range(0, len(rows), UPSERT_BATCH_SIZE):
        batch_end = min(batch_start + UPSERT_BATCH_SIZE, len(rows))
        batch = rows[batch_start:batch_end]

        try:
            _upsert_batch(client, batch)
            total_upserted += len(batch)
            progress = batch_end / len(rows) * 100
            print(f"  [{batch_end:>5}/{len(rows)}] ({progress:.1f}%) Upserted batch")
        except Exception as e:
            print(f"  ✗ Failed to upsert batch {batch_start}-{batch_end}: {e}")

        # Small delay between batches
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"✓ Ingestion complete!")
    print(f"  Pages processed : {len(pages)}")
    print(f"  Chunks created  : {len(chunks)}")
    print(f"  Rows upserted   : {total_upserted}")
    print(f"  Rows skipped    : {skipped}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    run_ingestion()
