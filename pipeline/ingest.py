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
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential, retry_if_exception

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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_FILE = os.path.join(PROJECT_ROOT, ".env")
SUPABASE_TABLE = "gitlab_chunks"
UPSERT_BATCH_SIZE = 50  # Rows per upsert batch
DEFAULT_DIRECTION_MAX_PAGES = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_chunk_hash(text: str) -> str:
    """Compute SHA-256 hash of chunk text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _retry_non_value_errors(exc: Exception) -> bool:
    """Retry transient failures, but fail fast on configuration and validation errors."""
    return not isinstance(exc, ValueError)


def _load_scraped_data(
    filepath: str = None,
    auto_scrape: bool = True,
    handbook_max: int = 300,
    direction_max: int = DEFAULT_DIRECTION_MAX_PAGES,
) -> list[dict]:
    """Load scraped data from JSON; optionally scrape automatically if missing."""
    path = filepath or SCRAPED_DATA_FILE
    if not os.path.exists(path):
        if not auto_scrape:
            raise FileNotFoundError(
                f"Scraped data file not found: {path}. "
                "Run scraper/scrape.py first or enable auto-scrape."
            )

        print(f"⚠ Scraped data file not found: {path}")
        print("  Running scraper automatically before ingestion...")

        from scraper.scrape import scrape_all

        scrape_all(
            handbook_max=handbook_max,
            direction_max=direction_max,
            output_file=path,
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Scraper did not create expected file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} pages from {path}")
    return data


def _get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in env vars or .env"
        )

    return create_client(url, key)


def _ensure_required_env() -> None:
    """Fail fast with a clear message if required env vars are missing."""
    missing = [
        name for name in ("GOOGLE_API_KEY", "SUPABASE_URL", "SUPABASE_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise ValueError(
            "Missing required environment variables: "
            f"{', '.join(missing)}. "
            "Set them in .env at project root or export them before running ingestion."
        )


def _ensure_supabase_table_ready(client: Client) -> None:
    """Verify the target table exists and is reachable before upserts."""
    try:
        client.table(SUPABASE_TABLE).select("id").limit(1).execute()
    except Exception as e:
        message = str(e)
        if "PGRST205" in message or "Could not find the table" in message:
            raise ValueError(
                "Supabase table public.gitlab_chunks is missing. "
                "Run supabase_setup.sql in your Supabase SQL Editor first."
            ) from e
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception(_retry_non_value_errors),
)
def _upsert_batch(client: Client, rows: list[dict]):
    """Upsert a batch of rows to Supabase, with retry logic."""
    try:
        client.table(SUPABASE_TABLE).upsert(
            rows,
            on_conflict="chunk_hash",  # Deduplication key
        ).execute()
    except Exception as e:
        message = str(e).lower()
        if "row-level security" in message or "42501" in message:
            raise ValueError(
                "Supabase RLS blocked writes to gitlab_chunks. Re-run supabase_setup.sql "
                "to create insert/update policies, or use a service-role key for ingestion."
            ) from e
        raise


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def run_ingestion(
    scraped_file: str = None,
    auto_scrape: bool = True,
    handbook_max: int = 300,
    direction_max: int = DEFAULT_DIRECTION_MAX_PAGES,
) -> bool:
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

    try:
        _ensure_required_env()

        # Step 1: Load scraped data
        print("\n📄 Step 1: Loading scraped data...")
        pages = _load_scraped_data(
            filepath=scraped_file,
            auto_scrape=auto_scrape,
            handbook_max=handbook_max,
            direction_max=direction_max,
        )

        # Step 2: Chunk pages
        print("\n✂️  Step 2: Chunking pages...")
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError("No chunks produced. Check scraped data quality.")

        # Step 3: Embed chunks
        print("\n🧠 Step 3: Embedding chunks...")
        chunks = embed_chunks(chunks)

        # Step 4: Upsert to Supabase
        print("\n💾 Step 4: Upserting to Supabase...")
        client = _get_supabase_client()
        _ensure_supabase_table_ready(client)

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
        failed_batches = 0
        for batch_start in range(0, len(rows), UPSERT_BATCH_SIZE):
            batch_end = min(batch_start + UPSERT_BATCH_SIZE, len(rows))
            batch = rows[batch_start:batch_end]

            try:
                _upsert_batch(client, batch)
                total_upserted += len(batch)
                progress = batch_end / len(rows) * 100
                print(f"  [{batch_end:>5}/{len(rows)}] ({progress:.1f}%) Upserted batch")
            except RetryError as e:
                root_error = e.last_attempt.exception() if e.last_attempt else e
                print(f"  ✗ Failed to upsert batch {batch_start}-{batch_end}: {root_error}")
                failed_batches += 1
            except Exception as e:
                print(f"  ✗ Failed to upsert batch {batch_start}-{batch_end}: {e}")
                failed_batches += 1

            # Small delay between batches
            time.sleep(0.3)

        print(f"\n{'='*60}")
        print("✓ Ingestion complete!")
        print(f"  Pages processed : {len(pages)}")
        print(f"  Chunks created  : {len(chunks)}")
        print(f"  Rows upserted   : {total_upserted}")
        print(f"  Rows skipped    : {skipped}")
        print(f"  Batch failures  : {failed_batches}")
        print(f"{'='*60}\n")

        if rows and total_upserted == 0:
            print("✗ Ingestion did not upsert any rows. Check Supabase setup and credentials.\n")
            return False

        return True

    except RetryError as e:
        root_error = e.last_attempt.exception() if e.last_attempt else e
        print(f"\n✗ Ingestion failed: {root_error}\n")
        return False
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}\n")
        return False


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv(dotenv_path=DOTENV_FILE)
    success = run_ingestion()
    if not success:
        sys.exit(1)
