"""
GitLab Handbook & Direction Scraper
====================================
Crawls GitLab's public Handbook and Direction pages, extracts text content
with heading hierarchy preserved, and saves structured data to JSON.

Usage:
    python scraper/scrape.py
"""

import json
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HANDBOOK_ROOT = "https://handbook.gitlab.com/"
DIRECTION_ROOT = "https://about.gitlab.com/direction/"

MAX_HANDBOOK_PAGES = 300
CRAWL_DELAY_SECONDS = 1

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scraped_data.json")

SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".mp4", ".mp3", ".zip", ".tar", ".gz", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".eot",
}

HEADERS = {
    "User-Agent": (
        "GitLabHandbookBot/1.0 "
        "(educational project; respectful crawling; +https://github.com)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _url_depth(url: str) -> int:
    """Return the path depth of a URL (number of non-empty segments)."""
    path = urlparse(url).path.strip("/")
    if not path:
        return 0
    return len(path.split("/"))


def _should_skip_url(url: str) -> bool:
    """Return True if the URL should be skipped (binary file, anchor, etc.)."""
    parsed = urlparse(url)
    # Skip fragment-only links
    if not parsed.scheme and not parsed.netloc and parsed.fragment:
        return True
    # Skip non-http(s) schemes
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return True
    # Skip file extensions we don't want
    path_lower = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return True
    return False


def _normalize_url(url: str) -> str:
    """Remove fragment, ensure trailing slash consistency."""
    url, _ = urldefrag(url)
    return url


def _extract_heading_text(element) -> str:
    """Get clean text from a heading element."""
    return element.get_text(strip=True) if element else ""


def _extract_page_content(soup: BeautifulSoup, url: str, source_type: str) -> dict:
    """
    Extract structured content from a parsed page.
    Returns a dict with page_title, source_url, source_type, and sections.
    Each section has: heading_level, title, and text content.
    """
    # Try to find the main content area
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("div", class_=re.compile(r"content|markdown|handbook", re.I))
        or soup.find("body")
    )

    if not main_content:
        return None

    # Get page title
    h1 = main_content.find("h1")
    page_title = _extract_heading_text(h1) if h1 else ""
    if not page_title:
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path

    # Build sections from heading hierarchy
    sections = []
    current_section = {
        "heading_level": 0,
        "title": page_title,
        "text": "",
    }

    # Track heading context for nesting
    heading_context = {1: page_title, 2: "", 3: ""}

    for element in main_content.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre", "code", "blockquote", "span", "div"]
    ):
        tag_name = element.name

        if tag_name in ("h1", "h2", "h3", "h4"):
            # Save previous section if it has content
            if current_section["text"].strip():
                sections.append(current_section.copy())

            level = int(tag_name[1])
            heading_text = _extract_heading_text(element)

            # Update heading context
            if level <= 3:
                heading_context[level] = heading_text
                # Clear lower-level headings
                for l in range(level + 1, 4):
                    heading_context[l] = ""

            # Build section title with hierarchy
            title_parts = [heading_context[l] for l in range(1, level + 1) if heading_context.get(l)]
            section_title = " > ".join(title_parts) if title_parts else heading_text

            current_section = {
                "heading_level": level,
                "title": section_title,
                "text": "",
            }

        elif tag_name in ("p", "li", "blockquote"):
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 5:  # Skip trivial fragments
                current_section["text"] += text + "\n"

        elif tag_name == "pre":
            # Code blocks — keep but don't expand too much
            code_text = element.get_text(strip=True)
            if code_text and len(code_text) < 1000:
                current_section["text"] += f"```\n{code_text}\n```\n"

        elif tag_name in ("td", "th"):
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_section["text"] += text + " | "

    # Don't forget the last section
    if current_section["text"].strip():
        sections.append(current_section)

    # Filter out empty sections and very short ones
    sections = [s for s in sections if len(s["text"].strip()) > 20]

    if not sections:
        return None

    return {
        "page_title": page_title,
        "source_url": url,
        "source_type": source_type,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Crawlers
# ---------------------------------------------------------------------------
def _crawl_domain(
    root_url: str,
    source_type: str,
    max_pages: int,
    domain_filter: str = None,
    path_prefix: str = None,
) -> list[dict]:
    """
    BFS-crawl starting from root_url.
    Discovers internal links, respects max_pages, sorts by URL depth
    (shallower first) to prioritize top-level pages.
    """
    parsed_root = urlparse(root_url)
    allowed_domain = domain_filter or parsed_root.netloc
    allowed_path = path_prefix or parsed_root.path

    visited = set()
    discovered = set()
    discovered.add(_normalize_url(root_url))

    pages = []

    print(f"\n{'='*60}")
    print(f"Starting crawl: {root_url}")
    print(f"  Domain filter : {allowed_domain}")
    print(f"  Path prefix   : {allowed_path}")
    print(f"  Max pages     : {max_pages}")
    print(f"{'='*60}\n")

    while discovered and len(visited) < max_pages:
        # Sort discovered URLs by depth (shallower first)
        sorted_urls = sorted(discovered - visited, key=_url_depth)
        if not sorted_urls:
            break

        url = sorted_urls[0]
        discovered.discard(url)
        visited.add(url)

        print(f"  [{len(visited):>3}/{max_pages}] Scraping: {url}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
            response.raise_for_status()

            # Only process HTML content
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract page content
            page_data = _extract_page_content(soup, url, source_type)
            if page_data:
                pages.append(page_data)

            # Discover new links
            for link_tag in soup.find_all("a", href=True):
                href = link_tag["href"]
                if _should_skip_url(href):
                    continue

                absolute_url = _normalize_url(urljoin(url, href))
                parsed = urlparse(absolute_url)

                # Must match allowed domain and path prefix
                if parsed.netloc != allowed_domain:
                    continue
                if not parsed.path.startswith(allowed_path):
                    continue

                if absolute_url not in visited:
                    discovered.add(absolute_url)

        except requests.RequestException as e:
            print(f"    ⚠ Error fetching {url}: {e}")
        except Exception as e:
            print(f"    ⚠ Unexpected error on {url}: {e}")

        # Polite crawl delay
        time.sleep(CRAWL_DELAY_SECONDS)

    print(f"\n  ✓ Crawl complete: {len(pages)} pages scraped from {root_url}\n")
    return pages


def scrape_handbook(max_pages: int = MAX_HANDBOOK_PAGES) -> list[dict]:
    """Scrape GitLab Handbook pages."""
    return _crawl_domain(
        root_url=HANDBOOK_ROOT,
        source_type="handbook",
        max_pages=max_pages,
        domain_filter="handbook.gitlab.com",
        path_prefix="/",
    )


def scrape_direction(max_pages: int = 100) -> list[dict]:
    """Scrape GitLab Direction pages."""
    return _crawl_domain(
        root_url=DIRECTION_ROOT,
        source_type="direction",
        max_pages=max_pages,
        domain_filter="about.gitlab.com",
        path_prefix="/direction/",
    )


def scrape_all(
    handbook_max: int = MAX_HANDBOOK_PAGES,
    direction_max: int = 100,
    output_file: str = None,
) -> list[dict]:
    """
    Scrape both Handbook and Direction pages and save to JSON.
    Returns the combined list of page data dicts.
    """
    output_path = output_file or OUTPUT_FILE

    print("=" * 60)
    print("GitLab Handbook & Direction Scraper")
    print("=" * 60)

    handbook_pages = scrape_handbook(max_pages=handbook_max)
    direction_pages = scrape_direction(max_pages=direction_max)

    all_pages = handbook_pages + direction_pages

    print(f"\nTotal pages scraped: {len(all_pages)}")
    print(f"  Handbook : {len(handbook_pages)}")
    print(f"  Direction: {len(direction_pages)}")

    # Save to JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return all_pages


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scrape_all()
