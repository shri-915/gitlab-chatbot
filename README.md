# 🦊 GitLab Handbook Assistant

> An AI-powered RAG chatbot that answers questions about GitLab's public [Handbook](https://handbook.gitlab.com/) and [Direction](https://about.gitlab.com/direction/) pages — grounded strictly in those sources, with full citations and transparency.

![GitLab Handbook Assistant Screenshot](screenshot_placeholder.png)

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE (One-Time)                     │
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐  │
│   │  Scraper  │───▶│  Chunker │───▶│ Embedder │───▶│  Supabase   │  │
│   │ (BS4 +   │    │(LangChain│    │ (Local    │    │ (pgvector)  │  │
│   │ requests)│    │ RCTS)    │    │  BGE)     │    │             │  │
│   └──────────┘    └──────────┘    └──────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE (Real-Time)                      │
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐  │
│   │  User    │───▶│Guardrail │───▶│  Router  │───▶│  Retriever  │  │
│   │  Query   │    │(On-Topic │    │(Category │    │(Supabase    │  │
│   │          │    │ Check)   │    │ Classify)│    │ pgvector)   │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────┬──────┘  │
│                                                          │         │
│                   ┌──────────┐    ┌──────────┐    ┌──────▼──────┐  │
│                   │Streamlit │◀───│ Sources  │◀───│  Gemini 1.5 │  │
│                   │   UI     │    │& Evidence│    │    Flash    │  │
│                   └──────────┘    └──────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

1. **Scraper** → Crawls GitLab Handbook (≤300 pages) and Direction sub-pages using `requests` + `BeautifulSoup4`. Saves structured JSON with heading hierarchy preserved.
2. **Chunker** → Splits scraped text by sections using LangChain's `RecursiveCharacterTextSplitter` (~600 tokens, 100 overlap). Prepends H1/H2/H3 context to each chunk.
3. **Embedder** → Embeds all chunks locally in batches using Sentence-Transformers `BAAI/bge-base-en-v1.5` (768 dimensions), avoiding Gemini embedding quota limits.
4. **Supabase** → Stores chunks with embeddings in a `pgvector`-enabled table. Uses HNSW indexing for fast cosine similarity search. SHA-256 hash deduplication.
5. **Guardrail** → Lightweight Gemini classifier rejects off-topic queries before RAG runs.
6. **Router** → Classifies queries into categories (engineering, people_ops, product_direction, values_culture, general) and boosts retrieval from matching sources.
7. **Retriever** → Embeds the query, calls `match_chunks` RPC in Supabase, returns top-5 chunks with similarity scores.
8. **Generator** → Builds a grounded prompt with retrieved context + conversation history, calls Gemini 1.5 Flash with strict rules.
9. **Streamlit UI** → Renders answers with confidence indicators, clickable source citations, collapsible evidence panels, and onboarding mode.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Core RAG Pipeline** | Scrape → Chunk → Embed → Store → Retrieve → Generate with full source grounding |
| **Source Citations** | Every answer includes clickable source links with section titles |
| **On-Topic Guardrailing** | Lightweight classifier rejects off-topic queries with a helpful redirect message |
| **Evidence Panel** | Collapsible panel showing raw chunks, similarity scores, and source URLs for verification |
| **Confidence Gating** | Amber warning when top similarity score < 0.75; prevents hallucination on weak matches |
| **Section-Aware Routing** | Query classification boosts retrieval from relevant sources (handbook vs. direction) |
| **Onboarding Mode** | Sidebar toggle with 8 curated starter questions for new GitLab employees |
| **Conversation Memory** | Last 3 turns of conversation context passed to Gemini for natural follow-ups |

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (needed for answer generation with Gemini Flash)
- A [Supabase](https://supabase.com/) project (free tier available)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/gitlab-chatbot.git
cd gitlab-chatbot
```

### Step 2: Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Set Up Supabase

1. Create a new Supabase project at [supabase.com](https://supabase.com/)
2. Go to **SQL Editor** in your Supabase dashboard
3. Copy the contents of `supabase_setup.sql` and run it
4. This creates the `gitlab_chunks` table, HNSW index, and `match_chunks` function

### Step 4: Set Environment Variables

**For local development:**

```bash
cp .env.example .env
# Edit .env and fill in your actual values:
# GOOGLE_API_KEY=your_key_here  # Needed for generation, not embeddings
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_KEY=your_anon_key_here
```

**For Streamlit:**

```bash
# Edit .streamlit/secrets.toml with your actual values
```

### Step 5: Run the Scraper (One-Time)

```bash
python scraper/scrape.py
```

This crawls GitLab's Handbook (up to 300 pages) and Direction pages, saving structured data to `scraped_data.json`. Takes ~10-15 minutes depending on your connection.

### Step 6: Run the Ingestion Pipeline (One-Time)

```bash
python pipeline/ingest.py
```

This reads `scraped_data.json`, chunks the content, embeds each chunk, and upserts everything to Supabase. Takes ~5-10 minutes depending on the number of chunks.

### Step 7: Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click **"New app"** and select your repository
4. Set the main file path to `app.py`
5. Go to **"Advanced settings"** → **"Secrets"** and paste your secrets:

```toml
GOOGLE_API_KEY = "your_actual_key"
SUPABASE_URL = "your_actual_url"
SUPABASE_KEY = "your_actual_key"
```

6. Click **"Deploy"**

> **Note:** Make sure the ingestion pipeline has already been run and data exists in your Supabase database before deploying. The scraper and ingestion scripts are not run on Streamlit Cloud — they only need to run once locally.

---

## 📁 Project Structure

```
gitlab-chatbot/
├── .streamlit/
│   └── secrets.toml          # Streamlit secrets template
├── scraper/
│   ├── __init__.py
│   └── scrape.py              # Full scraper for Handbook + Direction
├── pipeline/
│   ├── __init__.py
│   ├── chunk.py               # LangChain chunking logic
│   ├── embed.py               # Local sentence-transformers embedding
│   └── ingest.py              # Orchestrator: scrape → chunk → embed → upsert
├── rag/
│   ├── __init__.py
│   ├── retriever.py           # Supabase vector search
│   ├── guardrail.py           # On-topic classifier
│   ├── router.py              # Section-aware query router
│   └── chain.py               # Full RAG chain orchestration
├── app.py                     # Streamlit UI (main entry point)
├── supabase_setup.sql         # Database setup script
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🔧 Technical Details

### Scraping Strategy
- BFS crawl sorted by URL depth (shallower = higher priority)
- Handbook capped at 300 pages; Direction crawls all `/direction/` sub-pages
- 1-second polite delay between requests
- Skips PDFs, images, external links, and anchor-only links
- Preserves heading hierarchy (H1 → H2 → H3) per section

### Chunking Strategy
- LangChain `RecursiveCharacterTextSplitter` with ~600 token chunks and 100 token overlap
- Each chunk is prefixed with its heading context (`[Page Title] Section > Subsection`)
- Separators: `\n\n` → `\n` → `. ` → ` ` → `""`

### Embedding & Storage
- Local `BAAI/bge-base-en-v1.5` embeddings (768 dimensions)
- Batched locally with configurable batch size and retry
- HNSW index on Supabase for sub-second similarity search
- SHA-256 chunk hash deduplication prevents re-insertion

### RAG Pipeline
- Guardrail: zero-temperature Gemini classifier (ON_TOPIC / OFF_TOPIC)
- Router: category classification with source_type boosting
- Retrieval: cosine similarity via Supabase RPC, top-5 chunks
- Generation: Gemini 1.5 Flash with strict grounding rules
- Confidence gate: top similarity < 0.75 triggers low-confidence warning

---

## 📝 License

This project is for educational and evaluation purposes. GitLab's Handbook and Direction content is publicly available under their respective licenses.

---

Built with ❤️ using Gemini + RAG + Streamlit
