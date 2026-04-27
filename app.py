"""
GitLab Handbook Assistant — Streamlit UI
==========================================
Main application file for the RAG-based chatbot.
Provides a full chat interface with guardrailing, confidence indicators,
evidence panels, onboarding mode, and conversation memory.

"""
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — ensure imports work from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ---------------------------------------------------------------------------
# Configure API keys from Streamlit secrets (must happen before imports that use them)
# ---------------------------------------------------------------------------
def _load_secret_to_env(secret_name: str) -> None:
    """Load a Streamlit secret into env vars without crashing if secrets.toml is missing."""
    try:
        secret_value = st.secrets.get(secret_name)
    except Exception:
        secret_value = None

    if secret_value and secret_name not in os.environ:
        os.environ[secret_name] = secret_value


_load_secret_to_env("GOOGLE_API_KEY")
_load_secret_to_env("SUPABASE_URL")
_load_secret_to_env("SUPABASE_KEY")

from rag.chain import run_rag_chain


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GitLab Handbook Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — Premium styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Main title styling */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FC6D26;
        margin-bottom: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .main-subtitle {
        font-size: 0.95rem;
        color: #8B8D91;
        margin-top: 4px;
        margin-bottom: 20px;
    }

    /* ── Hero Section ──────────────────────────────────────────────── */
    .hero-section {
        background: linear-gradient(135deg, #0E0E2C 0%, #1A1A3E 50%, #2A1040 100%);
        border: 1px solid rgba(252, 109, 38, 0.25);
        border-radius: 20px;
        padding: 48px 40px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        text-align: center;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(252,109,38,0.12) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-section::after {
        content: '';
        position: absolute;
        bottom: -60px;
        left: -60px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(56,13,117,0.2) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-fox {
        font-size: 4rem;
        line-height: 1;
        margin-bottom: 16px;
        display: block;
        filter: drop-shadow(0 0 20px rgba(252,109,38,0.5));
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FC6D26 0%, #E24329 40%, #9B6DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #A8A8C0;
        max-width: 580px;
        margin: 0 auto 28px;
        line-height: 1.65;
    }

    .hero-badges {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }

    .hero-badge {
        background: rgba(252, 109, 38, 0.1);
        border: 1px solid rgba(252, 109, 38, 0.25);
        color: #FC6D26;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .hero-prompt-hint {
        font-size: 0.85rem;
        color: #6B6D80;
        margin-top: 20px;
    }

    /* ── Source link styling ─────────────────────────────────────────── */
    .source-link {
        display: inline-block;
        background: linear-gradient(135deg, #292961 0%, #380D75 100%);
        color: #E8E8FF !important;
        padding: 6px 14px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
        transition: all 0.2s ease;
        border: 1px solid rgba(252, 109, 38, 0.2);
    }

    .source-link:hover {
        background: linear-gradient(135deg, #FC6D26 0%, #E24329 100%);
        color: white !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(252, 109, 38, 0.3);
    }

    /* ── Confidence indicators ──────────────────────────────────────── */
    .confidence-high {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(46, 204, 113, 0.12);
        color: #2ecc71;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .confidence-low {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(241, 196, 15, 0.12);
        color: #f39c12;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* ── Category badge ──────────────────────────────────────────────── */
    .category-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(252, 109, 38, 0.1);
        color: #FC6D26;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 8px;
    }

    /* ── Evidence panel styling ──────────────────────────────────────── */
    .evidence-chunk {
        background: rgba(30, 30, 50, 0.4);
        border: 1px solid rgba(252, 109, 38, 0.15);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .evidence-chunk-title {
        font-weight: 600;
        color: #FC6D26;
        font-size: 0.88rem;
        margin-bottom: 6px;
    }

    .evidence-chunk-score {
        font-size: 0.75rem;
        color: #8B8D91;
        margin-bottom: 8px;
    }

    .evidence-chunk-text {
        font-size: 0.83rem;
        color: #C8C8D0;
        line-height: 1.5;
    }

    /* ── Onboarding suggestions ──────────────────────────────────────── */
    .suggestion-btn {
        display: block;
        width: 100%;
        text-align: left;
        background: linear-gradient(135deg, rgba(41, 41, 97, 0.5) 0%, rgba(56, 13, 117, 0.3) 100%);
        border: 1px solid rgba(252, 109, 38, 0.15);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        color: #E8E8FF;
        font-size: 0.84rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .suggestion-btn:hover {
        border-color: #FC6D26;
        background: linear-gradient(135deg, rgba(252, 109, 38, 0.15) 0%, rgba(226, 67, 41, 0.1) 100%);
    }

    /* ── Welcome message ────────────────────────────────────────────── */
    .welcome-box {
        background: linear-gradient(135deg, rgba(252, 109, 38, 0.08) 0%, rgba(226, 67, 41, 0.05) 100%);
        border: 1px solid rgba(252, 109, 38, 0.2);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .welcome-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #FC6D26;
        margin-bottom: 6px;
    }

    .welcome-text {
        font-size: 0.92rem;
        color: #C8C8D0;
        line-height: 1.6;
    }

    /* ── Sidebar styling ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E0E2C 0%, #1A1A3E 100%);
    }

    .sidebar-footer {
        font-size: 0.72rem;
        color: #6B6D71;
        text-align: center;
        margin-top: 30px;
        line-height: 1.5;
    }

    /* ── Error / off-topic cards ─────────────────────────────────────── */
    .offtopic-card {
        background: rgba(241, 196, 15, 0.07);
        border: 1px solid rgba(241, 196, 15, 0.28);
        border-radius: 12px;
        padding: 18px 20px;
        color: #D4AF37;
        line-height: 1.6;
    }

    .offtopic-card strong {
        color: #F1C40F;
        font-weight: 600;
    }

    .error-card {
        background: rgba(231, 76, 60, 0.07);
        border: 1px solid rgba(231, 76, 60, 0.28);
        border-radius: 12px;
        padding: 18px 20px;
        color: #E88080;
        line-height: 1.6;
    }

    .error-card strong {
        color: #E74C3C;
        font-weight: 600;
    }

    .lowconf-notice {
        background: rgba(241, 196, 15, 0.07);
        border-left: 3px solid #f39c12;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        color: #D4AF37;
        font-size: 0.84rem;
        margin-top: 10px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "ss_messages" not in st.session_state:
    st.session_state.ss_messages = []

if "ss_onboarding_mode" not in st.session_state:
    st.session_state.ss_onboarding_mode = False

if "ss_last_sources" not in st.session_state:
    st.session_state.ss_last_sources = []

if "ss_pending_question" not in st.session_state:
    st.session_state.ss_pending_question = None


# ---------------------------------------------------------------------------
# Onboarding Suggested Questions
# ---------------------------------------------------------------------------
SUGGESTED_QUESTIONS = [
    "What are GitLab's core values and how do they guide daily work?",
    "How does GitLab's remote-first culture actually work in practice?",
    "What is GitLab's product vision and where is it heading?",
    "How does code review work at GitLab?",
    "What is GitLab's approach to performance reviews and career growth?",
    "How are decisions made at GitLab — who has authority over what?",
    "What does GitLab's engineering department structure look like?",
    "How does GitLab handle asynchronous communication across time zones?",
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="main-title">🦊 GitLab Handbook Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="main-subtitle">AI-powered answers from GitLab\'s Handbook &amp; Direction</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Onboarding mode toggle
    onboarding = st.checkbox(
        "🎓 New Employee Mode",
        value=st.session_state.ss_onboarding_mode,
        key="onboarding_toggle",
        help="Show suggested starter questions for new GitLab team members",
    )
    st.session_state.ss_onboarding_mode = onboarding

    # Suggested questions (when onboarding mode is ON)
    if st.session_state.ss_onboarding_mode:
        st.markdown("#### Suggested Questions")
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if st.button(
                question,
                key=f"suggest_{i}",
                use_container_width=True,
            ):
                st.session_state.ss_pending_question = question
                st.rerun()

    st.divider()

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.ss_messages = []
        st.session_state.ss_last_sources = []
        st.session_state.ss_pending_question = None
        st.rerun()

    # Footer
    st.markdown(
        '<div class="sidebar-footer">'
        "Grounded in GitLab Handbook &amp; Direction<br>"
        "Built with Gemini Flash + RAG + Supabase<br><br>"
        "© 2025 GitLab Handbook Assistant"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

# ── Hero (shown only when no messages exist) ──────────────────────────────
if not st.session_state.ss_messages:
    st.markdown("""
<div class="hero-section">
    <span class="hero-fox">🦊</span>
    <div class="hero-title">GitLab Handbook Assistant</div>
    <div class="hero-subtitle">
        Ask anything about GitLab's culture, values, engineering practices, people operations, or product direction —
        every answer is grounded strictly in the official Handbook and Direction pages with full citations.
    </div>
    <div class="hero-badges">
        <span class="hero-badge">Handbook</span>
        <span class="hero-badge">Direction Pages</span>
        <span class="hero-badge">Gemini Flash</span>
        <span class="hero-badge">RAG + pgvector</span>
    </div>
    <div class="hero-prompt-hint">Type your question below to get started</div>
</div>
""", unsafe_allow_html=True)

# Onboarding welcome box (when onboarding mode is ON and no messages yet)
if st.session_state.ss_onboarding_mode and not st.session_state.ss_messages:
    st.markdown(
        '<div class="welcome-box">'
        '<div class="welcome-title">👋 Welcome to GitLab!</div>'
        '<div class="welcome-text">'
        "Ask me anything about how we work, our values, and our product direction. "
        "I'm here to help you navigate GitLab's Handbook and Direction pages. "
        "Try one of the suggested questions in the sidebar to get started!"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# Display chat history
for msg in st.session_state.ss_messages:
    role = msg["role"]
    content = msg["content"]
    metadata = msg.get("metadata", {})

    with st.chat_message(role, avatar="🦊" if role == "assistant" else "👤"):
        if role == "assistant":
            # Off-topic response
            if metadata.get("is_off_topic"):
                st.markdown(
                    f'<div class="offtopic-card">{content}</div>',
                    unsafe_allow_html=True,
                )
            # Error response
            elif metadata.get("error"):
                st.markdown(
                    f'<div class="error-card">{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Confidence indicator
                is_confident = metadata.get("is_confident", True)
                confidence_score = metadata.get("confidence_score", 0.0)
                category_label = metadata.get("category_label", "")

                if is_confident:
                    confidence_html = (
                        f'<span class="confidence-high">&#x25CF; High Confidence ({confidence_score:.0%})</span>'
                    )
                else:
                    confidence_html = (
                        f'<span class="confidence-low">&#x25CF; Low Confidence ({confidence_score:.0%})</span>'
                    )

                if category_label:
                    confidence_html += f'<span class="category-badge">{category_label}</span>'

                st.markdown(confidence_html, unsafe_allow_html=True)

                # Answer text
                st.markdown(content)

                # Source citations
                sources = metadata.get("sources", [])
                if sources:
                    st.markdown("---")
                    st.markdown("**Sources**")
                    source_links_html = ""
                    for src in sources:
                        title = src.get("section_title", "Source")
                        url = src.get("source_url", "#")
                        display_title = title if len(title) <= 60 else title[:57] + "..."
                        source_links_html += (
                            f'<a href="{url}" target="_blank" class="source-link">'
                            f"{display_title}</a>\n"
                        )
                    st.markdown(source_links_html, unsafe_allow_html=True)

                # Evidence panel (collapsible)
                chunks = metadata.get("chunks", [])
                if chunks:
                    with st.expander("View Sources & Evidence", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            sim_score = chunk.get("similarity", 0.0)
                            section = chunk.get("section_title", "Unknown")
                            url = chunk.get("source_url", "")
                            text_preview = chunk.get("chunk_text", "")[:300]

                            st.markdown(
                                f'<div class="evidence-chunk">'
                                f'<div class="evidence-chunk-title">Source {i}: {section}</div>'
                                f'<div class="evidence-chunk-score">'
                                f"Similarity: {sim_score:.4f} | "
                                f'<a href="{url}" target="_blank" style="color: #FC6D26;">{url}</a>'
                                f"</div>"
                                f'<div class="evidence-chunk-text">{text_preview}...</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
        else:
            st.markdown(content)


# ---------------------------------------------------------------------------
# Chat Input Handling
# ---------------------------------------------------------------------------
def process_query(query: str):
    """Process a user query through the RAG chain and update the UI."""
    query = query.strip()
    if not query:
        return

    # Add user message to history
    st.session_state.ss_messages.append({
        "role": "user",
        "content": query,
        "metadata": {},
    })

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Run RAG chain with spinner
    with st.chat_message("assistant", avatar="🦊"):
        with st.spinner("Searching GitLab's Handbook & Direction..."):
            # Build conversation history for context (last 3 turns)
            conv_history = []
            for msg in st.session_state.ss_messages[:-1]:  # Exclude current message
                conv_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            result = run_rag_chain(query=query, history=conv_history)

        # ── Off-topic ────────────────────────────────────────────────────
        if not result["is_on_topic"]:
            st.markdown(
                f'<div class="offtopic-card">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.ss_messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": {"is_off_topic": True},
            })
            return

        # ── Error ────────────────────────────────────────────────────────
        if result.get("error"):
            st.markdown(
                f'<div class="error-card">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.ss_messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": {"error": result["error"]},
            })
            return

        # ── Confidence indicator ─────────────────────────────────────────
        is_confident = result["is_confident"]
        confidence_score = result["confidence_score"]
        category_label = result["category_label"]

        if is_confident:
            confidence_html = (
                f'<span class="confidence-high">&#x25CF; High Confidence ({confidence_score:.0%})</span>'
            )
        else:
            confidence_html = (
                f'<span class="confidence-low">&#x25CF; Low Confidence ({confidence_score:.0%})</span>'
            )

        if category_label:
            confidence_html += f'<span class="category-badge">{category_label}</span>'

        st.markdown(confidence_html, unsafe_allow_html=True)

        # ── Answer ───────────────────────────────────────────────────────
        st.markdown(result["answer"])

        # Low confidence notice below the answer
        if not is_confident:
            st.markdown(
                '<div class="lowconf-notice">'
                "The retrieved content had a low similarity score for this query. "
                "The answer above may be incomplete — please verify at the source links below."
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Source citations ─────────────────────────────────────────────
        sources = result["sources"]
        if sources:
            st.markdown("---")
            st.markdown("**Sources**")
            source_links_html = ""
            for src in sources:
                title = src.get("section_title", "Source")
                url = src.get("source_url", "#")
                display_title = title if len(title) <= 60 else title[:57] + "..."
                source_links_html += (
                    f'<a href="{url}" target="_blank" class="source-link">'
                    f"{display_title}</a>\n"
                )
            st.markdown(source_links_html, unsafe_allow_html=True)

        # ── Evidence panel ───────────────────────────────────────────────
        chunks = result["chunks"]
        if chunks:
            with st.expander("View Sources & Evidence", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    sim_score = chunk.get("similarity", 0.0)
                    section = chunk.get("section_title", "Unknown")
                    url = chunk.get("source_url", "")
                    text_preview = chunk.get("chunk_text", "")[:300]

                    st.markdown(
                        f'<div class="evidence-chunk">'
                        f'<div class="evidence-chunk-title">Source {i}: {section}</div>'
                        f'<div class="evidence-chunk-score">'
                        f"Similarity: {sim_score:.4f} | "
                        f'<a href="{url}" target="_blank" style="color: #FC6D26;">{url}</a>'
                        f"</div>"
                        f'<div class="evidence-chunk-text">{text_preview}...</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Save to session state ────────────────────────────────────────
        st.session_state.ss_messages.append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": {
                "sources": sources,
                "chunks": chunks,
                "is_confident": is_confident,
                "confidence_score": confidence_score,
                "category_label": category_label,
                "is_off_topic": False,
            },
        })
        st.session_state.ss_last_sources = sources


# Handle pending question from sidebar button
if st.session_state.ss_pending_question:
    query = st.session_state.ss_pending_question
    st.session_state.ss_pending_question = None
    process_query(query)

# Chat input
user_input = st.chat_input("Ask about GitLab's Handbook or Direction pages...")
if user_input:
    process_query(user_input)
