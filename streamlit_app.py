"""
Streamlit web interface for the feedback pipeline.

Run with: streamlit run streamlit_app.py
Or double-click: run_app.command (macOS) / run_app.bat (Windows)
"""

import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Paper Feedback Pipeline",
    page_icon="📝",
    layout="wide",
)

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment. Add it to your .env file.")
    st.stop()

st.title("📝 Paper Feedback Pipeline")
st.markdown("""
A single AI response is often hit-or-miss. This tool uses multiple specialized reviewers
(Theorists, Methodologists, Rivals) to catch different types of errors—from logical gaps to
statistical flaws. Their feedback is scored, critiqued, and refined before being synthesized
into a final report.
""")

# --- How it Works ---
with st.expander("How it works"):
    st.markdown("""
This pipeline mimics a rigorous academic review process using specialized AI agents.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR PAPER TEXT                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. GENERATION                                                      │
│     8 specialized agents review your paper in parallel:             │
│     • 3 Theorists (contribution, logic, assumptions)                │
│     • 2 Rivals (alternative explanations, rival hypotheses)         │
│     • 2 Methodologists (empirical design, statistical issues)       │
│     • 1 Editor (clarity, structure, organization)                   │
│     Each agent proposes ONE high-impact feedback item.              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. SCORING (Dual-Pass)                                             │
│     Each proposal is scored twice to remove positional bias:        │
│     • Pass 1: Paper shown first, then proposal                      │
│     • Pass 2: Proposal shown first, then paper                      │
│     Scores averaged across: Importance, Specificity,                │
│     Actionability, Uniqueness → Composite score                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. CRITIQUE & REVISION                                             │
│     Top proposals (composite ≥ 3.0) enter a "Discussant" loop:      │
│     • A Discussant Agent critiques each proposal                    │
│     • Original agents revise based on critique                      │
│     • Revised proposals are re-scored and merged with originals     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. SYNTHESIS                                                       │
│     A Meta-Reviewer synthesizes the top-K proposals into:           │
│     • Prioritized list of key revisions                             │
│     • Section-by-section guidance (contribution, logic,             │
│       interpretation, writing)                                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FINAL META-REVIEW                             │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this approach?**
- **Ensemble generation**: Multiple perspectives catch different issues
- **Dual-pass scoring**: Removes AI's tendency to prefer text shown first
- **Critique loop**: Refines proposals like human peer review
- **Structured output**: Actionable guidance, not just criticism
""")

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Model",
        options=["gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"],
        index=0,
        help="Stronger models cost more but may give better feedback.",
    )

    agents = st.select_slider(
        "Number of Agents",
        options=[8, 16, 24, 32],
        value=8,
        help="More agents = more diverse feedback, but higher cost.",
    )

    top_k = st.slider(
        "Top-K Proposals",
        min_value=3,
        max_value=15,
        value=5,
        help="Number of top proposals to include in the meta-review.",
    )

    st.divider()
    st.markdown("**Cost Warning**: Using many agents with premium models can be expensive. Start small to test.")

# --- Main: Input ---
st.header("1. Provide Your Paper")

input_method = st.radio(
    "Input method:",
    options=["Paste text", "Upload PDF"],
    horizontal=True,
)

paper_text = ""

if input_method == "Paste text":
    paper_text = st.text_area(
        "Paste your paper text here:",
        height=300,
        placeholder="Copy and paste your paper text from Overleaf, Word, or any editor...",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a PDF file:",
        type=["pdf"],
    )
    if uploaded_file is not None:
        try:
            import fitz  # pymupdf

            # Read PDF from uploaded bytes
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            paper_text = "\n".join(text_parts)

            if paper_text.strip():
                st.success(f"Extracted {len(paper_text):,} characters from PDF.")
                with st.expander("Preview extracted text"):
                    st.text(paper_text[:2000] + ("..." if len(paper_text) > 2000 else ""))
            else:
                st.error("Could not extract text from PDF. It may be scanned/image-based.")
        except ImportError:
            st.error("pymupdf is not installed. Run: pip install pymupdf")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

# --- Main: Run ---
st.header("2. Generate Feedback")

can_run = bool(paper_text.strip())

if not paper_text.strip():
    st.info("Provide paper text above to continue.")
else:
    # Show cost estimate before running
    from feedback_pipeline import estimate_cost_before_run

    estimate = estimate_cost_before_run(
        paper_text,
        num_agents=agents,
        gen_model=model,
        top_k=top_k,
    )
    estimated_cost = estimate["estimated_total_cost_usd"]

    st.info(f"**Estimated cost: ${estimated_cost:.2f}** (actual cost may vary)")

if st.button("Generate Feedback", type="primary", disabled=not can_run):
    from feedback_pipeline import full_feedback_pipeline, _format_cost_estimate

    # Progress display
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(step: int, total: int, message: str):
        progress_bar.progress(step / total)
        status_text.markdown(f"**Step {step} of {total}:** {message}")

    try:
        result = asyncio.run(
            full_feedback_pipeline(
                paper_text,
                num_agents=agents,
                gen_model=model,
                top_k=top_k,
                progress_callback=update_progress,
            )
        )

        progress_bar.progress(1.0)
        status_text.empty()
        st.success("Feedback generated!")

        # Display meta-review
        st.header("3. Results")
        st.markdown(result["meta_review"])

        # Display cost estimate
        cost = result.get("cost_estimate")
        if cost:
            with st.expander("Cost Estimate"):
                st.text(_format_cost_estimate(cost))

    except ValueError as e:
        st.error(f"Configuration Error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
