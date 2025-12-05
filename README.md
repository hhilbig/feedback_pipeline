# feedback_llm

A minimal, async feedback pipeline for quantitative social science papers.

## How it works

**Generation**: 8 specialized AI workers (theorists, rival researchers, methodologists, editors) review the text and propose high-impact feedback.

**Scoring & Critique**: Proposals are scored for importance and actionability, then scrutinized by a "discussant" layer.

**Synthesis**: The highest-quality feedback is synthesized into a concise meta-review.

## Quick Start

### 1. Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Setup API Key (One-time)

Create a file named `.env` in this folder and paste your OpenAI API key inside:

```text
OPENAI_API_KEY=sk-123456789...
```

(The tool loads this automatically. You can also export the key in your terminal if you prefer.)

### 3. Run the Tool (Easiest Way)

1. Create a file named `paper.txt` in this folder.
2. Paste your paper text (e.g., from Overleaf) into it and save.
3. Run:

```bash
python -m feedback_pipeline
```

The tool will find `paper.txt`, generate feedback, and print the meta-review to your screen.

## Advanced / CLI Usage

If you prefer not to use `paper.txt`, you can use flags:

```bash
# Run on a specific file
python -m feedback_pipeline --file drafts/my_paper.txt

# Interactive Paste (paste text directly into terminal)
python -m feedback_pipeline --paste

# Estimate cost (add this flag to any command)
python -m feedback_pipeline --estimate-cost
```
