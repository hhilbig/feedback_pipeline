## feedback_llm (very lightweight prototype)

This repository contains a **minimal, async feedback pipeline** for quantitative social science papers.
It is intentionally small: one core module, a few functions, and no framework or heavy dependencies.

### What it does

- **Generation**: 8 independent workers each propose one high-impact piece of feedback for a paper.
- **Scoring**: Each proposal is scored on importance, specificity, actionability, and uniqueness.
- **Selection (Python only)**: Proposals are ranked and classified using simple, deterministic logic
  (no hidden logic inside prompts).
- **Meta-review**: All high-quality proposals feed into a short meta-review plus global priorities.

### Design principles

- **Lightweight by default**: keep the core small and inspectable.
- **Async, but simple**: a single async pipeline, plus a small synchronous `feedback(paper_text)` wrapper.
- **Deterministic control**: ranking, thresholds, and filtering all live in Python.
- **Easily extensible**: prompts and thresholds can be refined without changing the overall structure.

If you add new features, **avoid bloat**:

- Prefer small functions over new modules or frameworks.
- Avoid adding heavy dependencies unless absolutely necessary.
- Keep configuration minimal and close to the code.

### Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key (for example in your shell):

```bash
export OPENAI_API_KEY="sk-..."
```

3. Use the pipeline from Python:

```python
from feedback_pipeline import feedback

paper_text = "... your paper text ..."
meta_review = feedback(paper_text)
print(meta_review)
```

For more detailed outputs (proposals, scores, and selection metadata), import and call
`full_feedback_pipeline` from `feedback_pipeline.py`.

### CLI usage

You can also run the pipeline from the command line:

- **From a file:**

```bash
python -m feedback_pipeline --file path/to/paper.txt
```

- **From stdin:**

```bash
cat path/to/paper.txt | python -m feedback_pipeline
```

Both commands will print the meta-review to stdout.



