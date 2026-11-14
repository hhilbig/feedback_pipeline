# Paper Feedback Pipeline

An experimental tool that uses language models to comment on drafts of quantitative social science papers. It is a work in progress. I have not yet tested whether its comments are accurate or useful, so treat the output as a list of things to check, not as a review.

The pipeline reads a manuscript, asks several models to write critiques from different perspectives (theory, rival explanations, methods, research design, editing), checks each critique against the text of the paper, and combines the remaining points into a short report.

## Running it

1. Get an API key from [OpenRouter](https://openrouter.ai) (or OpenAI, which limits you to OpenAI models).
2. Create a file called `.env` in this folder with `OPENROUTER_API_KEY=sk-or-your-key-here`.
3. Double-click `run_app.command`, or run `pip install -r requirements.txt` and `streamlit run streamlit_app.py`.
4. Paste the paper text or upload a PDF, then click "Generate Feedback".

Each run calls paid APIs. The app shows a cost estimate before the run and the actual cost afterward.

The manuscript is sent to the model providers. By default, OpenRouter requests exclude hosts that may store or train on prompts. Do not upload papers you are not allowed to share with these providers.

## More detail

[docs/usage.md](docs/usage.md) describes the nine pipeline stages, the model settings, the command-line options, and the evaluation tools. [DESIGN.md](DESIGN.md) explains the design choices.
