# Paper Feedback Pipeline

Get AI-powered feedback on your quantitative social science paper. Multiple specialized AI reviewers analyze your work and synthesize their insights into actionable feedback.

## Getting Started

### Step 1: Get an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to API Keys and create a new key
4. Copy the key (starts with `sk-...`)

### Step 2: Save Your API Key

Open the folder containing this tool and create a new file called `.env` (just `.env`, no other name).

Inside the file, paste this line with your actual key:
```
OPENAI_API_KEY=sk-your-key-here
```

Save the file.

### Step 3: Run the App

**Double-click `run_app.command`**

That's it! A browser window will open with the app.

- First time: It will take 1-2 minutes to set up (you'll see progress in the terminal)
- After that: It starts in a few seconds

### Step 4: Use the App

1. Paste your paper text OR upload a PDF
2. Adjust settings if you want (sidebar)
3. Click "Generate Feedback"
4. Wait a few minutes while the AI reviews your paper
5. Read your feedback!

---

## How It Works

The tool runs your paper through a simulated review panel:

1. **8 AI reviewers** read your paper (Theorists, Methodologists, Rival Researchers, Editors)
2. Each proposes their single most important critique
3. Proposals are **scored** for importance, specificity, and actionability
4. Top proposals go through a **revision round** to sharpen the feedback
5. A **meta-reviewer** synthesizes everything into a final report

Click "How it works" in the app for a detailed diagram.

---

## Settings

In the sidebar, you can adjust:

| Setting | What it does |
|---------|--------------|
| **Model** | Smarter models cost more but may give better feedback |
| **Number of Agents** | More agents = more diverse perspectives, but higher cost |
| **Top-K Proposals** | How many top insights to include in the final review |

**Cost note**: Each run costs money (paid to OpenAI). Start with default settings to test. The app shows a cost estimate after each run.

---

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure you created the `.env` file in the same folder as the app
- Make sure the file contains `OPENAI_API_KEY=sk-...` (no spaces around the `=`)

**App won't start / "command not found"**
- Make sure you have Python 3 installed
- On Mac: Open Terminal and run `python3 --version` to check

**PDF upload doesn't work**
- Some PDFs are scanned images, not text. Try pasting the text instead.

---

## Advanced: Command Line

If you prefer the terminal:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run with different input methods
python -m feedback_pipeline --clipboard      # reads from clipboard
python -m feedback_pipeline --pdf paper.pdf  # reads from PDF
python -m feedback_pipeline --file paper.txt # reads from text file

# Customize the run
python -m feedback_pipeline --agents 16 --model gpt-5.2 --top-k 10 --file paper.txt
```

See [DESIGN.md](DESIGN.md) for technical details on the pipeline architecture.
