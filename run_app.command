#!/bin/bash
# Double-click this file to launch the Paper Feedback web app

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "First run: Setting up virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies if streamlit isn't available
if ! command -v streamlit &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
    echo "Setup complete!"
    echo ""
fi

streamlit run streamlit_app.py
