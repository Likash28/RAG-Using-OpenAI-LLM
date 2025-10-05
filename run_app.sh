#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run the app (will load .env file automatically)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
