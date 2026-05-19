 # AI Automotive Decision Support Expert

A domain-specific AI chatbot for car buying decisions,
comparisons, maintenance advice, and troubleshooting —
powered by Llama 3.2 with a hybrid rule-based architecture.

---

## What It Does

Ask any car-related question in natural language and get
structured, expert-level responses instantly.

Handles 9 query categories:
- Car specifications and comparisons
- Engine and mechanical explanations
- Maintenance schedules and costs
- Electric vehicle guidance
- Buying recommendations by budget
- Troubleshooting and diagnostics
- Motorsport information

---

## How It Works

Two-layer hybrid architecture:

1. Rule-based classifier categorizes the query first
2. Llama 3.2 generates a domain-specific response
   based on the detected category

This approach produces more accurate, focused answers
than a plain LLM query.

---

## Tech Stack

Python · Streamlit · Ollama · Llama 3.2 · REST API

---

## Setup

1. Install Ollama from ollama.ai and pull the model:

```bash
ollama pull llama3.2
```

2. Install dependencies:

```bash
pip install streamlit requests
```

3. Run the app:

```bash
streamlit run app.py
```

---

## Architecture

Single-file Streamlit application.
Rule-based classifier → LLM response engine.
No database. No external APIs required after setup.

---

## Status

Completed · 2025
Built as AI coursework submission
