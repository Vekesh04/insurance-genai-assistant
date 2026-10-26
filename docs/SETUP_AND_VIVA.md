# Enterprise Insurance Claims & Policy Intelligence Assistant

A local-first Gen-AI insurance project built in Python for structured claims data and unstructured policy documents.

## What this project demonstrates

- Python and pandas data ingestion
- SQLite relational database for customers, policies, claims, hospitals, claim documents and interactions
- Structured SQL tools
- Policy document chunking
- Vector embeddings with SentenceTransformers when available; TF-IDF vector fallback for an offline demo
- Persistent local vector index
- BM25 keyword retrieval
- Hybrid retrieval (vector + BM25)
- Metadata filtering by product
- Reranking using product/title relevance
- Retrieval-Augmented Generation (RAG)
- LangGraph routing between SQL, RAG and SQL+RAG analysis
- Tool calling through Python business tools
- Guardrails for prompt injection/destructive requests
- Hallucination handling by refusing unsupported answers
- OpenAI Responses API integration (optional)
- Ollama local-LLM integration (optional)
- FastAPI backend
- Streamlit chat UI
- Evaluation against supplied evaluation questions
- Optional MCP server for an advanced demonstration

## Architecture

```text
USER
  |
  v
Streamlit UI
  |
  v
FastAPI / direct Python call
  |
  v
LangGraph workflow
  |
  +-------------------+-------------------+
  |                   |                   |
  v                   v                   v
SQL route          RAG route          Combined route
  |                   |                   |
Python tools      Policy retriever     SQL tools + RAG
  |                   |                   |
SQLite          Vector + BM25          Evidence merge
  |                   |
  |             Metadata filter
  |                   |
  |               Reranking
  +-------------------+-------------------+
                      |
                      v
           Optional OpenAI / Ollama
                      |
                      v
             Grounded final answer
```

## Folder structure

```text
insurance_genai_python_project/
├── app.py                         # Streamlit chat UI
├── config.py                      # Paths and environment settings
├── setup_project.py               # Creates DB + retrieval index
├── run_demo.py                    # 3-command demo from terminal
├── requirements.txt
├── .env.example
├── api/
│   └── main.py                    # FastAPI endpoints
├── agent/
│   └── tool_registry.py           # Python tool registry
├── database/
│   ├── connection.py
│   └── create_database.py
├── graph/
│   ├── router.py                  # SQL/RAG/combined routing logic
│   └── workflow.py                # LangGraph state machine
├── guardrails/
│   └── validation.py
├── rag/
│   ├── document_loader.py         # policy parsing/chunking
│   ├── embeddings.py              # semantic + offline fallback
│   ├── bm25.py
│   ├── vector_store.py
│   ├── retriever.py               # hybrid retrieval/reranking
│   ├── rag_service.py
│   └── build_index.py
├── services/
│   └── llm_service.py             # OpenAI Responses / Ollama
├── tools/
│   ├── claims_tool.py
│   ├── customer_tool.py
│   ├── document_tool.py
│   ├── hospital_tool.py
│   ├── interaction_tool.py
│   └── policy_tool.py
├── evaluation/
│   └── evaluate.py
├── optional_mcp/
│   └── mcp_server.py
├── tests/
└── data/
    ├── customers.csv
    ├── policies.csv
    ├── claims.csv
    ├── claim_documents.csv
    ├── hospitals.csv
    ├── customer_interactions.csv
    ├── evaluation_questions.csv
    └── policy_documents/*.txt
```

# VS Code setup

## 1. Open the project

Extract the ZIP, then in VS Code choose **File -> Open Folder** and select `insurance_genai_python_project`.

Open **Terminal -> New Terminal**.

## 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

In VS Code, press `Cmd/Ctrl + Shift + P` -> **Python: Select Interpreter** -> select `.venv`.

## 3. Install packages

For the most reliable first run:

```bash
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn "fastapi[standard]" uvicorn streamlit langgraph openai python-dotenv pytest
```

Optional semantic embeddings for the final demo:

```bash
pip install sentence-transformers
```

Or simply install the supplied requirements file:

```bash
pip install -r requirements.txt
```

## 4. Environment file

Copy `.env.example` to `.env`.

macOS / Linux:

```bash
cp .env.example .env
```

Windows:

```powershell
copy .env.example .env
```

The application runs without an API key. If you have an OpenAI API key, add it to `.env`:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
```

If you do not want an external API:

```text
LLM_PROVIDER=none
```

## 5. Create the database and policy index

```bash
python setup_project.py
```

This loads the CSV files into `database/insurance.db`, creates indexes, parses the policy files, chunks them and builds the local retrieval index.

## 6. Quick terminal test

```bash
python run_demo.py
```

You should see three routes: `sql`, `rag`, and `combined`.

## 7. Run automated tests

```bash
pytest -q
```

## 8. Start FastAPI

```bash
uvicorn api.main:app --reload
```

Open:

- API: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

Test `POST /assistant/ask` with:

```json
{
  "question": "Analyze claim CLM300001 and tell me if documents are missing."
}
```

## 9. Start Streamlit

Keep FastAPI running if you want to show it separately. Open a second VS Code terminal, activate `.venv`, then run:

```bash
streamlit run app.py
```

The browser will normally open at http://localhost:8501.

# Demo questions for the lecturer

### SQL only

```text
What is the status of claim CLM300001?
```

Explain: the claim status is structured business data, so the router uses SQL rather than vector search.

### RAG only

```text
What is the maternity waiting period under Health Secure Plus?
```

Explain: waiting-period wording is in an unstructured policy document, so the router uses hybrid RAG.

### SQL + RAG

```text
Analyze claim CLM300001 and tell me if documents are missing.
```

Explain: claim status/documents come from SQL; product policy rules come from RAG; the graph combines them into a grounded summary.

# How to explain each concept in your viva

**Why SQLite?** The structured CSVs form a relational business model. Database facts should be looked up deterministically instead of asking an LLM to remember them.

**What is RAG?** The system retrieves relevant policy sections first and gives only that evidence to the answer generator. This reduces hallucination.

**What are embeddings?** Policy chunks are represented as vectors so semantically similar questions and text can be compared. The project supports SentenceTransformers and has an offline vector fallback.

**Why hybrid search?** Dense/vector retrieval is good for semantic similarity, while BM25 is good for exact insurance terms. The system combines both scores.

**What is metadata filtering?** If a product such as `Health Secure Plus` is known, retrieval can be restricted to that product's chunks.

**What is reranking?** After initial retrieval, results get extra relevance weight for exact product mentions and matching section titles.

**Why LangGraph?** It makes the workflow explicit: safety -> route -> SQL/RAG/combined -> answer. You can show that the LLM is not blindly called for every question.

**What are tools?** Claims, policies, documents, hospitals and customers each have Python lookup functions. The graph invokes only the relevant tools.

**Guardrails?** The project blocks prompt-injection/destructive patterns, limits question length, and refuses to invent an answer when evidence is absent.

**Hallucination handling?** The response is generated only from SQL rows and retrieved policy chunks. When evidence is missing, the assistant says it cannot answer reliably.

**Evaluation?** `evaluation/evaluate.py` runs the supplied evaluation questions and writes an evaluation report.

# Evaluation

```bash
python -m evaluation.evaluate
```

The report is saved at `evaluation/evaluation_report.json`.

# Optional MCP demonstration

MCP is deliberately isolated so it cannot break the main submission.

```bash
pip install mcp
python optional_mcp/mcp_server.py
```

It exposes claim summary, document-issue, and policy-search functions as MCP tools.

# Important submission notes

1. Do not claim the LLM decides claim approval. It only summarizes evidence.
2. Be able to explain why SQL is used for claim status and why RAG is used for policy wording.
3. Show the router result in the Streamlit UI during the demo.
4. Run `python run_demo.py`, `pytest -q`, and the evaluation before submission.
5. Keep your API key only in `.env`; never hard-code it in Python files.
