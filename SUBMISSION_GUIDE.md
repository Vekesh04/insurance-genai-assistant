# Submission / Demonstration Guide

## 1. 30-second project introduction

> My project is an Enterprise Insurance Claims and Policy Intelligence Assistant. The main design idea is that not every question is sent to an LLM. Structured operational facts such as a claim status are retrieved from SQLite, while policy wording such as waiting periods and exclusions is retrieved from policy documents using RAG. LangGraph routes the question to SQL, RAG, or a combined workflow, and the final response is grounded in retrieved evidence.

## 2. Best live demonstration order

1. Start Streamlit and show the system-status sidebar.
2. Ask: `What is the status of claim CLM300001?`
   - Point to route = SQL.
   - Expand structured database evidence.
3. Ask: `What is the maternity waiting period under Health Secure Plus?`
   - Point to route = RAG.
   - Expand policy evidence and show the policy section.
4. Ask: `Analyze claim CLM300001 and tell me if documents are missing.`
   - Point to route = combined.
   - Explain SQL provides claim/documents and RAG provides policy rules.
5. Open FastAPI `/docs` and show the same application is also available as an API.
6. If time allows, run the evaluation script.

## 3. Key files to open in VS Code during viva

- `graph/router.py` — routing decision
- `graph/workflow.py` — LangGraph nodes and conditional edges
- `tools/claims_tool.py` — structured SQL tool example
- `rag/retriever.py` — vector + BM25 hybrid retrieval and reranking
- `guardrails/validation.py` — guardrails
- `services/llm_service.py` — optional OpenAI/Ollama integration
- `app.py` — Streamlit UI
- `api/main.py` — FastAPI layer

## 4. Questions your lecturer may ask

### Why not use the LLM for everything?
Database values such as claim status are deterministic business facts. Querying SQL is more reliable, traceable, and cheaper than asking an LLM to infer them.

### Why do you need RAG?
Policy rules live in unstructured documents. RAG retrieves the most relevant policy sections and then generates an answer from that evidence.

### What is hallucination in this application?
It would be the model inventing coverage, claim status, or approval decisions that are not present in the database or policy evidence. The app avoids that by grounding answers and returning an insufficient-evidence response when retrieval fails.

### What is hybrid search?
The application combines vector similarity with BM25 lexical matching. Exact insurance terminology benefits from BM25, while semantic similarity benefits from vector retrieval.

### What is reranking?
After retrieval, chunks get relevance boosts when the product name or section-title terms match the question.

### Why LangGraph?
The workflow has explicit nodes and transitions. It makes routing, tool execution, and combined reasoning inspectable instead of hiding the whole process in one prompt.

### What happens if the API key is not available?
The core system still works. SQL is deterministic and RAG has an extractive grounded fallback. The LLM is an enhancement for natural-language synthesis, not a dependency for correctness.

### Is the assistant deciding whether a claim must be approved?
No. It summarizes available evidence and can identify document or policy issues. The final claim decision belongs to the insurer's claim process.

## 5. Final commands before class

```bash
source .venv/bin/activate
python setup_project.py
python run_demo.py
python -m pytest -q
python -m evaluation.evaluate
```

Then run:

```bash
uvicorn api.main:app --reload
```

In another terminal:

```bash
source .venv/bin/activate
streamlit run app.py
```

## 6. Personalize before submitting

Read the code and change small presentation details yourself: page title, introductory text, comments, or UI wording. Do not change the underlying data logic unless you understand the effect. During the viva, explain the design choices in your own words rather than memorizing this guide word-for-word.
