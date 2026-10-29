# 📄 DocMind: AI Document Q&A (Groq + Streamlit)

Upload PDFs/notes, ask questions, get answers with citations grounded in your documents.

**Live demo:** <Streamlit link yahan daalein>

## Architecture
```
User -> Streamlit UI -> run_agent() generator -> Groq API (streaming + tool use)
                                              |- search_documents -> TF-IDF index (scikit-learn)
                                              |- list_documents
                                              '- usage tracker (session state)
```

## Features
- RAG with citations `[file #chunk]`
- LLM tool use (`search_documents`, `list_documents`)
- Streaming responses, conversation memory (last 10 messages)
- Usage tracking, error handling (auth, rate limit, network, malformed tool-call fallback)
- Responsive Streamlit UI

## Run locally
```
pip install -r requirements.txt
export GROQ_API_KEY=your_key
streamlit run app.py
```

## Tech choices
Groq for fast free-tier inference; TF-IDF retrieval to avoid embedding costs; Streamlit Community Cloud for free deployment.
