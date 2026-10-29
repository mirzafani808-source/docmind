# 📄 DocMind: AI Document Q&A (Groq)

Upload PDFs/notes, ask questions, get answers with citations grounded in your documents.

**Live demo:** <Render link yahan daalein>

## Architecture
```
User -> Gradio UI -> respond() generator -> Groq API (gpt-oss-120b / Llama, streaming + tool use)
                                         |- search_documents -> TF-IDF index (scikit-learn)
                                         |- list_documents
                                         '- usage tracker
```

## Features
- RAG with citations `[file #chunk]`
- LLM tool use (`search_documents`, `list_documents`)
- Streaming responses, conversation memory (last 10 messages)
- Usage tracking, error handling (auth, rate limit, network, malformed tool calls fallback)
- Responsive Gradio UI

## Run locally
```
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python app.py
```

## Tech choices
Groq for fast free-tier inference; TF-IDF retrieval to avoid embedding costs; Gradio so frontend and backend deploy together.
