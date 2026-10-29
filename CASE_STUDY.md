DocMind: AI Document Q&A Assistant

Author: Farhan Ali
Live app: <https://docmind-mcn53vqzj7mxn6e6g5sipd.streamlit.app/>
GitHub: https://github.com/mirzafani808-source/docmind

Problem

Long PDFs and guides are slow to search, and general chatbots often answer from memory instead of the document, which leads to hallucinations. I wanted an assistant that answers only from the user’s own files and shows exactly where each answer came from.

What I built

DocMind lets a user upload PDF, TXT or MD files, ask questions, and get streamed answers with citations like [file.pdf #3]. Features include RAG with tool use, streaming responses, conversation memory (last 10 messages), usage and cost tracking, and error handling for authentication, rate-limit and network failures.

Tech and prompt choices
Groq (gpt-oss-120b): fast, free-tier inference with tool calling. The app picks an available model at startup.
Tool use (search_documents, list_documents): the model decides when to retrieve.
TF-IDF retrieval (scikit-learn): no extra API and no embedding cost, good enough for small documents.
System prompt rules: answer only from retrieved passages, otherwise say “Documents mein yeh maloomat nahi mili”, and always cite sources.
Streamlit Community Cloud: free deployment with secrets management for the API key.
Challenge and how I solved it

The model searched repeatedly and my tool loop ended before it wrote an answer. I fixed this in three ways: the prompt now says to search only 1–2 times, tools are disabled on the final round so the model must answer, and results per search are capped at 5. I also added a fallback for malformed tool calls that runs retrieval directly.

Separately, Groq’s llama-3.3-70b returned model_not_found, so I used models.list() to select an available model automatically. Hugging Face Spaces and Render both required a paid plan or a card for Python apps, so I moved the deployment to Streamlit Cloud.

Limitations and next steps

TF-IDF tends to favour the first chunk on broad questions like “summarise this document”. Next I would add semantic embeddings (pgvector), OCR for scanned PDFs, and multi-user sessions.
