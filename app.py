import os, json
import streamlit as st
import groq
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

# ---------- API key + model ----------
def get_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")

API_KEY = get_key()
if not API_KEY:
    st.error("GROQ_API_KEY set nahi hai. App Settings → Secrets mein add karein.")
    st.stop()

client = Groq(api_key=API_KEY)

@st.cache_resource(show_spinner=False)
def pick_model():
    ids = [m.id for m in client.models.list().data]
    prefs = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
    return next((p for p in prefs if p in ids), None)

try:
    MODEL = pick_model()
except Exception as e:
    st.error(f"Groq se connect nahi ho paya: {e}")
    st.stop()
if not MODEL:
    st.error("Koi supported Groq model nahi mila.")
    st.stop()

# USD per 1M tokens, sirf andaza (free tier mein asal kharcha 0)
PRICE_IN, PRICE_OUT = 0.59, 0.79

# ---------- session state ----------
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("store", {"chunks": [], "vec": None, "mat": None})
ss.setdefault("usage", {"in": 0, "out": 0, "calls": 0})
ss.setdefault("status", "Koi document indexed nahi.")

# ---------- ingestion + retrieval ----------
def read_file(f):
    if f.name.lower().endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
    return f.getvalue().decode("utf-8", errors="ignore")

def chunk_text(text, size=900, overlap=150):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]

def ingest(files):
    chunks = []
    try:
        for f in files:
            for i, c in enumerate(chunk_text(read_file(f))):
                chunks.append({"source": f.name, "id": i, "text": c})
        if not chunks:
            return "⚠️ Files mein text nahi mila (scanned PDF ho sakta hai)."
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        ss.store = {"chunks": chunks, "vec": vec, "mat": vec.fit_transform([c["text"] for c in chunks])}
        n = len({c["source"] for c in chunks})
        return f"✅ {n} file(s) index ho gayin, {len(chunks)} chunks."
    except Exception as e:
        return f"❌ Ingestion error: {e}"

def search_documents(query, k=4):
    st_ = ss.store
    if not st_["chunks"]:
        return {"error": "Koi document indexed nahi hai."}
    sims = cosine_similarity(st_["vec"].transform([query]), st_["mat"])[0]
    top = sims.argsort()[::-1][:int(k)]
    res = [{"source": st_["chunks"][i]["source"], "chunk_id": st_["chunks"][i]["id"],
            "score": round(float(sims[i]), 3), "text": st_["chunks"][i]["text"]}
           for i in top if sims[i] > 0]
    return {"results": res} if res else {"results": [], "note": "Kuch relevant nahi mila."}

def list_documents():
    return {"documents": sorted({c["source"] for c in ss.store["chunks"]}), "total_chunks": len(ss.store["chunks"])}

# ---------- agent ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Uploaded documents mein query se relevant passages dhoondta hai. Jawab dene se pehle use karein.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query"},
                                      "k": {"type": "integer", "description": "Kitne passages (default 4)"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_documents", "description": "Indexed documents ki list deta hai.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = """Aap DocMind hain, ek document Q&A assistant.
Rules:
1. search_documents sirf 1-2 baar use karein, phir foran jawab likhein. Baar baar search na karein.
2. SIRF tool se mile passages ke basis par jawab dein. Agar jawab nahi milta to saaf kahein: "Documents mein yeh maloomat nahi mili."
3. Har claim ke baad citation lagayein, jaise [file.pdf #3].
4. User ki zubaan mein jawab dein (Urdu/Roman Urdu/English). Jawab mukhtasar rakhein."""

def run_tool(name, args):
    if name == "search_documents":
        return search_documents(args.get("query", ""), min(int(args.get("k", 4)), 5))
    if name == "list_documents":
        return list_documents()
    return {"error": f"Unknown tool {name}"}

def usage_md():
    u = ss.usage
    cost = u["in"] / 1e6 * PRICE_IN + u["out"] / 1e6 * PRICE_OUT
    return (f"**📊 Usage**  \nCalls: {u['calls']}  \nInput tokens: {u['in']:,}  \n"
            f"Output tokens: {u['out']:,}  \nEstimated cost: **${cost:.4f}**")

def run_agent(message, past):
    past = [m for m in past if m["role"] in ("user", "assistant")][-10:]   # memory
    while past and past[0]["role"] != "user":
        past.pop(0)
    msgs = [{"role": "system", "content": SYSTEM}] + past + [{"role": "user", "content": message}]
    answer, use_tools = "", True
    try:
        for step in range(4):
            answer_before = answer
            content, calls = "", {}
            try:
                kwargs = dict(model=MODEL, messages=msgs, max_tokens=1024, temperature=0.2, stream=True)
                if use_tools and step < 3:
                    kwargs.update(tools=TOOLS, tool_choice="auto")
                for chunk in client.chat.completions.create(**kwargs):
                    u = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None)
                    if u:
                        ss.usage["in"] += u.prompt_tokens; ss.usage["out"] += u.completion_tokens; ss.usage["calls"] += 1
                    if not chunk.choices:
                        continue
                    d = chunk.choices[0].delta
                    if d.content:
                        content += d.content
                        answer = answer_before + content
                        yield answer
                    if d.tool_calls:
                        for tc in d.tool_calls:
                            c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            if tc.id: c["id"] = tc.id
                            if tc.function:
                                if tc.function.name: c["name"] = tc.function.name
                                if tc.function.arguments: c["args"] += tc.function.arguments
            except groq.BadRequestError:
                if use_tools:   # malformed tool call -> fallback: direct retrieval
                    use_tools, answer = False, answer_before
                    ctx = json.dumps(search_documents(message), ensure_ascii=False)
                    msgs.append({"role": "system", "content": "Tool available nahi. Sirf yeh context use karein:\n" + ctx})
                    continue
                raise
            if not calls:
                break
            msgs.append({"role": "assistant", "content": content or None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                                        for c in calls.values()]})
            for c in calls.values():
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                answer += f"\n\n🔎 _{c['name']}: {json.dumps(args, ensure_ascii=False)}_\n\n"
                yield answer
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(run_tool(c["name"], args), ensure_ascii=False)})
    except groq.AuthenticationError:
        yield "❌ API key ghalat hai. Secrets mein GROQ_API_KEY check karein."
    except groq.RateLimitError:
        yield "⏱️ Groq free tier ki rate limit lag gayi. Thori der baad try karein."
    except groq.APIConnectionError:
        yield "🌐 Network masla. Internet check karein."
    except groq.APIStatusError as e:
        yield f"❌ API error ({e.status_code}): {e.message}"
    except Exception as e:
        yield f"❌ Unexpected error: {e}"

# ---------- UI ----------
with st.sidebar:
    st.header("📄 DocMind")
    files = st.file_uploader("PDF / TXT / MD upload karein", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("📥 Index documents", type="primary", use_container_width=True):
        if not files:
            ss.status = "⚠️ Pehle koi file select karein."
        else:
            with st.spinner("Indexing..."):
                ss.status = ingest(files)
    st.markdown(ss.status)
    usage_box = st.empty()
    usage_box.markdown(usage_md())
    if st.button("🗑️ Chat clear", use_container_width=True):
        ss.messages = []
        ss.usage = {"in": 0, "out": 0, "calls": 0}
        st.rerun()
    st.caption(f"Model: {MODEL}")

st.title("📄 DocMind")
st.caption("Apne documents se sawal poochein. Jawab citations ke saath milega.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Sawal likhein aur Enter dabayein..."):
    past = list(ss.messages)
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        box = st.empty()
        box.markdown("⏳ Soch raha hoon...")
        final = ""
        for partial in run_agent(prompt, past):
            final = partial
            box.markdown(final)
        if not final.strip():
            final = "⚠️ Jawab nahi aaya. Dobara try karein."
            box.markdown(final)
    ss.messages.append({"role": "assistant", "content": final})
    usage_box.markdown(usage_md())
