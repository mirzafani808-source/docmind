import os, json
import groq, gradio as gr
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY secret set nahi hai (Space Settings -> Secrets).")

client = Groq(api_key=API_KEY)

_ids = [m.id for m in client.models.list().data]
_prefs = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
MODEL = next((p for p in _prefs if p in _ids), None)
if MODEL is None:
    raise RuntimeError("Koi supported model nahi mila. Available: " + str(_ids))
print("Using model:", MODEL)
# USD per 1M tokens (sirf andaza; free tier mein asal kharcha 0). Groq pricing page se check karein:
PRICE_IN, PRICE_OUT = 0.59, 0.79

STORE = {"chunks": [], "vec": None, "mat": None}
USAGE = {"in": 0, "out": 0, "calls": 0}

def read_file(path):
    if path.lower().endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    return open(path, encoding="utf-8", errors="ignore").read()

def chunk_text(text, size=900, overlap=150):
    text = " ".join(text.split())
    return [text[i:i + size] for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]

def ingest(files):
    if not files:
        return "⚠️ Pehle koi PDF/TXT/MD file select karein."
    chunks = []
    try:
        for f in files:
            path = f if isinstance(f, str) else f.name
            name = os.path.basename(path)
            for i, c in enumerate(chunk_text(read_file(path))):
                chunks.append({"source": name, "id": i, "text": c})
        if not chunks:
            return "⚠️ Files mein text nahi mila (scanned PDF ho sakta hai)."
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        STORE.update(chunks=chunks, vec=vec, mat=vec.fit_transform([c["text"] for c in chunks]))
        n = len({c["source"] for c in chunks})
        return f"✅ {n} file(s) index ho gayin, {len(chunks)} chunks."
    except Exception as e:
        return f"❌ Ingestion error: {e}"

def search_documents(query, k=4):
    if not STORE["chunks"]:
        return {"error": "Koi document indexed nahi hai."}
    sims = cosine_similarity(STORE["vec"].transform([query]), STORE["mat"])[0]
    top = sims.argsort()[::-1][:int(k)]
    res = [{"source": STORE["chunks"][i]["source"], "chunk_id": STORE["chunks"][i]["id"],
            "score": round(float(sims[i]), 3), "text": STORE["chunks"][i]["text"]}
           for i in top if sims[i] > 0]
    return {"results": res} if res else {"results": [], "note": "Kuch relevant nahi mila."}

def list_documents():
    return {"documents": sorted({c["source"] for c in STORE["chunks"]}), "total_chunks": len(STORE["chunks"])}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Uploaded documents mein query se relevant passages dhoondta hai. Jawab dene se pehle hamesha use karein.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query"},
                                      "k": {"type": "integer", "description": "Kitne passages (default 4)"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "Indexed documents ki list deta hai.",
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
    cost = USAGE["in"] / 1e6 * PRICE_IN + USAGE["out"] / 1e6 * PRICE_OUT
    return (f"**📊 Usage**  \nCalls: {USAGE['calls']}  \nInput tokens: {USAGE['in']:,}  \n"
            f"Output tokens: {USAGE['out']:,}  \nEstimated cost: **${cost:.4f}**")

def respond(message, history):
    history = history or []
    if not message or not message.strip():
        yield history, usage_md(); return
    past = [{"role": m["role"], "content": m["content"]} for m in history
            if isinstance(m.get("content"), str) and m["role"] in ("user", "assistant")][-10:]   # memory
    while past and past[0]["role"] != "user":
        past.pop(0)
    msgs = [{"role": "system", "content": SYSTEM}] + past + [{"role": "user", "content": message}]
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⏳ Soch raha hoon..."}]
    yield history, usage_md()
    answer, use_tools = "", True
    try:
        for step in range(4):
            answer_before = answer
            content, calls = "", {}
            try:
                kwargs = dict(model=MODEL, messages=msgs, max_tokens=1024, temperature=0.2, stream=True)
                if use_tools and step < 3:   # aakhri round mein tools band, taake jawab zaroor likhe
                    kwargs.update(tools=TOOLS, tool_choice="auto")
                for chunk in client.chat.completions.create(**kwargs):
                    u = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None)
                    if u:
                        USAGE["in"] += u.prompt_tokens; USAGE["out"] += u.completion_tokens; USAGE["calls"] += 1
                    if not chunk.choices:
                        continue
                    d = chunk.choices[0].delta
                    if d.content:
                        content += d.content
                        answer = answer_before + content
                        history[-1]["content"] = answer
                        yield history, usage_md()
                    if d.tool_calls:
                        for tc in d.tool_calls:
                            c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            if tc.id: c["id"] = tc.id
                            if tc.function:
                                if tc.function.name: c["name"] = tc.function.name
                                if tc.function.arguments: c["args"] += tc.function.arguments
            except groq.BadRequestError:
                # Llama kabhi kabhi tool call galat format mein deta hai -> fallback: seedha retrieval
                if use_tools:
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
                history[-1]["content"] = answer
                yield history, usage_md()
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(run_tool(c["name"], args), ensure_ascii=False)})
    except groq.AuthenticationError:
        history[-1]["content"] = "❌ API key ghalat hai. Colab Secrets mein GROQ_API_KEY check karein."
    except groq.RateLimitError:
        history[-1]["content"] = "⏱️ Groq free tier ki rate limit lag gayi. Thori der baad try karein."
    except groq.APIConnectionError:
        history[-1]["content"] = "🌐 Network masla. Internet check karein."
    except groq.APIStatusError as e:
        history[-1]["content"] = f"❌ API error ({e.status_code}): {e.message}"
    except Exception as e:
        history[-1]["content"] = f"❌ Unexpected error: {e}"
    yield history, usage_md()

def reset():
    USAGE.update({"in": 0, "out": 0, "calls": 0})
    return [], usage_md()

with gr.Blocks(title="DocMind", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 DocMind\nApne documents se sawal poochein. Jawab **citations** ke saath milega.")
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            files = gr.File(label="PDF / TXT / MD upload karein", file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"])
            index_btn = gr.Button("📥 Index documents", variant="primary")
            status = gr.Markdown("Koi document indexed nahi.")
            usage = gr.Markdown(usage_md())
            clear_btn = gr.Button("🗑️ Chat clear")
        with gr.Column(scale=2, min_width=320):
            chat = gr.Chatbot(type="messages", height=480, label="Chat")
            msg = gr.Textbox(placeholder="Sawal likhein aur Enter dabayein...", show_label=False)
            gr.Examples(["Is document ka khulasa batayein", "Main points kya hain?"], inputs=msg)
    index_btn.click(ingest, files, status)
    msg.submit(respond, [msg, chat], [chat, usage]).then(lambda: "", None, msg)
    clear_btn.click(reset, None, [chat, usage])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))