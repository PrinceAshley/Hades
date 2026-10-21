import os
import io
import json
import base64
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title="Hades", page_icon="🤖")
st.title("Hello! I'm Hades")

# ---------- Save folder ----------
CHAT_DIR = Path(__file__).parent / "chats"
CHAT_DIR.mkdir(exist_ok=True)


def new_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def save_chat():
    """Write the current chat to disk. Called after every message."""
    if not st.session_state.messages:
        return
    msgs = [
        {
            "role": m["role"],
            "content": m["content"],
            "display": m.get("display", m["content"]),
            "files": [a["name"] for a in m.get("attachments", [])],
        }
        for m in st.session_state.messages
    ]
    first = next((m["display"] for m in msgs if m["role"] == "user"), "")
    data = {
        "id": st.session_state.chat_id,
        "title": first.strip()[:40] or "New chat",
        "messages": msgs,
    }
    path = CHAT_DIR / f"{st.session_state.chat_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chat(chat_id):
    data = json.loads((CHAT_DIR / f"{chat_id}.json").read_text(encoding="utf-8"))
    st.session_state.chat_id = chat_id
    st.session_state.messages = [
        {
            "role": m["role"],
            "content": m["content"],
            "display": m.get("display", m["content"]),
            "attachments": [{"kind": "doc", "name": n, "data": b""} for n in m.get("files", [])],
        }
        for m in data["messages"]
    ]


def delete_chat(chat_id):
    """Delete the chat file from the folder."""
    path = CHAT_DIR / f"{chat_id}.json"
    if path.exists():
        path.unlink()
    if st.session_state.chat_id == chat_id:
        st.session_state.chat_id = new_id()
        st.session_state.messages = []


def list_chats():
    items = []
    files = sorted(CHAT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            items.append((d["id"], d["title"]))
        except Exception:
            pass
    return items


# ---------- Session setup (reopens your last chat after closing the browser) ----------
if "chat_id" not in st.session_state:
    existing = list_chats()
    if existing:
        st.session_state.chat_id = existing[0][0]
        load_chat(existing[0][0])
    else:
        st.session_state.chat_id = new_id()
        st.session_state.messages = []

# ---------- Sidebar ----------
api_key = st.sidebar.text_input(
    "Groq API key",
    type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
)

CHAT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
model = st.sidebar.selectbox("Chat model", CHAT_MODELS)
vision_model = st.sidebar.text_input("Vision model (for images)", value="qwen/qwen3.8-27b")
stt_model = st.sidebar.selectbox("Speech-to-text model", ["whisper-large-v3-turbo", "whisper-large-v3"])

st.sidebar.divider()
if st.sidebar.button("➕ New chat", use_container_width=True):
    st.session_state.chat_id = new_id()
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("### Saved chats")
for cid, title in list_chats()[:30]:
    col1, col2 = st.sidebar.columns([5, 1])
    label = ("▶ " if cid == st.session_state.chat_id else "") + title
    if col1.button(label, key=f"load_{cid}", use_container_width=True):
        load_chat(cid)
        st.rerun()
    if col2.button("🗑", key=f"del_{cid}", help="Delete this chat"):
        delete_chat(cid)
        st.rerun()

if not api_key.strip():
    st.info("Enter your Groq API key in the sidebar to start.")
    st.stop()

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key.strip())

# ---------- File types ----------
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif"]
DOC_TYPES = ["pdf", "docx", "txt", "md", "csv", "json", "py"]
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "flac", "mpga"]
VIDEO_TYPES = ["mp4", "webm", "mpeg"]
MAX_DOC_CHARS = 30000


# ---------- Helpers ----------
def read_document(f):
    name = f.name.lower()
    data = f.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        text = data.decode("utf-8", errors="ignore")
    return text[:MAX_DOC_CHARS]


def transcribe(f):
    result = client.audio.transcriptions.create(
        model=stt_model, file=(f.name, f.getvalue())
    )
    return result.text


def data_url(att):
    b64 = base64.b64encode(att["data"]).decode()
    return f"data:{att['mime']};base64,{b64}"


def build_api_messages():
    out = []
    for m in st.session_state.messages:
        images = [a for a in m.get("attachments", []) if a["kind"] == "image" and a["data"]]
        if images:
            content = [{"type": "text", "text": m["content"]}]
            for a in images:
                content.append({"type": "image_url", "image_url": {"url": data_url(a)}})
            out.append({"role": m["role"], "content": content})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def has_images():
    return any(
        a["kind"] == "image" and a["data"]
        for m in st.session_state.messages
        for a in m.get("attachments", [])
    )


def show_message(m):
    with st.chat_message(m["role"]):
        for a in m.get("attachments", []):
            if not a["data"]:
                st.caption(f"📎 {a['name']}")
            elif a["kind"] == "image":
                st.image(a["data"], caption=a["name"], width=300)
            elif a["kind"] == "audio":
                st.audio(a["data"])
            elif a["kind"] == "video":
                st.video(a["data"])
            else:
                st.caption(f"📄 {a['name']}")
        st.markdown(m.get("display", m["content"]))


# ---------- Chat history ----------
for m in st.session_state.messages:
    show_message(m)

# ---------- Input ----------
prompt = st.chat_input(
    "Type a message or attach files...",
    accept_file="multiple",
    file_type=IMAGE_TYPES + DOC_TYPES + AUDIO_TYPES + VIDEO_TYPES,
)

if prompt:
    typed_text = (prompt.text or "").strip()
    files = prompt.files or []

    attachments, extra = [], []
    if files:
        with st.spinner("Reading attachments..."):
            for f in files:
                ext = f.name.rsplit(".", 1)[-1].lower()
                data = f.getvalue()
                try:
                    if ext in IMAGE_TYPES:
                        attachments.append({"kind": "image", "name": f.name, "data": data,
                                            "mime": f.type or f"image/{ext}"})
                    elif ext in DOC_TYPES:
                        extra.append(f"[Document: {f.name}]\n{read_document(f)}")
                        attachments.append({"kind": "doc", "name": f.name, "data": data})
                    elif ext in AUDIO_TYPES:
                        extra.append(f"[Audio transcript: {f.name}]\n{transcribe(f)}")
                        attachments.append({"kind": "audio", "name": f.name, "data": data})
                    elif ext in VIDEO_TYPES:
                        extra.append(f"[Video audio transcript: {f.name}]\n{transcribe(f)}")
                        attachments.append({"kind": "video", "name": f.name, "data": data})
                except Exception as e:
                    st.error(f"Could not read {f.name}: {e}")

    if not typed_text and attachments:
        typed_text = "Please analyze the attached file(s)."

    full_text = typed_text
    if extra:
        full_text += "\n\n" + "\n\n".join(extra)

    user_msg = {"role": "user", "content": full_text, "display": typed_text, "attachments": attachments}
    st.session_state.messages.append(user_msg)
    save_chat()  # saved immediately, before the AI even answers
    show_message(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        try:
            active_model = vision_model.strip() if has_images() else model
            stream = client.chat.completions.create(
                model=active_model,
                messages=build_api_messages(),
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    reply += chunk.choices[0].delta.content
                    placeholder.markdown(reply + "▌")
            placeholder.markdown(reply)
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # Runs even if the browser is closed mid-answer: keeps whatever was generated
            if reply:
                st.session_state.messages.append({"role": "assistant", "content": reply})
                save_chat()