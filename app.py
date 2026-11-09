"""Streamlit arayüzü: RAG tabanlı ruh sağlığı SSS asistanı.

Çalıştırma:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from rag import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, Answer, build_pipeline, load_faq

st.set_page_config(page_title="Ruh Sağlığı SSS Asistanı", page_icon="🧠", layout="centered")

CRISIS_MESSAGE = (
    "💛 **Şu an kendini güvende hissetmiyorsan yalnız değilsin.** "
    "Türkiye'de acil durumlarda **112**'yi arayabilirsin. Başka bir ülkedeysen "
    "[findahelpline.com](https://findahelpline.com) üzerinden ücretsiz destek hatlarını bulabilirsin. "
    "Güvendiğin biriyle konuşmak da iyi bir ilk adım olabilir."
)
EXAMPLES_EN = [
    "Can people with mental illness recover?",
    "What's the difference between anxiety and stress?",
    "How can I manage grief?",
    "Someone I love has been diagnosed with depression. How can I help?",
]
EXAMPLES_TR = [
    "Ruhsal bir hastalıktan iyileşmek mümkün mü?",
    "Stres ile kaygı arasındaki fark nedir?",
    "Yas sürecini nasıl atlatabilirim?",
    "Arkadaşım depresyon tanısı aldı, ona nasıl yardım edebilirim?",
]


def get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  # secrets.toml yoksa Streamlit hata fırlatır
        value = None
    return value or os.environ.get(name)


@st.cache_resource(show_spinner="Bilgi havuzu hazırlanıyor...")
def get_pipeline(api_key: str | None, chat_model: str, embedding_model: str, k: int):
    return build_pipeline(api_key, chat_model=chat_model, embedding_model=embedding_model, k=k)


def render_answer(answer: Answer) -> None:
    if answer.crisis:
        st.warning(CRISIS_MESSAGE)
    st.markdown(answer.text)
    if answer.sources:
        with st.expander(f"📚 Kaynaklar ({len(answer.sources)})"):
            for i, src in enumerate(answer.sources, 1):
                st.markdown(f"**[{i}] {src.question}** · benzerlik {src.score:.2f}")
                excerpt = src.excerpt if len(src.excerpt) < 400 else src.excerpt[:400] + "…"
                st.caption(excerpt)


# --- Kenar çubuğu ------------------------------------------------------------
api_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
chat_model = get_secret("GEMINI_MODEL") or DEFAULT_CHAT_MODEL
embedding_model = get_secret("GEMINI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL

with st.sidebar:
    st.header("⚙️ Ayarlar")
    k = st.slider("Getirilecek parça sayısı (k)", 2, 8, 4)
    if api_key:
        st.success(f"Gemini modu\n\nLLM: `{chat_model}`\n\nEmbedding: `{embedding_model}`")
    else:
        st.info(
            "**Yerel mod** (API anahtarı yok)\n\n"
            "Cevaplar TF-IDF araması ile doğrudan SSS'ten alıntılanır. "
            "Gemini ile üretken yanıtlar ve Türkçe soru desteği için "
            "`.streamlit/secrets.toml` dosyasına `GOOGLE_API_KEY` ekle."
        )
    st.metric("Bilgi havuzu", f"{len(load_faq())} soru-cevap")
    if st.button("🗑️ Sohbeti temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Bu asistan bilgilendirme amaçlıdır; tanı koymaz ve profesyonel "
        "desteğin yerini tutmaz."
    )

# --- Ana ekran ----------------------------------------------------------------
st.title("🧠 Ruh Sağlığı SSS Asistanı")
st.caption("RAG · LangChain · FAISS · Gemini — yalnızca güvenilir SSS içeriğine dayanarak yanıt verir.")

try:
    pipeline = get_pipeline(api_key, chat_model, embedding_model, k)
except Exception as exc:  # ağ, anahtar veya model hatası
    st.error(f"Asistan başlatılamadı: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg["answer"])
        else:
            st.markdown(msg["content"])

question = st.chat_input("Ruh sağlığıyla ilgili bir soru sor...") or st.session_state.pop("pending", None)

if not st.session_state.messages and not question:
    st.markdown("**Örnek sorular:**")
    cols = st.columns(2)
    examples = EXAMPLES_TR if pipeline.mode == "gemini" else EXAMPLES_EN
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Kaynaklar taranıyor..."):
            try:
                answer = pipeline.ask(question)
            except Exception as exc:
                answer = Answer(f"⚠️ Bir hata oluştu: {exc}", mode=pipeline.mode)
        render_answer(answer)
    st.session_state.messages.append({"role": "assistant", "answer": answer})
