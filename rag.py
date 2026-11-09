"""RAG çekirdeği: veri yükleme, parçalama, geri getirme ve yanıt üretimi.

Streamlit'ten bağımsızdır; böylece hem `app.py` hem testler hem de
`evaluation/evaluate.py` aynı kodu kullanır.

İki çalışma modu vardır:

* **gemini**: Google embedding + FAISS ile anlamsal arama, Gemini ile yanıt üretimi.
* **yerel**:  API anahtarı yoksa TF-IDF ile arama ve en ilgili SSS cevabından
  alıntı (LLM çağrısı yapılmaz). Uygulama anahtarsız da denenebilsin diye vardır.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Mental_Health_FAQ.csv"
INDEX_DIR = ROOT / ".faiss_index"

DEFAULT_CHAT_MODEL = "gemini-3.6-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_K = 4
LOCAL_MIN_SCORE = 0.12  # yerel modda bu benzerliğin altı "bilgi yok" sayılır

NO_ANSWER_TR = "Bu konuda bilgi havuzumda bir bilgi bulunmuyor."
NO_ANSWER_EN = "I don't have information on this in my knowledge base."

SYSTEM_PROMPT = """Sen ruh sağlığı konularında bilgilendirici yanıtlar veren, \
nazik ve destekleyici bir asistansın.

Kurallar:
1. YALNIZCA aşağıdaki "Bağlam" bölümündeki bilgileri kullan. Kendi bilgini ekleme.
2. Bağlam soruyu yanıtlamaya yetmiyorsa yalnızca şunu söyle: \
"{no_answer_tr}" (soru İngilizceyse: "{no_answer_en}").
3. Kullanıcının yazdığı dilde yanıt ver (Türkçe soruya Türkçe, İngilizce soruya İngilizce).
4. Tanı koyma, ilaç dozu önerme. Uygun olduğunda bir uzmana danışmayı öner.
5. Kullanılan kaynakları cümle sonunda [1], [2] gibi numaralarla belirt.

Bağlam:
{context}"""

# Kriz ifadeleri: eşleşirse yanıttan önce acil yardım bilgisi gösterilir.
CRISIS_PATTERNS = [
    r"kendimi öldür", r"ölmek istiyorum", r"yaşamak istemiyorum",
    r"canıma kıy", r"kendime zarar", r"hayatıma son",
    r"intihar", r"yaşamıma son", r"ölmeyi düşün",
    r"suicid", r"kill(ing)? myself", r"end(ing)? (my|it all|my own) ?life", r"ending it all",
    r"take my (own )?life", r"want(ed)? to die", r"better off dead",
    r"self[- ]?harm", r"hurt(ing)? myself", r"cutting myself", r"don'?t want to (live|be alive)",
]
_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Veri
# --------------------------------------------------------------------------- #
def load_faq(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """SSS veri setini okur ve temizler."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {path}")
    df = pd.read_csv(path)
    missing = {"Question_ID", "Questions", "Answers"} - set(df.columns)
    if missing:
        raise ValueError(f"Veri setinde eksik sütunlar var: {sorted(missing)}")
    df = df.dropna(subset=["Questions", "Answers"]).copy()
    df["Questions"] = df["Questions"].str.strip()
    df["Answers"] = df["Answers"].str.strip()
    return df.reset_index(drop=True)


def build_documents(
    df: pd.DataFrame, chunk_size: int = 1200, chunk_overlap: int = 150
) -> list[Document]:
    """Her SSS kaydını parçalara böler; her parçanın başına soruyu ekler.

    Soru metninin her parçada bulunması, uzun cevapların ortasındaki parçaların
    da doğru soruyla eşleşmesini sağlar.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs: list[Document] = []
    for row in df.itertuples(index=False):
        for i, chunk in enumerate(splitter.split_text(row.Answers)):
            docs.append(
                Document(
                    page_content=f"Question: {row.Questions}\nAnswer: {chunk}",
                    metadata={
                        "question_id": int(row.Question_ID),
                        "question": row.Questions,
                        "chunk": i,
                        "answer_chunk": chunk,
                    },
                )
            )
    return docs


# --------------------------------------------------------------------------- #
# Geri getirme
# --------------------------------------------------------------------------- #
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Document, float]]: ...


class TfidfRetriever:
    """API anahtarı gerektirmeyen, kelime tabanlı yedek arama."""

    def __init__(self, docs: Sequence[Document]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = list(docs)
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(d.page_content for d in self.docs)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        from sklearn.metrics.pairwise import linear_kernel

        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [(self.docs[i], float(scores[i])) for i in top]


class VectorStoreRetriever:
    """Embedding + FAISS ile anlamsal (kosinüs benzerliği) arama.

    Doküman vektörleri bir kez hesaplanıp `index_dir` altına kaydedilir; sonraki
    açılışlarda embedding API'si tekrar çağrılmaz. Veri seti değişirse indeks
    otomatik olarak yeniden oluşturulur.
    """

    def __init__(self, docs: Sequence[Document], embeddings, index_dir: Path | None = None):
        import faiss
        import numpy as np

        self.docs = list(docs)
        self.embeddings = embeddings
        fingerprint = hashlib.sha256(
            "\n".join(d.page_content for d in self.docs).encode()
        ).hexdigest()

        index_file = index_dir / "index.faiss" if index_dir else None
        meta_file = index_dir / "meta.json" if index_dir else None
        if index_file and index_file.exists() and meta_file.exists() and \
                json.loads(meta_file.read_text()).get("fingerprint") == fingerprint:
            self.index = faiss.read_index(str(index_file))
        else:
            vectors = np.asarray(
                embeddings.embed_documents([d.page_content for d in self.docs]), dtype="float32"
            )
            faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            if index_dir:
                index_dir.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(index_file))
                meta_file.write_text(json.dumps({"fingerprint": fingerprint, "size": len(self.docs)}))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        import faiss
        import numpy as np

        q = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(k, len(self.docs)))
        return [(self.docs[i], float(sc)) for sc, i in zip(scores[0], ids[0]) if i >= 0]


# --------------------------------------------------------------------------- #
# Yanıt
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    question_id: int
    question: str
    excerpt: str
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "yerel"
    crisis: bool = False


def is_crisis(text: str) -> bool:
    """Mesajda intihar veya kendine zarar verme ifadesi var mı?"""
    return bool(_CRISIS_RE.search(text))


def looks_turkish(text: str) -> bool:
    return bool(re.search(r"[çğışöüÇĞİŞÖÜ]", text)) or bool(
        re.search(r"\b(ne|nasıl|neden|mi|mı|mu|mü|bir|ve|için)\b", text.lower())
    )


def unique_sources(hits: Sequence[tuple[Document, float]]) -> list[Source]:
    """Aynı soruya ait parçaları birleştirir, en yüksek skoru tutar."""
    seen: dict[int, Source] = {}
    for doc, score in hits:
        qid = doc.metadata["question_id"]
        if qid not in seen:
            seen[qid] = Source(
                question_id=qid,
                question=doc.metadata["question"],
                excerpt=doc.metadata.get("answer_chunk", doc.page_content),
                score=score,
            )
    return list(seen.values())


def format_context(hits: Sequence[tuple[Document, float]]) -> str:
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, (doc, _) in enumerate(hits, 1))


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    ).partial(no_answer_tr=NO_ANSWER_TR, no_answer_en=NO_ANSWER_EN)


class RAGPipeline:
    """Soru → geri getirme → (LLM varsa) üretim → Answer."""

    def __init__(self, retriever: Retriever, llm=None, k: int = DEFAULT_K):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.chain = (build_prompt() | llm | StrOutputParser()) if llm is not None else None

    @property
    def mode(self) -> str:
        return "gemini" if self.llm is not None else "yerel"

    def ask(self, question: str) -> Answer:
        question = question.strip()
        crisis = is_crisis(question)
        # Kriz mesajlarında intihar SSS'inin mutlaka bulunması için sorguyu güçlendir.
        query = f"{question} (thoughts of suicide, what should I do)" if crisis else question
        hits = self.retriever.search(query, self.k)
        no_answer = NO_ANSWER_TR if looks_turkish(question) else NO_ANSWER_EN

        if self.chain is None:  # yerel mod: en iyi eşleşmeden alıntı
            if not hits or hits[0][1] < LOCAL_MIN_SCORE:
                return Answer(no_answer, [], self.mode, crisis)
            best_doc, _ = hits[0]
            sources = unique_sources([h for h in hits if h[1] >= LOCAL_MIN_SCORE])
            body = re.sub(r"\s*\n\s*", "\n\n", best_doc.metadata["answer_chunk"])
            text = f"**{best_doc.metadata['question']}**\n\n{body}"
            return Answer(text, sources, self.mode, crisis)

        text = self.chain.invoke({"context": format_context(hits), "question": question})
        refused = NO_ANSWER_TR in text or NO_ANSWER_EN in text
        return Answer(text, [] if refused else unique_sources(hits), self.mode, crisis)


# --------------------------------------------------------------------------- #
# Kurulum yardımcıları
# --------------------------------------------------------------------------- #
def build_pipeline(
    api_key: str | None = None,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    k: int = DEFAULT_K,
    data_path: Path | str = DATA_PATH,
    index_dir: Path | None = INDEX_DIR,
) -> RAGPipeline:
    """API anahtarı varsa Gemini modunu, yoksa yerel modu kurar."""
    docs = build_documents(load_faq(data_path))
    if not api_key:
        return RAGPipeline(TfidfRetriever(docs), llm=None, k=k)

    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=api_key)
    if index_dir is not None:
        index_dir = index_dir / embedding_model.replace("/", "_")
    retriever = VectorStoreRetriever(docs, embeddings, index_dir)
    llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=api_key)
    return RAGPipeline(retriever, llm=llm, k=k)
