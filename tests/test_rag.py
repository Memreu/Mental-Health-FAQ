"""API anahtarı gerektirmeyen birim testleri (sahte embedding ve sahte LLM ile)."""
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag  # noqa: E402


@pytest.fixture(scope="module")
def docs():
    return rag.build_documents(rag.load_faq())


def test_veri_seti_yukleniyor():
    df = rag.load_faq()
    assert len(df) == 98
    assert df[["Questions", "Answers"]].notna().all().all()


def test_her_parca_soruyu_ve_kimligi_tasiyor(docs):
    assert len(docs) > 98  # uzun cevaplar birden fazla parçaya bölünür
    for d in docs:
        assert d.page_content.startswith("Question: ")
        assert {"question_id", "question", "answer_chunk"} <= d.metadata.keys()


def test_yerel_mod_dogru_sssi_buluyor():
    p = rag.build_pipeline(api_key=None)
    ans = p.ask("Can people with mental illness recover?")
    assert p.mode == "yerel"
    assert ans.sources[0].question == "Can people with mental illness recover?"


def test_yerel_mod_kapsam_disinda_bilgi_yok_diyor():
    ans = rag.build_pipeline(api_key=None).ask("What is the capital of France?")
    assert ans.text == rag.NO_ANSWER_EN and ans.sources == []
    ans_tr = rag.build_pipeline(api_key=None).ask("Bugün hava nasıl?")
    assert ans_tr.text == rag.NO_ANSWER_TR


@pytest.mark.parametrize("text", ["I want to kill myself", "Bazen intihar etmeyi düşünüyorum",
                                  "I have been thinking about ending my life",
                                  "kendime zarar vermek istiyorum"])
def test_kriz_ifadeleri_algilaniyor(text):
    assert rag.is_crisis(text)


@pytest.mark.parametrize("text", ["How can I manage grief?", "Stres ile kaygı arasındaki fark nedir?"])
def test_normal_sorular_kriz_sayilmiyor(text):
    assert not rag.is_crisis(text)


def test_kriz_mesajinda_intihar_sss_kaynak_olarak_geliyor():
    ans = rag.build_pipeline(api_key=None).ask("I want to kill myself")
    assert ans.crisis
    assert any("suicide" in s.question.lower() for s in ans.sources)


def test_llm_zinciri_baglam_ve_kaynaklarla_calisiyor(docs, tmp_path):
    retriever = rag.VectorStoreRetriever(docs, DeterministicFakeEmbedding(size=32), tmp_path)
    llm = FakeListChatModel(responses=["Evet, iyileşme mümkündür [1]."])
    ans = rag.RAGPipeline(retriever, llm=llm, k=3).ask("Iyilesme mumkun mu?")
    assert ans.mode == "gemini"
    assert ans.text.startswith("Evet")
    assert 1 <= len(ans.sources) <= 3
    assert (tmp_path / "index.faiss").exists()  # indeks diske kaydedildi


def test_llm_bilgi_yok_dediginde_kaynak_gosterilmiyor(docs):
    retriever = rag.TfidfRetriever(docs)
    llm = FakeListChatModel(responses=[rag.NO_ANSWER_TR])
    ans = rag.RAGPipeline(retriever, llm=llm).ask("Python'da liste nasıl sıralanır?")
    assert ans.sources == []


def test_prompt_baglami_ve_kurallari_iceriyor():
    msgs = rag.build_prompt().format_messages(context="[1] örnek", question="soru")
    assert "[1] örnek" in msgs[0].content and rag.NO_ANSWER_TR in msgs[0].content


class CountingEmbedding(DeterministicFakeEmbedding):
    calls: int = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def test_kayitli_indeks_tekrar_kullaniliyor(docs, tmp_path):
    first = CountingEmbedding(size=16)
    rag.VectorStoreRetriever(docs, first, tmp_path)
    second = CountingEmbedding(size=16)
    rag.VectorStoreRetriever(docs, second, tmp_path)
    assert first.calls == 1 and second.calls == 0  # ikinci açılışta API çağrısı yok
    third = CountingEmbedding(size=16)
    rag.VectorStoreRetriever(docs[:-1], third, tmp_path)  # veri değişti → yeniden oluştur
    assert third.calls == 1


def test_gemini_ayarlari_dogru_modelleri_kullaniyor():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=rag.DEFAULT_CHAT_MODEL, google_api_key="test")
    assert llm.temperature is None  # Gemini 3+ modellerinde sampling parametresi gönderilmez
