# Akbank GenAI Bootcamp: Mental Sağlık Chatbot Projesi

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilen, RAG (Retrieval Augmented Generation) tabanlı bir chatbot projesidir.

## Projenin Amacı

Bu projenin amacı, "Mental Health FAQ" veri setini kullanarak, ruh sağlığıyla ilgili sorulara bilgilendirici ve destekleyici yanıtlar veren RAG tabanlı bir chatbot oluşturmaktır. Chatbot, kullanıcı sorularına cevap üretirken sadece sağlanan veri setindeki bilgileri kullanacak şekilde tasarlanmıştır.

## Veri Seti Hakkında Bilgi

Projede [Mental Health FAQ Kaggle Dataseti](https://www.kaggle.com/datasets/narendgeek/mental-health-faq-for-chatbot) kullanılmıştır. Bu veri seti, ruh sağlığıyla ilgili sıkça sorulan soruları (`Questions`) ve bu soruların detaylı cevaplarını (`Answers`) içermektedir.

RAG mimarisi için, `Answers` sütunundaki her bir cevap metni, vektör veritabanına yüklenecek ayrı bir "doküman" olarak ele alınmıştır.

## Kullanılan Yöntemler ve Çözüm Mimarisi

Proje, RAG (Retrieval Augmented Generation) mimarisini temel almaktadır. Çözüm, Streamlit kullanılarak bir web arayüzü üzerinden sunulmuştur.

Kullanılan teknolojiler şunlardır:

* **Generation Model:** Google Gemini API (`gemini-pro`)
* **Embedding Model:** Google (`models/embedding-001`)
* **Vektör Veritabanı:** FAISS (lokal bellek tabanlı)
* **RAG Pipeline Framework:** LangChain
* **Web Arayüzü:** Streamlit

**RAG Akışı:**
Kullanıcının girdiği soru (`Query`), `GoogleGenerativeAIEmbeddings` modeli kullanılarak bir vektöre dönüştürülür. Bu vektör, `FAISS` veritabanında anlamsal olarak en yakın doküman parçalarını (`Answers` metinleri) bulmak için kullanılır. Bulunan bu dokümanlar "Bağlam" (Context) olarak, kullanıcının orijinal sorusuyla birlikte `ChatGoogleGenerativeAI` (Gemini Pro) modeline bir prompt şablonuyla gönderilir. Model, bu bağlama dayanarak nihai yanıtı üretir.

## Elde Edilen Sonuçlar

Sonuç olarak, "Mental Health FAQ" veri seti ile beslenen, RAG tabanlı bir chatbot başarıyla oluşturulmuştur. Uygulama, kullanıcının sorduğu sorulara *sadece veri setindeki bağlamı* kullanarak yanıt vermektedir. Bilgi havuzunda bulunmayan sorular için "Bu konuda bilgim bulunmuyor." yanıtını vererek, modelin halüsinasyon görmesi (uydurma cevaplar vermesi) engellenmiştir.

## Kodunuzun Çalışma Kılavuzu

Projenin lokalde çalıştırılabilmesi için gereken adımlar:

1.  Bu projeyi klonlayın veya indirin.
2.  Proje ana dizininde bir Python sanal ortamı (virtual env) oluşturun:
    ```bash
    python -m venv venv
    ```
3.  Sanal ortamı aktive edin:
    * Windows: `.\venv\Scripts\activate`
    * Mac/Linux: `source venv/bin/activate`
4.  `requirements.txt` dosyasındaki kütüphaneleri kurun:
    ```bash
    pip install -r requirements.txt
    ```
5.  Proje ana dizininde `.streamlit` adında bir klasör oluşturun.
6.  Bu klasörün içine `secrets.toml` adında bir dosya oluşturun ve Gemini API anahtarınızı ekleyin:
    ```toml
    GOOGLE_API_KEY = "SIZIN_API_ANAHTARINIZ_BURAYA"
    ```
7.  Uygulamayı çalıştırın:
    ```bash
    streamlit run app.py
    ```

## Web Arayüzü & Product Kılavuzu

**Deploy Linki (Web Linki):**
Bu proje öncelikle lokalde çalıştırılmak üzere geliştirilmiştir. Projeyi yayınlamak (deploy) ve kalıcı bir web linki almak için, bu GitHub reposu [Streamlit Community Cloud](https://streamlit.io/cloud) veya [Hugging Face Spaces](https://huggingface.co/spaces) gibi ücretsiz platformlara bağlanabilir.

**Ürün Kılavuzu:**
Uygulama (`streamlit run app.py` ile) başlatıldığında, tarayıcıda bir sohbet arayüzü açılır. Kullanıcı, alttaki metin kutusuna ruh sağlığıyla ilgili bir soru yazıp gönderdiğinde, RAG pipeline'ı tetiklenir ve modelin ürettiği yanıt ekranda gösterilir.
