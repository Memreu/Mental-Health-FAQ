# [cite_start]Akbank GenAI Bootcamp: Mental Sağlık Chatbot Projesi [cite: 1]

[cite_start]Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilen, RAG (Retrieval Augmented Generation) [cite: 2] tabanlı bir chatbot projesidir.

## [cite_start]Projenin Amacı [cite: 9]

[cite_start]Bu projenin amacı, "Mental Health FAQ" veri setini [cite: 10] [cite_start]kullanarak, ruh sağlığıyla ilgili sorulara bilgilendirici ve destekleyici yanıtlar veren RAG tabanlı bir chatbot [cite: 2] oluşturmaktır. Chatbot, kullanıcı sorularına cevap üretirken sadece sağlanan veri setindeki bilgileri kullanacak şekilde tasarlanmıştır.

## [cite_start]Veri Seti Hakkında Bilgi [cite: 10]

Projede [Mental Health FAQ Kaggle Dataseti](https://www.kaggle.com/datasets/narendgeek/mental-health-faq-for-chatbot) kullanılmıştır. Bu veri seti, ruh sağlığıyla ilgili sıkça sorulan soruları (`Questions`) ve bu soruların detaylı cevaplarını (`Answers`) içermektedir.

[cite_start]RAG mimarisi için, `Answers` sütunundaki her bir cevap metni, vektör veritabanına yüklenecek ayrı bir "doküman" olarak ele alınmıştır[cite: 17].

## [cite_start]Kullanılan Yöntemler ve Çözüm Mimarisi [cite: 11, 23, 30]

[cite_start]Proje, RAG (Retrieval Augmented Generation) [cite: 2] mimarisini temel almaktadır. [cite_start]Çözüm, Streamlit kullanılarak bir web arayüzü [cite: 25] üzerinden sunulmuştur.

[cite_start]Kullanılan teknolojiler [cite: 41] şunlardır:

* [cite_start]**Generation Model:** Google Gemini API (`gemini-pro`) [cite: 42]
* [cite_start]**Embedding Model:** Google (`models/embedding-001`) [cite: 43]
* [cite_start]**Vektör Veritabanı:** FAISS (lokal bellek tabanlı) [cite: 43]
* [cite_start]**RAG Pipeline Framework:** LangChain [cite: 44]
* **Web Arayüzü:** Streamlit

**RAG Akışı:**
Kullanıcının girdiği soru (`Query`), `GoogleGenerativeAIEmbeddings` modeli kullanılarak bir vektöre dönüştürülür. Bu vektör, `FAISS` veritabanında anlamsal olarak en yakın doküman parçalarını (`Answers` metinleri) bulmak için kullanılır. Bulunan bu dokümanlar "Bağlam" (Context) olarak, kullanıcının orijinal sorusuyla birlikte `ChatGoogleGenerativeAI` (Gemini Pro) modeline bir prompt şablonuyla gönderilir. Model, bu bağlama dayanarak nihai yanıtı üretir.

## [cite_start]Elde Edilen Sonuçlar [cite: 12]

[cite_start]Sonuç olarak, "Mental Health FAQ" veri seti [cite: 10] [cite_start]ile beslenen, RAG tabanlı bir chatbot [cite: 2] başarıyla oluşturulmuştur. Uygulama, kullanıcının sorduğu sorulara *sadece veri setindeki bağlamı* kullanarak yanıt vermektedir. Bilgi havuzunda bulunmayan sorular için "Bu konuda bilgim bulunmuyor." yanıtını vererek, modelin halüsinasyon görmesi (uydurma cevaplar vermesi) engellenmiştir.

## [cite_start]Kodunuzun Çalışma Kılavuzu [cite: 19, 21, 29]

[cite_start]Projenin lokalde çalıştırılabilmesi için gereken adımlar[cite: 20]:

1.  Bu projeyi klonlayın veya indirin.
2.  [cite_start]Proje ana dizininde bir Python sanal ortamı (virtual env) oluşturun[cite: 21]:
    ```bash
    python -m venv venv
    ```
3.  Sanal ortamı aktive edin:
    * Windows: `.\venv\Scripts\activate`
    * Mac/Linux: `source venv/bin/activate`
4.  [cite_start]`requirements.txt` dosyasındaki kütüphaneleri kurun[cite: 21]:
    ```bash
    pip install -r requirements.txt
    ```
5.  Proje ana dizininde `.streamlit` adında bir klasör oluşturun.
6.  Bu klasörün içine `secrets.toml` adında bir dosya oluşturun ve Gemini API anahtarınızı ekleyin:
    ```toml
    GOOGLE_API_KEY = "SIZIN_API_ANAHTARINIZ_BURAYA"
    ```
7.  Uygulamayı çalıştırın[cite: 21]:
    ```bash
    streamlit run app.py
    ```

## [cite_start]Web Arayüzü & Product Kılavuzu [cite: 25, 31]

**Deploy Linki (Web Linki):**
[cite_start]Bu proje öncelikle lokalde çalıştırılmak üzere geliştirilmiştir[cite: 14]. [cite_start]Projeyi yayınlamak (deploy) ve kalıcı bir web linki [cite: 13] almak için, bu GitHub reposu [Streamlit Community Cloud](https://streamlit.io/cloud) veya [Hugging Face Spaces](https://huggingface.co/spaces) gibi ücretsiz platformlara bağlanabilir.

[cite_start]**Ürün Kılavuzu[cite: 25]:**
Uygulama (`streamlit run app.py` ile) başlatıldığında, tarayıcıda bir sohbet arayüzü açılır. Kullanıcı, alttaki metin kutusuna ruh sağlığıyla ilgili bir soru yazıp gönderdiğinde, RAG pipeline'ı tetiklenir ve modelin ürettiği yanıt ekranda gösterilir.
