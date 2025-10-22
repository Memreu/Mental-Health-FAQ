# Akbank GenAI Bootcamp: RAG Tabanlı Mental Sağlık Chatbotu

Bu proje, Akbank GenAI Bootcamp programı için geliştirilmiş, Retrieval Augmented Generation (RAG) mimarisini kullanan interaktif bir chatbot uygulamasıdır.

## Projenin Amacı

Bu projenin temel amacı, "Mental Health FAQ" (Ruh Sağlığı Sıkça Sorulan Sorular) veri setini bir bilgi havuzu olarak kullanarak, kullanıcıların ruh sağlığıyla ilgili sorularına bilgilendirici ve destekleyici yanıtlar sunan bir yapay zeka asistanı oluşturmaktır.

Chatbot, cevaplarını *sadece* sağlanan veri setinin kapsamıyla sınırlandıracak ve bilgi havuzunda bulunmayan konularda spekülasyon yapmayacak (halüsinasyon görmeyecek) şekilde tasarlanmıştır.

## Veri Seti Hakkında Bilgi

Projede, Kaggle üzerinde bulunan [Mental Health FAQ Dataset](https://www.kaggle.com/datasets/narendgeek/mental-health-faq-for-chatbot) veri seti kullanılmıştır. Bu veri seti, ruh sağlığı konularında (`Questions`) ve bu sorulara karşılık gelen açıklayıcı (`Answers`) metin çiftlerini içermektedir.

RAG mimarisinin "Retrieval" (Geri Getirme) adımı için, `Answers` sütunundaki her bir açıklayıcı metin, anlamsal arama yapılabilmesi amacıyla vektör veritabanına yüklenecek ayrı bir "doküman" olarak kabul edilmiştir.

## Çözüm Mimarisi ve Teknoloji Dökümü

Proje, RAG (Retrieval Augmented Generation) mimarisini temel almaktadır. Çözüm, Streamlit kullanılarak interaktif bir web arayüzü üzerinden sunulmuştur.

**Kullanılan Teknolojiler:**

* **RAG Pipeline Framework:** LangChain
* **Generation Model:** Google Gemini API (`gemini-pro`)
* **Embedding Model:** Google (`models/embedding-001`)
* **Vektör Veritabanı:** FAISS (Yüksek hızlı anlamsal arama için lokal bellek tabanlı)
* **Web Arayüzü:** Streamlit

**Mimari Akışı:**
Kullanıcının girdiği soru (Query), öncelikle `GoogleGenerativeAIEmbeddings` modeli ile bir vektör temsiline dönüştürülür. Bu vektör, `FAISS` veritabanında anlamsal olarak en yakın doküman parçalarını (ilgili cevap metinlerini) bulmak için kullanılır. Elde edilen bu dokümanlar "Bağlam" (Context) olarak, kullanıcının orijinal sorusuyla birlikte `ChatGoogleGenerativeAI` (Gemini Pro) modeline bir prompt şablonu aracılığıyla sunulur. Dil modeli, bu bağlama dayanarak nihai, tutarlı ve doğru yanıtı üretir.

## Elde Edilen Sonuçlar

Sonuç olarak, "Mental Health FAQ" veri seti ile beslenen, RAG tabanlı bir chatbot başarıyla geliştirilmiştir. Uygulama, kullanıcının sorduğu sorulara *sadece veri setindeki bağlamı* kullanarak yanıt vermektedir. Bilgi havuzunda bulunmayan sorular için "Bu konuda bilgim bulunmuyor." yanıtını vererek, modelin halüsinasyon görmesi (uydurma cevaplar vermesi) etkin bir şekilde engellenmiştir.

## Lokal Çalıştırma Rehberi

Projenin lokal makinenizde çalıştırılabilmesi için gereken adımlar:

1.  Bu projeyi klonlayın veya GitHub'dan ZIP olarak indirin.
2.  Proje ana dizininde bir Python sanal ortamı (virtual env) oluşturun:
    ```bash
    python -m venv venv
    ```
3.  Oluşturulan sanal ortamı aktive edin:
    * Windows: `.\venv\Scripts\activate`
    * Mac/Linux: `source venv/bin/activate`
4.  Gerekli tüm kütüphaneleri `requirements.txt` dosyasından kurun:
    ```bash
    pip install -r requirements.txt
    ```
5.  API anahtarınızı güvenli bir şekilde saklamak için proje ana dizininde `.streamlit` adında bir klasör oluşturun.
6.  Bu yeni klasörün içine `secrets.toml` adında bir dosya oluşturun ve Gemini API anahtarınızı aşağıdaki formatta ekleyin:
    ```toml
    GOOGLE_API_KEY = "SIZIN_API_ANAHTARINIZ_BURAYA"
    ```
7.  Uygulamayı başlatın:
    ```bash
    streamlit run app.py
    ```
    Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

## Web Arayüzü ve Kullanıcı Deneyimi

**Deploy Linki (Web Linki):**
Bu proje öncelikle lokalde çalıştırılmak üzere geliştirilmiştir. Projeyi kalıcı bir web linki üzerinden yayınlamak (deploy) için, bu GitHub reposu [Streamlit Community Cloud](https://streamlit.io/cloud) veya [Hugging Face Spaces](https://huggingface.co/spaces) gibi ücretsiz platformlara kolayca bağlanabilir.

**Arayüz Kullanımı:**
Uygulama başlatıldığında, kullanıcıyı temiz bir sohbet arayüzü karşılar. Kullanıcı, sayfanın altındaki metin giriş alanına ruh sağlığıyla ilgili bir soru yazıp gönderdiğinde, RAG pipeline'ı tetiklenir. Modelin ürettiği yanıt, sohbet ekranında asistan mesajı olarak görüntülenir.
