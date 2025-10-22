import os
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- 1. Streamlit Arayüz Ayarları ---
st.set_page_config(page_title="Ruh Sağlığı Chatbotu", layout="wide")
st.title("Akbank GenAI Bootcamp - RAG Chatbot")
st.subheader("Mental Health FAQ Veri Seti ile LangChain ve Gemini Kullanımı")

# --- 2. API Anahtarı Yönetimi ---
# .streamlit/secrets.toml dosyasından anahtarı okur
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
except KeyError:
    st.error("GOOGLE_API_KEY bulunamadı! Lütfen .streamlit/secrets.toml dosyasına ekleyin.")
    st.stop()

# --- 3. RAG Pipeline Kurulumu (Cache'leme ile) ---
@st.cache_resource
def load_rag_pipeline():
    # Veriyi Yükle
    file_path = 'Mental_Health_FAQ.csv'
    if not os.path.exists(file_path):
        st.error(f"{file_path} dosyası proje klasöründe bulunamadı!")
        return None
        
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=['Answers'])
    
    # Dokümanları LangChain formatına çevir
    documents = [Document(page_content=text) for text in df_clean['Answers'].tolist()]
    
    # LangChain Bileşenlerini Kur
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # FAISS veritabanını oluştur
        db = FAISS.from_documents(documents, embeddings)
        
        # Retriever (Arayıcı)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # LLM (Gemini)
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

        # Prompt
        prompt_template = """
        Sen, ruh sağlığı konularında yardımcı olan bir asistansın.
        Sadece aşağıda verilen "Bağlam" (Context) içindeki bilgileri kullanarak kullanıcının "Soru"suna cevap ver.
        Eğer "Bağlam" içinde cevap yoksa, "Bu konuda bilgim bulunmuyor." de.
        
        Bağlam:
        {context}
        
        Soru:
        {question}
        
        Cevap:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # RAG Zinciri (Chain)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain

    except Exception as e:
        st.error(f"Pipeline kurulurken hata oluştu: {e}")
        return None

# Pipeline'ı yükle
pipeline = load_rag_pipeline()

# --- 4. Chat Arayüzü ---
if pipeline:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ruh sağlığıyla ilgili bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                try:
                    # Pipeline'ı çalıştır
                    response = pipeline.invoke(prompt)
                except Exception as e:
                    response = f"Bir hata oluştu: {e}"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})