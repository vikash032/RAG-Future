# Fix SQLite for Chroma - YE LINES SABSE UPAR HONI CHAHIYE
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import requests
import uuid
import gc
import numpy as np
import pandas as pd

# Lazy / guarded import for torch so deployment doesn't crash if pip install is still running
try:
    import torch
except Exception as e:
    torch = None
    st.warning("PyTorch not available yet. Model features will be disabled until torch is installed.")
    # optionally: st.error(str(e))

import arxiv
import duckdb
import streamlit as st
import spacy
import chromadb
from datetime import datetime, timedelta
from unstructured.partition.pdf import partition_pdf
from tempfile import NamedTemporaryFile
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
from spacy.matcher import Matcher
import dlt
import psutil
import re

# ----------------------
# Configuration
# ----------------------
st.set_page_config(
    page_title="Next-Gen RAG System",
    page_icon="🧠",
    layout="wide"
)

# Set default tensor type
torch.set_default_tensor_type(torch.FloatTensor)

# ----------------------
# Memory Management
# ----------------------
def check_memory(threshold=85):
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > threshold:
        st.warning(f"High memory usage: {memory_usage}%")
        return False
    return True

def smart_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

# ----------------------
# Lazy Loading Components
# ----------------------
@st.cache_resource
def get_embedding_model():
    if not check_memory():
        return None
    return SentenceTransformer('BAAI/bge-base-en-v1.5')

@st.cache_resource
def get_reranker():
    if not check_memory():
        return None
    return FlagReranker('BAAI/bge-reranker-base', use_fp16=True)

@st.cache_resource
def get_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.warning("SpaCy model not found. Some features disabled.")
        return None

# Initialize UMLS ontology
UMLS_ONTOLOGY = {
    "cancer": ["neoplasm", "tumor", "malignancy", "carcinoma"],
    "treatment": ["therapy", "intervention", "medication", "procedure"],
    "drug": ["pharmaceutical", "compound", "agent", "medication"],
    "diagnosis": ["detection", "screening", "identification", "assessment"]
}

# ----------------------
# ChromaDB Initialization
# ----------------------
@st.cache_resource
def get_chroma_client():
    client = chromadb.PersistentClient(path="./nextgen_rag_db")
    try:
        collection = client.get_collection("knowledge_base")
    except:
        model = get_embedding_model()
        embedding_dimension = model.get_sentence_embedding_dimension()
        collection = client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine", "embedding_dimension": str(embedding_dimension)}
        )
    return client, collection

client, collection = get_chroma_client()

# ----------------------
# Core Functions
# ----------------------
def get_weather_data(latitude=40.71, longitude=-74.01):
    """Fetch real-time weather data"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&timezone=auto"
        response = requests.get(url)
        data = response.json()
        current = data.get("current", {})
        current_time = current.get("time", datetime.now().isoformat())
        
        return {
            "temperature": current.get("temperature_2m", "N/A"),
            "humidity": current.get("relative_humidity_2m", "N/A"),
            "precipitation": current.get("precipitation", "N/A"),
            "windspeed": current.get("wind_speed_10m", "N/A"),
            "time": current_time,
            "location": f"{latitude},{longitude}",
            "text": f"""Weather Report ({current_time.split('T')[0]}):
- Temperature: {current.get('temperature_2m', 'N/A')}°C
- Humidity: {current.get('relative_humidity_2m', 'N/A')}%
- Precipitation: {current.get('precipitation', 'N/A')}mm
- Wind: {current.get('wind_speed_10m', 'N/A')}km/h"""
        }
    except Exception as e:
        return {
            "text": "Weather data currently unavailable",
            "time": datetime.now().isoformat(),
            "location": f"{latitude},{longitude}"
        }

def fetch_arxiv_papers(query, max_results=3):
    """Fetch papers from Arxiv API"""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results)
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id.split('/')[-1]
            })
        return papers
    except:
        return []

def fetch_clinical_trials(condition="cancer", max_results=3):
    """Fetch clinical trials"""
    try:
        url = f"https://clinicaltrials.gov/api/v2/studies?query.cond={condition}&pageSize={max_results}"
        response = requests.get(url, timeout=10)
        data = response.json()
        trials = []
        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            trials.append({
                "id": protocol.get("identificationModule", {}).get("nctId", "N/A"),
                "title": protocol.get("identificationModule", {}).get("briefTitle", "Untitled"),
                "status": protocol.get("statusModule", {}).get("overallStatus", "Unknown"),
                "text": f"Clinical Trial: {protocol.get('identificationModule',{}).get('briefTitle','Untitled')}"
            })
        return trials
    except:
        return []

def process_pdf(uploaded_file, source_type="general"):
    """Process PDF with domain-aware chunking"""
    try:
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        elements = partition_pdf(
            filename=tmp_path,
            strategy="auto",
            chunking_strategy="by_title",
            max_characters=1000,
            combine_text_under_n_chars=300
        )
        
        os.unlink(tmp_path)
        chunks = []
        
        for element in elements:
            if hasattr(element, "text") and element.text.strip():
                # Medical domain chunking
                if source_type == "medical" and "medical" in uploaded_file.name.lower():
                    medical_chunks = re.split(r"\n(ABSTRACT|BACKGROUND|METHODS|RESULTS|CONCLUSIONS):", 
                                             element.text, flags=re.IGNORECASE)
                    for i in range(0, len(medical_chunks)-1, 2):
                        if i+1 < len(medical_chunks):
                            chunks.append({
                                "text": medical_chunks[i+1] + " " + medical_chunks[i+2],
                                "section": medical_chunks[i]
                            })
                else:
                    chunks.append({
                        "text": element.text,
                        "source": uploaded_file.name,
                        "page": element.metadata.page_number if element.metadata else 1
                    })
        return chunks
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return []

def generate_qa_pairs(text):
    """Generate Q&A pairs using pattern matching"""
    patterns = [
        (r'([A-Z][A-Za-z\s]+):\s*([^\n\.]+)', 
         lambda m: (f"What is {m.group(1).lower()}?", m.group(2))),
        (r'([A-Z]{2,})\s*\(([^)]+)\)', 
         lambda m: (f"What does {m.group(1)} stand for?", m.group(2)))
    ]
    
    qa_pairs = []
    for pattern, generator in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                question, answer = generator(match)
                if len(question) < 100 and len(answer) < 200:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "context": text[:100] + "..."
                    })
            except:
                continue
    return qa_pairs

def extract_entities(text):
    """Extract medical entities using spaCy"""
    nlp = get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def ingest_data(data, data_type="weather", metadata=None):
    """Embed and store data in vector DB"""
    if not data:
        return False
        
    try:
        texts = []
        metadatas = []
        
        if data_type == "weather":
            texts.append(data["text"])
            metadatas.append({"type": "weather", "time": data["time"], "location": data["location"]})
        elif data_type == "pdf":
            for chunk in data:
                texts.append(chunk["text"])
                meta = {
                    "type": "pdf",
                    "source": chunk.get("source", "unknown"),
                    "page": chunk.get("page", 0)
                }
                if "section" in chunk:
                    meta["section"] = chunk["section"]
                metadatas.append(meta)
        elif data_type == "arxiv":
            texts.append(data["summary"])
            metadatas.append({
                "type": "arxiv",
                "title": data["title"],
                "authors": ", ".join(data["authors"]),
                "published": data["published"]
            })
        elif data_type == "clinical":
            texts.append(data["text"])
            metadatas.append({
                "type": "clinical_trial",
                "status": data["status"],
                "id": data["id"]
            })
        elif data_type == "qa":
            for qa in data:
                texts.append(f"Q: {qa['question']}\nA: {qa['answer']}")
                metadatas.append({
                    "type": "qa_pair",
                    "context": qa.get("context", "")
                })
        
        embeddings = get_embedding_model().encode(texts).tolist()
        ids = [f"{data_type}_{uuid.uuid4().hex[:8]}" for _ in texts]
        
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return True
    except Exception as e:
        st.error(f"Ingestion error: {str(e)}")
        return False

def hybrid_retrieval(question, max_results=5):
    """Hybrid search with lexical + semantic + reranking"""
    # First-stage: Semantic search
    query_embedding = get_embedding_model().encode(question).tolist()
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results*3,
        include=["metadatas", "documents", "distances"]
    )
    
    # Second-stage: Lexical search (if enabled)
    lexical_results = {"documents": [], "metadatas": []}
    if "lexical_search" in st.session_state and st.session_state.lexical_search:
        try:
            lexical_results = collection.query(
                query_texts=[question],
                n_results=max_results,
                include=["metadatas", "documents"]
            )
        except:
            pass
    
    # Combine results
    all_docs = semantic_results["documents"][0] + lexical_results["documents"][0]
    all_metas = semantic_results["metadatas"][0] + lexical_results["metadatas"][0]
    
    if not all_docs:
        return {"documents": [], "metadatas": []}
    
    # Third-stage: Reranking
    reranker = get_reranker()
    pairs = [(question, doc) for doc in all_docs]
    rerank_scores = reranker.compute_score(pairs)
    
    # Sort by rerank score
    sorted_indices = np.argsort(rerank_scores)[::-1]
    return {
        "documents": [all_docs[i] for i in sorted_indices[:max_results]],
        "metadatas": [all_metas[i] for i in sorted_indices[:max_results]]
    }

def fine_tune_embeddings(dataset):
    """Fine-tune embeddings using SetFit"""
    try:
        df = pd.read_csv(dataset)
        if len(df.columns) < 2:
            return "CSV needs at least 2 columns"
            
        text_col = st.session_state.get("text_col", df.columns[0])
        label_col = st.session_state.get("label_col", df.columns[1])
        
        df = df.dropna(subset=[text_col, label_col])
        dataset = Dataset.from_pandas(df)
        
        model = SetFitModel.from_pretrained("BAAI/bge-base-en-v1.5")
        trainer = SetFitTrainer(
            model=model,
            train_dataset=dataset,
            column_mapping={"text": text_col, "label": label_col},
            num_iterations=10
        )
        trainer.train()
        
        # Update global embedding model
        global embedding_model
        embedding_model = model.model_body
        return "Embeddings fine-tuned successfully!"
    except Exception as e:
        return f"Training failed: {str(e)}"

# ----------------------
# Streamlit UI
# ----------------------
st.title("🚀 Next-Gen RAG System")
st.caption("Hybrid Retrieval | Knowledge Graph | Real-time Data")

# Initialize session states
if 'arxiv_papers' not in st.session_state:
    st.session_state.arxiv_papers = []
if 'clinical_trials' not in st.session_state:
    st.session_state.clinical_trials = []
if 'lexical_search' not in st.session_state:
    st.session_state.lexical_search = False

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.subheader("Data Sources")
    latitude = st.number_input("Latitude", value=40.71)
    longitude = st.number_input("Longitude", value=-74.01)
    refresh_rate = st.slider("Weather Refresh (min)", 1, 60, 15)
    
    st.subheader("Retrieval Options")
    st.session_state.lexical_search = st.checkbox("Enable Lexical Search", True)
    use_knowledge_graph = st.checkbox("Enable Knowledge Graph", True)
    
    st.subheader("Model Operations")
    dataset = st.file_uploader("Upload training data (CSV)", type="csv")
    if dataset and st.button("Fine-tune Embeddings"):
        with st.spinner("Training..."):
            result = fine_tune_embeddings(dataset)
            st.info(result)
    
    st.subheader("System Info")
    st.write(f"Documents in DB: {collection.count()}")
    st.write(f"Memory: {psutil.virtual_memory().percent}% used")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌤️ Live Weather", "📄 PDF Ingestion", "📚 Arxiv Research", 
    "🏥 Clinical Trials", "❓ Q&A Generator", "🔍 Knowledge Query"
])

# Tab 1: Live Weather
with tab1:
    st.header("Real-time Weather Data")
    if st.button("Update Weather", key="weather_refresh"):
        st.cache_data.clear()
    
    @st.cache_data(ttl=refresh_rate * 60)
    def update_weather():
        return get_weather_data(latitude, longitude)
    
    weather_data = update_weather()
    ingest_data(weather_data, "weather")
    
    if weather_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{weather_data.get('temperature', 'N/A')}°C")
        col2.metric("Humidity", f"{weather_data.get('humidity', 'N/A')}%")
        col3.metric("Precipitation", f"{weather_data.get('precipitation', 'N/A')}mm")
        col4.metric("Wind Speed", f"{weather_data.get('windspeed', 'N/A')} km/h")
        st.code(weather_data["text"])

# Tab 2: PDF Ingestion
with tab2:
    st.header("Document Processing")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    source_type = st.radio("Content Type", ["General", "Medical"], horizontal=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                chunks = process_pdf(uploaded_file, source_type.lower())
                if chunks:
                    if ingest_data(chunks, "pdf"):
                        st.success(f"✅ Processed {len(chunks)} chunks")
                        with st.expander("View chunks"):
                            for i, chunk in enumerate(chunks[:3]):
                                st.caption(f"Chunk {i+1}")
                                st.text(chunk["text"][:300] + "...")

# Tab 3: Arxiv Research
with tab3:
    st.header("Arxiv Research Pipeline")
    arxiv_query = st.text_input("Search Query", "large language models")
    if st.button("Fetch Papers"):
        st.session_state.arxiv_papers = fetch_arxiv_papers(arxiv_query, 3)
    
    if st.session_state.arxiv_papers:
        paper = st.selectbox("Select Paper", [p["title"] for p in st.session_state.arxiv_papers])
        selected_paper = next(p for p in st.session_state.arxiv_papers if p["title"] == paper)
        
        st.write(f"**Authors:** {', '.join(selected_paper['authors'])}")
        st.write(f"**Published:** {selected_paper['published']}")
        st.write(selected_paper["summary"])
        
        if st.button("Ingest Paper"):
            if ingest_data(selected_paper, "arxiv"):
                st.success("Paper ingested!")

# Tab 4: Clinical Trials
with tab4:
    st.header("Clinical Trials Pipeline")
    condition = st.text_input("Medical Condition", "cancer")
    if st.button("Fetch Trials"):
        st.session_state.clinical_trials = fetch_clinical_trials(condition, 3)
    
    if st.session_state.clinical_trials:
        for trial in st.session_state.clinical_trials:
            with st.expander(trial["title"]):
                st.write(f"**Status:** {trial['status']}")
                st.write(trial["text"])
                if st.button("Ingest Trial", key=trial["id"]):
                    if ingest_data(trial, "clinical"):
                        st.success("Trial ingested!")

# Tab 5: Q&A Generator
with tab5:
    st.header("Rule-based Q&A Generation")
    search_term = st.text_input("Search term for context", "cancer treatment")
    
    if st.button("Find Context"):
        results = hybrid_retrieval(search_term, 1)
        if results["documents"]:
            st.session_state.qa_context = results["documents"][0]
    
    if "qa_context" in st.session_state:
        st.subheader("Retrieved Context")
        st.write(st.session_state.qa_context[:500] + "...")
        
        if st.button("Generate Q&A Pairs"):
            qa_pairs = generate_qa_pairs(st.session_state.qa_context)
            if qa_pairs:
                st.success(f"Generated {len(qa_pairs)} Q&A pairs!")
                for qa in qa_pairs:
                    st.markdown(f"**Q:** {qa['question']}")
                    st.markdown(f"**A:** {qa['answer']}")
                    st.divider()
                
                if st.button("Ingest Q&A Pairs"):
                    if ingest_data(qa_pairs, "qa"):
                        st.success("Q&A pairs ingested!")

# Tab 6: Knowledge Query
with tab6:
    st.header("Hybrid Knowledge Query")
    question = st.text_input("Ask a question", "What's the current weather in New York?")
    max_results = st.slider("Max Results", 1, 5, 3)
    
    if st.button("Search"):
        with st.spinner("Searching with hybrid retrieval..."):
            results = hybrid_retrieval(question, max_results)
            
            if results["documents"]:
                for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
                    with st.expander(f"Result {i+1} ({meta.get('type', 'data')}", expanded=i==0):
                        # Knowledge Graph Enhancement
                        if use_knowledge_graph:
                            entities = extract_entities(doc)
                            if entities:
                                st.caption("**Related Entities:** " + 
                                          ", ".join([f"{e['text']} ({e['label']})" for e in entities]))
                        
                        st.write(doc)
                        
                        # Metadata display
                        meta_info = []
                        if meta.get("source"): meta_info.append(f"Source: {meta['source']}")
                        if meta.get("title"): meta_info.append(f"Title: {meta['title']}")
                        if meta.get("time"): meta_info.append(f"Time: {meta['time']}")
                        if meta_info: st.caption(" | ".join(meta_info))
            else:
                st.warning("No relevant results found")

# System Status
st.sidebar.divider()
st.sidebar.subheader("Implementation Roadmap")
st.sidebar.markdown("""
- ✅ **Phase 1:** Core RAG System
- ✅ **Phase 2:** Hybrid Retrieval
- ✅ **Phase 3:** Knowledge Graph
- 🚧 **Phase 4:** Multi-modal Support
- 🚧 **Phase 5:** Edge Optimization
""")
st.sidebar.progress(65, text="Production Ready")
