import streamlit as st
import os
import zipfile
import urllib.request
import re
import shutil
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai

st.set_page_config(
    page_title="arXiv Research Assistant RAG",
    page_icon="🔬",
    layout="wide"
)

# ==============================================================================
# 1. DESCARGA AUTOMÁTICA Y CORRECCIÓN DE RUTA DE LA BASE DE DATOS
# ==============================================================================

# ID de tu archivo "chroma_db.zip" en Google Drive
DRIVE_INPUT = "1u24Q94CCrSGMdoqvWCIqTTT3AlWzG6kG" 

def get_clean_drive_id(input_string):
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', input_string)
    if match:
        return match.group(1)
    match_folder = re.search(r'/folders/([a-zA-Z0-9-_]+)', input_string)
    if match_folder:
        return match_folder.group(1)
    return input_string.strip()

DRIVE_FILE_ID = get_clean_drive_id(DRIVE_INPUT)
DB_DIR_PATH = "./chroma_db"

def download_db_from_drive():
    """Descarga la base de datos de 716MB, la descomprime y asegura la ruta de ChromaDB."""
    sqlite_file = os.path.join(DB_DIR_PATH, "chroma.sqlite3")
    
    # Si la carpeta existe pero no tiene el archivo sqlite, está corrupta. La borramos.
    if os.path.exists(DB_DIR_PATH) and not os.path.exists(sqlite_file):
        shutil.rmtree(DB_DIR_PATH)
        
    if not os.path.exists(DB_DIR_PATH):
        with st.spinner("📦 Descargando la base de datos vectorial desde Google Drive (esto puede tardar un minuto la primera vez)..."):
            url = f"https://docs.google.com/uc?export=download&id={DRIVE_FILE_ID}"
            temp_zip = "downloaded_db.zip"
            try:
                # Descargar el zip con un nombre temporal seguro
                urllib.request.urlretrieve(url, temp_zip)
                
                # Crear carpeta destino temporal para extraer
                temp_extract_dir = "./temp_extract"
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                os.makedirs(temp_extract_dir)
                
                # Descomprimir
                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                # Buscar dinámicamente el archivo chroma.sqlite3 en todo lo descomprimido
                sqlite_found_path = None
                for root, dirs, files in os.walk(temp_extract_dir):
                    if "chroma.sqlite3" in files:
                        sqlite_found_path = root
                        break
                
                if sqlite_found_path:
                    # Mover la carpeta que contiene el sqlite a './chroma_db'
                    shutil.move(sqlite_found_path, DB_DIR_PATH)
                    st.success("¡Base de datos cargada e inicializada exitosamente!")
                else:
                    st.error("No se encontró el archivo 'chroma.sqlite3' dentro de la descarga de Google Drive.")
                
                # Limpiar temporales
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                    
            except Exception as e:
                st.error(f"Error al descargar o procesar la base de datos: {e}")
                st.info("Asegúrate de que el archivo 'chroma_db.zip' en tu Drive sea público.")

# Ejecutar la descarga
download_db_from_drive()

# ==============================================================================
# CARGA DE RECURSOS (Modelos y Base de Datos Vectorial)
# ==============================================================================
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("ms-marco-MiniLM-L-6-v2")
    
    chroma_client = chromadb.PersistentClient(path=DB_DIR_PATH)
    collection = chroma_client.get_collection(name="arxiv_papers")
    
    return embedding_model, reranker, collection

try:
    embedding_model, reranker, collection = load_resources()
except Exception as e:
    st.error(f"Error al inicializar la base de datos: {e}")

# ==============================================================================
# 2. CLIENTE DE GEMINI Y GENERACIÓN RAG
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def generate_rag_response(query, context_documents):
    if not GEMINI_API_KEY:
        return "❌ Error: La variable de entorno GEMINI_API_KEY no está configurada en los Secrets de la plataforma."
    
    context_text = ""
    for i, (doc, meta, score) in enumerate(context_documents):
        context_text += f"--- Document Evidence #{i+1} (Re-rank Score: {score:.4f}) ---\n"
        context_text += f"Title: {meta.get('title')}\nAbstract: {meta.get('abstract')}\n\n"
        
    prompt = f"""
You are an expert scientific research assistant. Answer the user's query using strictly the provided document evidences from arXiv papers. 
If the evidence does not contain enough information to answer the query, clearly state that the corpus does not contain sufficient information.

Document Evidences:
{context_text}

User Query: {query}

Answer:
"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Error al conectar con la API de Gemini: {e}"

def retrieve_and_rerank_local(query, top_n=8, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()
    initial_results = collection.query(query_embeddings=query_embedding, n_results=top_n)
    
    fetched_docs = initial_results['documents'][0]
    fetched_metadatas = initial_results['metadatas'][0]
    
    if not fetched_docs:
        return []

    pairs = [[query, doc] for doc in fetched_docs]
    rerank_scores = reranker.predict(pairs)
    scored_docs = list(zip(fetched_docs, fetched_metadatas, rerank_scores))
    scored_docs.sort(key=lambda x: x[2], reverse=True)
    
    return scored_docs[:top_k]

# ==============================================================================
# 3. INTERFAZ GRÁFICA DE USUARIO (UI) - Estilo Chat
# ==============================================================================

st.title("🔬 Asistente de Investigación Científica - arXiv RAG")
st.markdown(
    """
    Bienvenido al sistema de consulta inteligente de papers científicos. 
    Este asistente utiliza un pipeline **RAG de dos etapas** (Retriever + Re-ranker) 
    y es alimentado por **Gemini 2.5** para garantizar respuestas precisas basadas en evidencias reales.
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "evidences" in message:
            with st.expander("🔍 Ver Evidencias Utilizadas para esta respuesta"):
                for idx, (doc, meta, score) in enumerate(message["evidences"]):
                    st.markdown(f"**Evidencia #{idx+1} (Score: {score:.4f})**")
                    st.markdown(f"**Título:** {meta.get('title')}")
                    st.markdown(f"**Abstract:** {meta.get('abstract')}")
                    st.markdown("---")

if user_query := st.chat_input("Escribe tu consulta sobre IA, Robótica, etc. (Ej: How is reinforcement learning used in robotics?)"):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Buscando en arXiv y analizando evidencias..."):
            
            evidences = retrieve_and_rerank_local(user_query)
            response_text = generate_rag_response(user_query, evidences)
            st.write(response_text)
            
            if evidences:
                with st.expander("🔍 Ver Evidencias Utilizadas para esta respuesta"):
                    for idx, (doc, meta, score) in enumerate(evidences):
                        st.markdown(f"**Evidencia #{idx+1} (Score: {score:.4f})**")
                        st.markdown(f"**Título:** {meta.get('title')}")
                        st.markdown(f"**Abstract:** {meta.get('abstract')}")
                        st.markdown("---")
                        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "evidences": evidences
        })