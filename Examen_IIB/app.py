import streamlit as st
import os
import zipfile
import urllib.request
import re
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai

st.set_page_config(
    page_title="arXiv Research Assistant RAG",
    page_icon="🔬",
    layout="wide"
)

# ==============================================================================
# 1. DESCARGA AUTOMÁTICA DE LA BASE DE DATOS DESDE GOOGLE DRIVE
# ==============================================================================

# URL o ID que proporcionaste de tu recurso en Drive
DRIVE_INPUT = "1u24Q94CCrSGMdoqvWCIqTTT3AlWzG6kG" 

# Extrae el ID limpio usando una expresión regular por si se pasa una URL completa
def get_clean_drive_id(input_string):
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', input_string)
    if match:
        return match.group(1)
    match_folder = re.search(r'/folders/([a-zA-Z0-9-_]+)', input_string)
    if match_folder:
        return match_folder.group(1)
    return input_string.strip()

DRIVE_FILE_ID = get_clean_drive_id(DRIVE_INPUT)
DB_ZIP_PATH = "chroma_db.zip"
DB_DIR_PATH = "./chroma_db"

def download_db_from_drive():
    """Descarga la base de datos de 716MB desde Google Drive si no existe."""
    if not os.path.exists(DB_DIR_PATH):
        with st.spinner("📦 Descargando la base de datos vectorial desde Google Drive (esto puede tardar un minuto la primera vez)..."):
            # URL de descarga directa para Google Drive
            url = f"https://docs.google.com/uc?export=download&id={DRIVE_FILE_ID}"
            try:
                # Descargar el zip
                urllib.request.urlretrieve(url, DB_ZIP_PATH)
                
                # Descomprimirlo
                with zipfile.ZipFile(DB_ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(".")
                
                # Borrar el archivo .zip descargado para ahorrar espacio en el servidor
                os.remove(DB_ZIP_PATH)
                st.success("¡Base de datos cargada exitosamente!")
            except Exception as e:
                st.error(f"Error al descargar la base de datos de Google Drive: {e}")
                st.info("Asegúrate de que subiste el archivo 'chroma_db.zip' individual a tu Drive, que sea público y que el ID sea el correcto.")

# Ejecutar descarga antes de cargar recursos
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
    
    # Concatenar los contextos seleccionados por el Re-ranker
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

# Función auxiliar para recuperar y re-rankear
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

# Inicializar el historial del chat en la sesión de Streamlit si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial de mensajes de forma dinámica sin reiniciar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Si tiene evidencias asociadas en el estado, las mostramos abajo
        if "evidences" in message:
            with st.expander("🔍 Ver Evidencias Utilizadas para esta respuesta"):
                for idx, (doc, meta, score) in enumerate(message["evidences"]):
                    st.markdown(f"**Evidencia #{idx+1} (Score: {score:.4f})**")
                    st.markdown(f"**Título:** {meta.get('title')}")
                    st.markdown(f"**Abstract:** {meta.get('abstract')}")
                    st.markdown("---")

# Capturar la entrada de texto del usuario
if user_query := st.chat_input("Escribe tu consulta sobre IA, Robótica, etc. (Ej: How is reinforcement learning used in robotics?)"):
    
    # 1. Mostrar la pregunta en la pantalla de inmediato
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # 2. Generar la respuesta del backend
    with st.chat_message("assistant"):
        with st.spinner("Buscando en arXiv y analizando evidencias..."):
            
            # Etapa de Recuperación y Re-ranking
            evidences = retrieve_and_rerank_local(user_query)
            
            # Etapa de Generación con LLM
            response_text = generate_rag_response(user_query, evidences)
            
            # Mostrar la respuesta
            st.write(response_text)
            
            # Mostrar las evidencias de forma interactiva
            if evidences:
                with st.expander("🔍 Ver Evidencias Utilizadas para esta respuesta"):
                    for idx, (doc, meta, score) in enumerate(evidences):
                        st.markdown(f"**Evidencia #{idx+1} (Score: {score:.4f})**")
                        st.markdown(f"**Título:** {meta.get('title')}")
                        st.markdown(f"**Abstract:** {meta.get('abstract')}")
                        st.markdown("---")
                        
        # Guardar la respuesta y sus evidencias en la sesión de Streamlit
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "evidences": evidences
        })