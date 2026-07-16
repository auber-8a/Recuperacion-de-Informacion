import streamlit as st
import os
import zipfile
import urllib.request
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
# 1. DESCARGA AUTOMÁTICA DESDE HUGGING FACE (Directo, rápido y sin límites)
# ==============================================================================

# URL directa de descarga de tu Hugging Face Dataset
DB_DOWNLOAD_URL = "https://huggingface.co/datasets/Git-8a/chroma_db/resolve/main/chroma_db.zip?download=true"

DB_ZIP_PATH = "chroma_db.zip"
DB_DIR_PATH = "./chroma_db"

def download_db_from_huggingface():
    """Descarga la base de datos de 716MB desde Hugging Face, descomprime y organiza las rutas."""
    sqlite_file = os.path.join(DB_DIR_PATH, "chroma.sqlite3")
    
    # Si la carpeta de la base de datos existe pero está corrupta/incompleta (sin el archivo sqlite), la borramos
    if os.path.exists(DB_DIR_PATH) and not os.path.exists(sqlite_file):
        shutil.rmtree(DB_DIR_PATH)
        
    if not os.path.exists(DB_DIR_PATH):
        with st.spinner("📦 Descargando base de datos vectorial desde Hugging Face (716 MB)... Este proceso de inicio es único y puede tardar un minuto."):
            try:
                # Descargar el archivo zip de Hugging Face
                urllib.request.urlretrieve(DB_DOWNLOAD_URL, DB_ZIP_PATH)
                
                # Crear una carpeta temporal para la extracción segura
                temp_extract_dir = "./temp_extract"
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                os.makedirs(temp_extract_dir)
                
                # Extraer todo el contenido del zip
                with zipfile.ZipFile(DB_ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                # Buscar dinámicamente en qué subcarpeta se encuentra el archivo 'chroma.sqlite3'
                sqlite_found_path = None
                for root, dirs, files in os.walk(temp_extract_dir):
                    if "chroma.sqlite3" in files:
                        sqlite_found_path = root
                        break
                
                if sqlite_found_path:
                    # Mover los contenidos correctos a la carpeta './chroma_db' de tu app
                    shutil.move(sqlite_found_path, DB_DIR_PATH)
                    st.success("¡Base de datos cargada e inicializada exitosamente!")
                else:
                    st.error("Error: No se encontró el archivo de base de datos 'chroma.sqlite3' dentro del paquete descargado.")
                
                # Limpiar los archivos y carpetas temporales para ahorrar espacio en la nube
                if os.path.exists(DB_ZIP_PATH):
                    os.remove(DB_ZIP_PATH)
                if os.path.exists(temp_extract_dir):
                    shutil.rmtree(temp_extract_dir)
                    
            except Exception as e:
                st.error(f"Error al descargar o procesar la base de datos: {e}")
                st.info("Por favor, asegúrate de que el dataset en Hugging Face sea 'Public' y de que la URL de descarga esté bien escrita.")

# Ejecutar la descarga automatizada
download_db_from_huggingface()

# ==============================================================================
# CARGA DE RECURSOS (Modelos y Base de Datos Vectorial)
# ==============================================================================
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("ms-marco-MiniLM-L-6-v2")
    
    # Inicializar cliente persistente apuntando al directorio de la base de datos
    chroma_client = chromadb.PersistentClient(path=DB_DIR_PATH)
    collection = chroma_client.get_collection(name="arxiv_papers")
    
    return embedding_model, reranker, collection

try:
    embedding_model, reranker, collection = load_resources()
except Exception as e:
    st.error(f"Error al inicializar la base de datos: {e}")

# ==============================================================================
# 2. CLIENTE DE GEMINI Y GENERACIÓN RESPUESTA RAG
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def generate_rag_response(query, context_documents):
    if not GEMINI_API_KEY:
        return "❌ Error: La variable de entorno GEMINI_API_KEY no está configurada en los Secrets de la plataforma."
    
    # Concatenar las evidencias recuperadas seleccionadas por el re-ranker
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

# Recuperar y ordenar usando el CrossEncoder
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

# Inicializar el historial del chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial de mensajes de forma dinámica sin reiniciar la aplicación
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

# Capturar la entrada de texto del usuario
if user_query := st.chat_input("Escribe tu consulta sobre IA, Robótica, etc. (Ej: How is reinforcement learning used in robotics?)"):
    
    # 1. Mostrar la pregunta del usuario en pantalla
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # 2. Generar la respuesta en segundo plano
    with st.chat_message("assistant"):
        with st.spinner("Buscando en arXiv y analizando evidencias..."):
            
            # Etapa de Recuperación y Re-ranking
            evidences = retrieve_and_rerank_local(user_query)
            
            # Etapa de Generación de Respuesta
            response_text = generate_rag_response(user_query, evidences)
            st.write(response_text)
            
            # Mostrar evidencias de forma estructurada
            if evidences:
                with st.expander("🔍 Ver Evidencias Utilizadas para esta respuesta"):
                    for idx, (doc, meta, score) in enumerate(evidences):
                        st.markdown(f"**Evidencia #{idx+1} (Score: {score:.4f})**")
                        st.markdown(f"**Título:** {meta.get('title')}")
                        st.markdown(f"**Abstract:** {meta.get('abstract')}")
                        st.markdown("---")
                        
        # Guardar respuesta y evidencias asociadas
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "evidences": evidences
        })