import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA E INTERFAZ
# ==============================================================================
st.set_page_config(
    page_title="arXiv Scientific RAG Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 arXiv Research RAG Assistant")
st.markdown("""
Bienvenido al asistente de recuperación científica. Realiza consultas en lenguaje natural 
sobre el corpus de artículos de arXiv. El sistema utiliza **Búsqueda Semántica Densa**, 
un modelo **Cross-Encoder para Re-ranking**, y un **LLM** para formular respuestas con base científica.
""")

# ==============================================================================
# 2. CARGA EFICIENTE DE MODELOS Y BASE DE DATOS (CACHED)
# ==============================================================================
@st.cache_resource
def load_resources():
    # Carga de modelos de SentenceTransformers
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker_model = CrossEncoder("ms-marco-MiniLM-L-6-v2")
    
    # Conexión a la Base de Datos Vectorial Persistente previamente guardada
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="arxiv_papers")
    
    return embedding_model, reranker_model, collection

try:
    bi_encoder, cross_encoder, vector_collection = load_resources()
    st.success(f"✅ Modelos indexados y base vectorial cargada ({vector_collection.count()} papers listos).")
except Exception as e:
    st.error(f"❌ Error al cargar los recursos o la carpeta ./chroma_db: {e}")
    st.stop()

# ==============================================================================
# 3. PIPELINE DE RECUPERACIÓN Y RE-RANKING
# ==============================================================================
def retrieve_and_rerank(query, top_n=10, top_k=3):
    query_embedding = bi_encoder.encode([query]).tolist()
    initial_results = vector_collection.query(
        query_embeddings=query_embedding,
        n_results=top_n
    )
    
    fetched_docs = initial_results['documents'][0]
    fetched_metadatas = initial_results['metadatas'][0]
    
    if not fetched_docs:
        return []

    # Re-ranking con el CrossEncoder
    pairs = [[query, doc] for doc in fetched_docs]
    rerank_scores = cross_encoder.predict(pairs)
    
    scored_docs = list(zip(fetched_docs, fetched_metadatas, rerank_scores))
    scored_docs.sort(key=lambda x: x[2], reverse=True)
    
    return scored_docs[:top_k]

# ==============================================================================
# 4. COMPONENTE DE GENERACIÓN LLM (EJEMPLO CON GEMINI API SECURE)
# ==============================================================================
def generate_llm_response(query, evidence_docs):
    # Recuperación segura de la API KEY desde variables de entorno de la nube
    api_key = os.environ.get("GEMINI_API_KEY")
    
    context_text = ""
    for i, (doc, meta, score) in enumerate(evidence_docs):
        context_text += f"--- Document Evidence #{i+1} (Re-rank Score: {score:.4f}) ---\n"
        context_text += f"{doc}\n\n"

    # Prompt con restricciones explícitas contra alucinaciones (Requerimiento de examen)
    prompt = f"""
You are an expert scientific research assistant. Answer the user's query using strictly the provided document evidences from arXiv papers. 
If the evidence does not contain enough information to answer the query, clearly state that the corpus does not contain sufficient information.

Document Evidences:
{context_text}

User Query: {query}

Answer:
"""
    
    if not api_key:
        # Fallback didáctico en caso de que falte configurar la variable de entorno en la nube
        return f"⚠️ [API Key no configurada]. El prompt estructurado con evidencias se generó con éxito:\n\n{prompt}"

    try:
        # Reemplazar por tu cliente LLM preferido (ej. google-genai)
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ Error en la llamada al LLM: {e}"

# ==============================================================================
# 5. HISTORIAL DEL CHAT INTERACTIVO (REQUERIMIENTO INTERFAZ WEB)
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Mostrar evidencias si están adjuntas al mensaje del asistente
        if "evidences" in message:
            with st.expander("🔍 Ver Evidencias Científicas Utilizadas"):
                for idx, (doc, meta, score) in enumerate(message["evidences"]):
                    st.markdown(f"**Documento #{idx+1} | Score: {score:.4f}**")
                    st.markdown(f"*Título:* {meta.get('title')}")
                    st.markdown(f"*Abstract:* {meta.get('abstract')}")
                    st.divider()

# Capturar entrada del usuario
if user_query := st.chat_input("Escribe tu consulta aquí (ej. What are main applications of GNNs?)..."):
    
    # 1. Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Procesar respuesta del Asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando en arXiv y re-rankeando evidencias..."):
            # Recuperar y filtrar evidencias relevantes
            top_evidences = retrieve_and_rerank(user_query)
            
        with st.spinner("Generando respuesta aumentada (RAG)..."):
            # Generar respuesta final usando el LLM
            llm_response = generate_llm_response(user_query, top_evidences)
            
        # Renderizar respuesta
        st.markdown(llm_response)
        
        # Presentar las evidencias de manera elegante en un contenedor colapsable (Requerimiento F)
        if top_evidences:
            with st.expander("🔍 Ver Evidencias Científicas Utilizadas"):
                for idx, (doc, meta, score) in enumerate(top_evidences):
                    st.markdown(f"**Documento #{idx+1} | Score: {score:.4f}**")
                    st.markdown(f"*Título:* {meta.get('title')}")
                    st.markdown(f"*Abstract:* {meta.get('abstract')}")
                    st.divider()
                    
        # Guardar en el estado de la sesión
        st.session_state.messages.append({
            "role": "assistant", 
            "content": llm_response,
            "evidences": top_evidences
        })