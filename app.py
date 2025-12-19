import streamlit as st
import requests

# =============================
# Configuración general
# =============================
API_URL = "http://127.0.0.1:8000/orquestador"

st.set_page_config(
    page_title="Chat Monotributo ARCA",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Chat RAG - Monotributo ARCA Argentina")
st.markdown(
    "Consultá sobre **Monotributo (ARCA)** utilizando documentos oficiales."
)

# =============================
# Estado de la conversación
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =============================
# Mostrar historial de chat
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =============================
# Input del usuario
# =============================
prompt = st.chat_input("Escribí tu consulta sobre Monotributo...")

if prompt:
    # Mostrar mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # =============================
    # Llamada al backend FastAPI
    # =============================
    try:
        with st.spinner("Consultando documentación oficial..."):
            response = requests.post(
                API_URL,
                json={"pregunta": prompt},
                timeout=180
            )

        if response.status_code == 200:
            data = response.json()
            answer = data["respuesta"]
        else:
            answer = "❌ Error al consultar el backend."

    except Exception as e:
        answer = f"⚠️ No se pudo conectar con la API.\n\nDetalle: `{e}`"

    # =============================
    # Mostrar respuesta del LLM
    # =============================
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)