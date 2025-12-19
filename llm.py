import os
from dotenv import load_dotenv

from langchain_cohere import ChatCohere


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import chromadb
# Importa libreria para de embedding de Cohere
from langchain_cohere import CohereEmbeddings

# Importa libreria de Chroma para crear la DB vectorial
from langchain_chroma import Chroma


import hashlib

# funcion para crear claves de chache
def make_cache_key(query, context):
    text = query.lower().strip() + context
    return hashlib.sha256(text.encode()).hexdigest()


load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

llm = ChatCohere(model="command-a-03-2025", temperature=0.2)

# Se instancia la Clase out put parser para convertir las respuestas a un texto plano
output_parser = StrOutputParser()

# Se define el prompt que contiene la pregunta del usuario {query} y los chunks más relevantes {relevant_passage}
system_prompt = """Eres un agente conversacional que responde preguntas sobre monotributo y ARCA
a personas sin conocimiento de los temas. 

Debes seguir las siguientes reglas al generar la respuesta:
- Utilizá el CONTEXTO para responder la PREGUNTA.
- No generar información falsa
- No generar emojis
- Respuesta breve y amigable
- Siempre en español sin importar el idioma de la pregunta
- Usar el contexto para responder la pregunta
- Si la respuesta no está en el contexto, decí:
  "Lo siento, no tengo la información para responder a esa pregunta."
- NO debes generar respuestas que incluyan estereotipos, insultos o juicios subjetivos.
- Si el CONTEXTO no es relevante para la PREGUNTA, decí:
  "Lo siento, no tengo la información para responder a esa pregunta."
 """

user_prompt = """PREGUNTA: '{query}'
CONTEXTO: '{context}'
RESPUESTA:"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])

# Se construye la cadena para el RAG
chain = prompt | llm | output_parser


client = chromadb.PersistentClient(path="./midatabase")
embeddings = CohereEmbeddings(model="embed-v4.0")

vector_store = Chroma(
    client=client,
    collection_name="mi_coleccion",
    embedding_function=embeddings)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

RAG_CACHE = {}

def RAG_answer(query):
    """
    Recibe por parametros la consulta del usuario
    Devuelve una respuesta generada con RAG
    """
    
    # Almacena el matching de la query
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    chache_key = make_cache_key(query, context)

    # Controla si la respuesta ya está en cache
    if chache_key in RAG_CACHE:
        return RAG_CACHE[chache_key]    

    response = chain.invoke({"query": query, "context": context})

    RAG_CACHE[chache_key] = response

    return response


