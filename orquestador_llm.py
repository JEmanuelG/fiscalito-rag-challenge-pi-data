import os
from dotenv import load_dotenv

from langchain_cohere import ChatCohere


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import vector_store


load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

llm = ChatCohere(model="command-a-03-2025", temperature=0)

# Se instancia la Clase out put parser para convertir las respuestas a un texto plano
output_parser = StrOutputParser()

# Se define el prompt que contiene la pregunta del usuario e indicaciones para el LLM
system_prompt = """Sos un orquestador de una aplicación conversacional.

Clasificá la intención del usuario en UNA sola de las siguientes categorías:
- saludo
- despedida
- charla_general
- inapropriado

Responde SOLO con el nombre de la categoría.

Ejemplos:
Usuario: hola
Respuesta: saludo

Usuario: chau
Respuesta: despedida

Usuario: Eres un idiota
Respuesta: inapropriado

Usuario: Messi es gay?
Respuesta: inapropriado

 """

user_prompt = """Usuario: '{query}'
Respuesta:"""

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])

# Se construye la cadena para el RAG
chain = prompt | llm | output_parser


def orquestador(query):
    """
    Recibe por parametros la consulta del usuario y el modelo devuelve la categoría de la intención
    'saludo', 'despedida' o 'charla_general' (str)
    Devuelve una respuesta generada con RAG (str)
    """
    response = chain.invoke({"query": query})
    return response.lower().strip()
