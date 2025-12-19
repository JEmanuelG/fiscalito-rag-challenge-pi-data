import chromadb

from langchain_community.document_loaders import PyMuPDFLoader

# Impota la libreria para hacer chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importa libreria para de embedding de Cohere
from langchain_cohere import CohereEmbeddings


# Importa libreria de Chroma para crear la DB vectorial
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

client = chromadb.PersistentClient(path="./midatabase")


def ingestar_docs_db(lista_rutas: list):
    """
    Recibe una lista de str con las rutas de los pdfs que se van a
    procesar.
    Genera los chunkings, les crea los embeddings y los almacena en una base de datos vectorial.
    Retorna base de datos vectorial.
    """

    # Lista donde se va a almacenar el contenido de los pdfs
    pdfs_totales = []

    # Itera sobre los pdfs para extraer el contenido de cada uno
    for ruta in lista_rutas:
        loader = PyMuPDFLoader(ruta)

        # Va agragando las paginas a docs totales con extend, ssimilar a append 
        pdfs_totales.extend(loader.load())

    # Realiza una limpieza borrando caracteres no deseados como tabulaciones,
    # espacios en blanco y saltos de linea
    pdfs_totales = list(filter(lambda doc: doc.page_content.strip(), pdfs_totales))

    # Hace chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=90)

    # Almacena el chunking en variable
    docs_chunks = text_splitter.split_documents(pdfs_totales)

    # Instancia el modelo de embedding
    embeddings_cohere = CohereEmbeddings(model="embed-v4.0")

    # Se crea la "mi_coleccion" y se almacenan chunks y embeddings
    vector_store = Chroma.from_documents(
        documents=docs_chunks,
        embedding=embeddings_cohere,
        client=client,
        collection_name="mi_coleccion")
    
