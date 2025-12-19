from fastapi import FastAPI, HTTPException
from schemas import PreguntaRequest, PreguntaResponse 
# Lista de documentos a cargar en base de datos 

lista_pdfs = ["./documentos/ley_monotributo.pdf", 
              "./documentos/monotributo_anses.pdf", 
              "./documentos/monotributo_arca.pdf", 
              "./documentos/monotributo_mi_arg.pdf"] 

from llm import RAG_answer 
from orquestador_llm import orquestador 

app = FastAPI(title="Challenge final", version="1.0.0") 

@app.post("/ingest", status_code=201) 
def ingest_documents(): 
    from vector_store import ingestar_docs_db 
    
    if len(lista_pdfs) == 0:
        raise HTTPException(400, "No se enviaron documentos")
    
    ingestar_docs_db(lista_pdfs) 
    return {"message": "Documentos cargados en la base de datos vectorial."} 

@app.post("/orquestador", response_model=PreguntaResponse) 
def clasificar_intencion(request: PreguntaRequest): 
    try: 
        category = orquestador(request.pregunta) 
        if category == "saludo":
            answer = "Hola 👋 ¿En qué te puedo ayudar? \n Estoy para ayudarte con temas del monotributo"
            return PreguntaResponse(pregunta=request.pregunta, respuesta=answer) 
        
        elif category == "despedida":
            answer = "¡Hasta luego! 😊" 
            return PreguntaResponse(pregunta=request.pregunta, respuesta=answer) 
        
        elif category == "inapropriado": 
            answer = "Lo siento, no puedo responder a ese tipo de consultas." 
            return PreguntaResponse(pregunta=request.pregunta, respuesta=answer) 
        
        else:
            answer = RAG_answer(request.pregunta)
            return PreguntaResponse(pregunta=request.pregunta, respuesta=answer) 
    except Exception: raise HTTPException(status_code=502, detail="Error al comunicarse con el proveedor del LLM") 

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=False)