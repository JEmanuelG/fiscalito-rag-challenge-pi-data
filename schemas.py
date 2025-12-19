from pydantic import BaseModel



class PreguntaRequest(BaseModel):
    pregunta: str

class PreguntaResponse(BaseModel):
    pregunta: str
    respuesta: str