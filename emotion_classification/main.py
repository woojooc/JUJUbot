from fastapi import FastAPI, Request
from generate import *

emotion_classification = EmotionClassification()
app = FastAPI()


@app.get("/")
def read_root(item_id: str, request: Request):
    client_host = request.client.host

    result = emotion_classification.predict(item_id)
    return result