from typing import Union
from fastapi import FastAPI

from models.kogpt2 import KoGPTChatbot
from models.blender import SpeachStyleConverter
from models.koBERT import EmotionClassification

app = FastAPI()
kogpt_chatbot = KoGPTChatbot()
speach_style_converter = SpeachStyleConverter()
emotion_classification = EmotionClassification()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/nlp/get_answer")
def read_item(text: Union[str, None] = None, target_style_name: Union[str, None] = None):
    answer = kogpt_chatbot.get_answer(text)
    result = speach_style_converter.convert(answer, target_style_name)
    emotion = emotion_classification.predict(result[0])
    return {"answer": result, "emotion": emotion}