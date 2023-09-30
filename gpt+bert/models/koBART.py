from transformers import AutoTokenizer
from transformers import pipeline
from pathlib import Path


class SpeachStyleConverter():
    def __init__(self) -> None:
        self.model_name = "gogamza/kobart-base-v2"
        self.set_style_map()
        self.set_tokenizer()
        self.set_pipeline()
        
    def set_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def set_pipeline(self):
        model_path = Path("resource/models/checkpoint-4000")
        self.pipeline = pipeline("text2text-generation", model=model_path, tokenizer=self.model_name)
    
    def convert(self, input:str, target_style_name:str) -> str:
        prompt = f"{target_style_name} 말투로 변환:{input}"
        output = self.pipeline(prompt, num_return_sequences=1, max_length=200)
        return [x["generated_text"] for x in output]
    
    def set_style_map(self):
        style_map = {
            "formal": "문어체",
            "informal": "구어체",
            "android": "안드로이드",
            "azae": "아재",
            "chat": "채팅",
            "choding": "초등학생",
            "emoticon": "이모티콘",
            "enfp": "enfp",
            "gentle": "신사",
            "halbae": "할아버지",
            "halmae": "할머니",
            "joongding": "중학생",
            "king": "왕",
            "naruto": "나루토",
            "seonbi": "선비",
            "sosim": "소심한",
            "translator": "번역기"
        }
        self.style_map = style_map
        self.style_rev_map = {value: key for key, value in style_map.items()}
        
