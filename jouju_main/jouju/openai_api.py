from pathlib import Path
import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory


class OpenAIChat():
    def __init__(self) -> None:
        self.prompt = self.get_prompt()
        self.model = self.get_model()
    
    def get_model(self) -> ChatOpenAI:
        secret_path = Path("jouju/resource").joinpath("secret.json")
        secrets = json.loads(open(secret_path).read())
        openai_api_key = secrets["OPENAI_API_KEY"]
        chat_model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
        conversation = ConversationChain(
            prompt=self.prompt,
            llm=chat_model,
            verbose=False,
            memory=ConversationBufferWindowMemory(ai_prefix="AI Assistant", k=5)
        )
        return conversation
    
    def get_prompt(self) -> ChatPromptTemplate:
        template = """
당신의 역할은 "시크릿 쥬쥬"라는 가상의 인물을 연기하는 것입니다.
"시크릿 쥬쥬"에 대한 정보는 https://namu.wiki/w/%EC%A5%AC%EC%A5%AC(%EC%B9%98%EB%A7%81%EC%B9%98%EB%A7%81%20%EC%8B%9C%ED%81%AC%EB%A6%BF%20%EC%A5%AC%EC%A5%AC) 링크를 참고하여 응답해야 함
링크된 문서에서 나이, 신체, 데뷔곡, MBTI, 별명, 취미를 참고해야 함
대화의 대상은 아동이므로 친근하고 친밀감을 줄 수 있는 말투와 쉬운 내용으로 응답해야 함
가상의 인물, 허구의 인물 등의 출력은 제외함
응답의 길이는 항상 20자 보다 작아야 함

Current conversation:
{history}
Human: {input}
AI Assistant:"""

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        return prompt
    
    def get_answer(self, text) -> str:
        return self.model.predict(input=text)
        
        
if __name__ == '__main__':
    text = '배고파'
    openai_chatbot = OpenAIChat()
    result = openai_chatbot.get_answer(text)
    print(result)

    # classification_server = "http://192.168.0.32:8000/nlp/classification"
    # emotion = server_connection.get_emotion(classification_server, result)
    # print(emotion)