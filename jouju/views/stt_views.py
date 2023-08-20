import speech_recognition as sr
from gtts import gTTS
import pyaudio
import playsound 
import time
import requests

from flask import Blueprint, render_template, request, redirect

bp = Blueprint('stt', __name__, url_prefix='/stt')

@bp.route('/', methods=['GET', 'POST'])
def speech_to_text(): 
    #일단은 한 번만 수행하도록 실행
    #방향 1. 일정 시간을 정하고 계속 입력 받기, 2. DB 구축해서 값 받아 저장
    r = sr.Recognizer()
    result = '안녕 만나서 반가워!' #default 값 설정

    try:
        #음성 입력

        with sr.Microphone() as source:
            print('음성을 입력하세요')
            audio = r.listen(source)
            result = r.recognize_google(audio, language='ko-KR')
            print(result)
                
            if result == '종료':
                return render_template('index.html') #'종료'라고 말할 시 초기 화면으로 이동

             # NLP 서버로 전송할 데이터
            nlp_server_url = "http://192.168.0.21:8000/nlp/get_answer"
            nlp_post_data = {
                "text": result, #STT 결과
                "target_style_name": "choding" #초등학생 말투
            }

            # NLP 서버에 요청 보내기
            nlp_response = requests.get(nlp_server_url, params=nlp_post_data)
            
            if nlp_response.status_code == 200:
                nlp_result = nlp_response.json()
                print("NLP 결과:", nlp_result)
            else:
                print("NLP 서버 응답 실패:", nlp_response.status_code)
                    

    except:
        
        #print('Error')
        # NLP 서버로 전송할 데이터
        nlp_server_url = "http://192.168.0.21:8000/nlp/get_answer"
        nlp_post_data = {
            "text": result, #STT 결과
            "target_style_name": "choding" #초등학생 말투
            }

        # NLP 서버에 요청 보내기
        nlp_response = requests.get(nlp_server_url, params=nlp_post_data)
            
        if nlp_response.status_code == 200:
            nlp_result = nlp_response.json()
            print("NLP 결과:", nlp_result)
        else:
            print("NLP 서버 응답 실패:", nlp_response.status_code)
    

    return render_template('stt.html', question = result, answer = nlp_result['answer'][0], emotion=nlp_result['emotion'])