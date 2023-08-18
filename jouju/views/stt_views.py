import speech_recognition as sr
from gtts import gTTS
import pyaudio
import playsound 
import time

from flask import Blueprint, render_template, request, redirect

bp = Blueprint('stt', __name__, url_prefix='/stt')

@bp.route('/', methods=['GET', 'POST'])
def speech_to_text(): 
    #일단은 한 번만 수행하도록 실행
    #방향 1. 일정 시간을 정하고 계속 입력 받기, 2. DB 구축해서 값 받아 저장
    r = sr.Recognizer()
    try:

        #음성 입력
        with sr.Microphone() as source:
            print('음성을 입력하세요')
            audio = r.listen(source)
            result = r.recognize_google(audio, language='ko-KR')
    except:
        print('Error')

    return render_template('stt.html', transcript = result)