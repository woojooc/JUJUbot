from flask import Blueprint, render_template, request, jsonify
import os
import time

import server_connection
from openai_api import OpenAIChat


bp = Blueprint('voice', __name__, url_prefix='/voice_call')
openai_chatbot = OpenAIChat()


@bp.route('/')
def voice_call():
    # 시작 전 오디오 파일 제거
    dir = 'jouju/static/audio'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    return render_template('voicecall.html')


@bp.route('/send_text', methods=['POST'])
def send_text():
    classification_server = "http://192.168.0.22:8000/nlp/classification"
    # chatbot_server = "http://192.168.0.32:8000/nlp/get_answer"
    voice_server = "http://192.168.0.8:5002/tts/synthesis"
    # vedio_server = "http://192.168.0.6:5004/"

    audio_file_path = 'jouju/static/audio/'

    data = request.get_json()
    question = data.get('text', '')
    # style = "informal"
    print('음성 인식 결과:', question)

    if '노래 불러 줘' in question:
        audio_file_name = 'joujou_sing.wav'
        time.sleep(3)
        return jsonify({'new_audio_filename': audio_file_name})
    else:
        answer = openai_chatbot.get_answer(question)
        emotion = server_connection.get_emotion(classification_server, answer)
        print(f"answer : {answer} ({emotion})")

        voice_file_name = server_connection.get_voice(voice_server, audio_file_path, answer)
        return jsonify({'new_audio_filename': 'audio/' + voice_file_name})

