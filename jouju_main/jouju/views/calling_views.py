from flask import Blueprint, render_template, send_file, Flask, request, jsonify
import os
import speech_recognition as sr
import time

import server_connection
from openai_api import OpenAIChat


bp = Blueprint('calling', __name__, url_prefix='/calling')
openai_chatbot = OpenAIChat()

@bp.route('/')
def base():
    # 시작 전 오디오 파일 제거
    dir = 'jouju/static/video'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir = 'jouju/static/audio'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    return render_template('calling.html')


@bp.route('/stt', methods=['POST'])
def stt():
    classification_server = "http://192.168.0.22:8000/nlp/classification"
    voice_server = "http://192.168.0.8:5002/tts/synthesis"
    # vedio_server = "http://192.168.0.50:5004/"
    vedio_server = "http://192.168.0.6:5004/"  # server

    audio_file_path = 'jouju/static/audio/'
    video_file_path = 'jouju/static/video/'

    # style = "informal"

    data = request.json
    question = data.get('text')
    print('음성 인식 결과:', question)

    if '노래 불러 줘' in question:
        vedio_file_name = 'jouju_sing.mp4'
        time.sleep(3)
        return jsonify({'new_audio_filename': vedio_file_name})
    else:
        answer = openai_chatbot.get_answer(question)
        emotion = server_connection.get_emotion(classification_server, answer)
        # result = {'answer': [answer], 'emotion': emotion['emotion']}
        print(f"answer : {answer} ({emotion})")
        voice_file_name = server_connection.get_voice(voice_server, audio_file_path, answer)
        vedio_file_name = server_connection.get_video(vedio_server, video_file_path, voice_file_name, emotion)
        return jsonify({'new_audio_filename': 'video/' + vedio_file_name})



    # return jsonify({'new_audio_filename': vedio_file_name})
    # return render_template('calling.html', vedio_file_name=vedio_file_name)
    


# @bp.route('/video/<string:videoName>')
# def video(videoName):
#     return send_file("video/"+videoName)
