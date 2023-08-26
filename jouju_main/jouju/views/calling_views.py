from flask import Blueprint, render_template, send_file, Flask, request, jsonify
import os
import server_connection as server_connection
import speech_recognition as sr



bp = Blueprint('calling', __name__, url_prefix='/calling')


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
    chatbot_server = "http://192.168.0.32:8000/nlp/get_answer"
    voice_server = "http://192.168.0.8:5002/tts/synthesis"
    vedio_server = "http://192.168.0.50:5004/"

    audio_file_path = 'jouju/static/audio/'
    video_file_path = 'jouju/static/video/'

    style = "informal"

    data = request.json
    question = data.get('text')
    print('음성 인식 결과:', question)

    # server communication
    result = server_connection.get_answer(chatbot_server, question, style)
    print("answer : " + result['answer'][0])
    voice_file_name = server_connection.get_voice(voice_server, audio_file_path, result['answer'][0])
    vedio_file_name = server_connection.get_video(vedio_server, video_file_path, voice_file_name, result['emotion'])

    # return jsonify({'new_audio_filename': vedio_file_name})
    # return render_template('calling.html', vedio_file_name=vedio_file_name)
    return jsonify({'new_audio_filename': vedio_file_name})


# @bp.route('/video/<string:videoName>')
# def video(videoName):
#     return send_file("video/"+videoName)
