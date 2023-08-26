from flask import Blueprint, render_template, request, jsonify
import server_connection as server_connection
import os

bp = Blueprint('voice', __name__, url_prefix='/voice_call')

@bp.route('/')
def voice_call():
    # 시작 전 오디오 파일 제거
    dir = 'jouju/static/audio'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    return render_template('voicecall.html')


@bp.route('/send_text', methods=['POST'])
def send_text():
    chatbot_server = "http://192.168.0.32:8000/nlp/get_answer"
    voice_server = "http://192.168.0.8:5002/tts/synthesis"
    vedio_server = "http://192.168.0.6:5004/"

    audio_file_path = 'jouju/static/audio/'

    data = request.get_json()
    question = data.get('text', '')
    style = "informal"
    print('음성 인식 결과:', question)

    # server communication
    result = server_connection.get_answer(chatbot_server, question, style)
    print("answer : " + result['answer'][0])
    
    voice_file_name = server_connection.get_voice(voice_server, audio_file_path, result['answer'][0])
    
    # vedio_file_name = test.get_video(vedio_server, voice_file_name)

    # result = test.temp_tts("http://192.168.0.32:8000/api/media_file", question, style)
    # print(result)

    return jsonify({'new_audio_filename': voice_file_name})
