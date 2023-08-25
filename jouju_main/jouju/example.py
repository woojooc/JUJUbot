from flask import Flask, request, jsonify, render_template
import test
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_text', methods=['POST'])
def send_text():
    chatbot_server = "http://192.168.0.32:8000/nlp/get_answer"
    voice_server = "http://192.168.0.8:5002/tts/synthesis"
    vedio_server = "http://192.168.0.6:5004/"

    question = ''
    style = "choding"

    # server communication
    result = test.get_answer(chatbot_server, question, style)
    print("answer : " + result['answer'][0])
    print('answer generate done')
    voice_file_name = test.get_voice(voice_server, result['answer'][0], result['emotion'])
    # vedio_file_name = test.get_video(vedio_server, voice_file_name)
    # print('vedio generate done')

    voice_filename = 'temp.wav'

    return jsonify({'new_audio_filename': voice_filename})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
