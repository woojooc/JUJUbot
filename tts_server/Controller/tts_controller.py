from flask import request, Blueprint, jsonify, send_file
import base64
import requests
import subprocess
import re

bp = Blueprint('tts', __name__, url_prefix="/tts")


@bp.route('/synthesis', methods=['POST'])
def synthesis_tts():
    try:
        data = request.get_json()  # JSON 데이터 가져오기
        text = data["result"]
    except Exception as e:
        return jsonify(error="Invalid JSON data", message=str(e))

    print(f'원본 텍스트 : {text}')
    clean_text = re.sub(r'[^가-힣.!?~,0-9a-zA-Z]+', ' ', text)
    # bash
    command = f'cd tts_server/vits && python inference.py "{clean_text}"'
    print(f'서버 필터 : {command}')
    result = subprocess.run(command, shell=True)
    # 결과 출력
    print("Return Code:", result.returncode)
    print("Standard Output:", result.stdout)
    print("Standard Error:", result.stderr)

    wav_filename = 'tts.wav'
    return send_file(wav_filename, mimetype='audio/wav')

    # with open('tts_server/tts.wav', 'rb') as f:
    #     wav_data = f.read()
    # # base64로 디코드 필요
    # base64_encoded_wav = base64.b64encode(wav_data).decode('utf-8')
    # # main 서버 url
    # url = "http://localhost:5003/main"
    # # json.dump로 받아주세요
    # data_to_send = {
    #     "audio": base64_encoded_wav
    # }
    # response = requests.post(url, json=data_to_send)
    # print("Response:", response.text)

