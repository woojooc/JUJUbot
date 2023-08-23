from flask import Blueprint, render_template, request, url_for, send_file
from werkzeug.utils import redirect
from Wav2Lip import inference as Inf
import os

import time, datetime
import threading, shutil

from enum import Enum, auto

py_path = "Wav2Lip/inference.py"
model_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"
res_path = "static/video/result_voice.mp4"

bp = Blueprint('main', __name__,url_prefix='/' )

Inf.load_detector()
Inf.load_model(model_path)

# 추론 결과를 저장할 변수
inf_completed = False
event = threading.Event()

class E_emo(Enum):
    fear = auto()  # 1
    angry = auto()
    disgust = auto()
    happiness = auto()
    neutral = auto()
    sadness = auto()
    surprise = auto()


def save_audio(file_data, filename):
    with open(filename, 'wb') as f:
        f.write(file_data)
    print(f"[*] Saved audio file as {filename}")

def Backup_mp4():
    delete_path = os.path.join(os.getcwd(), "Wav2Lip", "results", "result_voice.mp4")
    
    src_path = os.path.join(os.getcwd(), "flask_", "static","video", "result_voice.mp4")
    dst_path = os.path.join(os.getcwd(), "flask_", "static","video","backup")
    files = os.listdir(dst_path)
    dst_name = "result" + '{:03}'.format(len(files)) + '.mp4'
    dst_path = os.path.join(dst_path, dst_name)

    # 라이브러리에 있는 파일 지우기
    try:
        os.remove(delete_path)
        print(f"{delete_path} 파일이 삭제되었습니다.")
    except FileNotFoundError:
        print(f"{delete_path} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파일 삭제 중 오류가 발생했습니다: {e}")

    # flask에 있는 파일 옮기기
    try:
        shutil.move(src_path, dst_path)
        print(f"{src_path} 파일이 {dst_path}로 이동되었습니다.")
    except FileNotFoundError:
        print(f"{src_path} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파일 이동 중 오류가 발생했습니다: {e}")

    return dst_path


def select_mp4(f_name):
    root = os.getcwd()
    face_path = os.path.join(root, "flask_", "static", "video")
    str_name = f_name.split('.')[0]
    print("this is ==== ", str_name)
    e_name = E_emo[str_name]
    print("this is === ",e_name)
    if e_name in ( E_emo.angry , E_emo.disgust ,E_emo.fear):
        face_path = os.path.join(face_path, "ang.mp4")
    elif e_name in (E_emo.neutral, E_emo.surprise):
        face_path = os.path.join(face_path, "neu.mp4")
    elif e_name == E_emo.happiness:
        face_path = os.path.join(face_path, "hap.mp4")
    elif e_name == E_emo.sadness:
        face_path = os.path.join(face_path, "sad.mp4")
    else:
        face_path = os.path.join(face_path, "neu.mp4")
    
    print("face path = ", face_path)
    return face_path

# 스레드 1 :  추론
def thd_inference(face_p, audio_p):
    print("Inffffff")

    Inf.addparser(model_path,face_p,audio_p)
    Inf.main()
    
    time.sleep(5)

# 스레드 2 :  결과 파일 생성 확인
def thd_new_files():
    global inf_completed, event

    target_folder = "Wav2Lip/results"  # D:/GitHub/JUJUbot/wj/ 감시할 폴더 경로
    print('------find file start -----')

    while True:
        time.sleep(1)  # 1초마다 폴더 스캔

        files = os.listdir(target_folder)
        new_files = [file for file in files if file.endswith(".mp4")]  # 새로운 .txt 파일 찾기

        if new_files:
            print("새로운 파일이 생성되었습니다:", new_files)
            time.sleep(1)
            event.set()
            break

@bp.route('/', methods=['POST'])
def main_index():
    global inf_completed, event

    if request.method == 'POST':
        inf_completed = False

        # 보이스 데이터 받기
        print(request.files['file'])
        print(request.files['file'].filename)
        f = request.files['file']
        filename = f.filename
        save_path = os.path.join(os.getcwd(), "flask_", "static", "audio", filename)
        audio = f.save(save_path)

        face_path = select_mp4(filename)

        # 추론을 백그라운드에서 실행하는 스레드 생성
        inf_thread = threading.Thread(target=thd_inference, args=(face_path, save_path))
        inf_thread.start()

        # 파일 감시를 백그라운드에서 실행하는 스레드 생성
        file_thread = threading.Thread(target=thd_new_files)
        file_thread.start()

        while True:
            if event.wait(1):  # 이벤트를 1초마다 체크
                print("File Backup and Mp4 send to Main Server")
                inf_completed = True
                event.clear()

                #res_path = os.path.join(os.getcwd(),"flask_", "static", "video", "result_voice.mp4")

                dst_path = Backup_mp4()
                return send_file(dst_path, mimetype='video/mp4')
    
    return "File sended"
