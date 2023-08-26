import requests
import json
import os


def get_answer(url, question, style):
    data = {"text": question, #STT 결과
            "target_style_name": style}
    nlp_response = requests.get(url, params=data)

    if nlp_response.status_code == 200:
        nlp_result = nlp_response.json()
        print('answer generate done')
        return nlp_result
    else:
        return nlp_response.status_code
    
    
def temp_tts(url, question, style):
    data = {"text": question, #STT 결과
            "target_style_name": style}

    file_response = requests.get(url, params=data)

    if file_response.status_code == 200:
        name = "static/audio/temp.wav"
        with open(name, 'wb') as file:
            file.write(file_response.content)
        return name
    else:
        return file_response.status_code
    

def get_voice(url, path, answer):
    data = {'result': answer}
    headers = {"Content-Type": "application/json"}
    
    file_response = requests.post(url, headers=headers, data=json.dumps(data))

    if file_response.status_code == 200:
        for i in range(100):
            name = f"temp{i}.wav"
            if os.path.isfile(path + name):
                pass
            else:
                with open(path + name, 'wb') as file:
                    file.write(file_response.content)
                    print('voice generate done')
                    break
        return name
    else:
        return 'vidoe generate err : ' + str(file_response.status_code)

def get_video(url, path, file_path, emotion):
    voice_path = 'jouju/static/audio/'
    try:
        os.rename(voice_path + file_path, voice_path + f'{emotion}.wav')
    except:
        os.remove(voice_path + f'{emotion}.wav')
        os.rename(voice_path + file_path, voice_path + f'{emotion}.wav')
    files = open(voice_path + f'{emotion}.wav', 'rb')
    upload = {'file': files}

    file_response = requests.post(url, files = upload)

    if file_response.status_code == 200:
        for i in range(100):
            name = f"temp{i}.mp4"
            if os.path.isfile(path + name):
                pass
            else:
                with open(path + name, 'wb') as file:
                    file.write(file_response.content)
                    print('vedio generate done')
                    break
        return name
    else:
        return 'vidoe generate err : ' + str(file_response.status_code)




if __name__ == '__main__':
    server_url_1 = "http://192.168.0.32:8000/nlp/get_answer"
    server_url_2 = "http://192.168.0.8:5002/tts/synthesis"

    server_url_3 = "http://192.168.0.6:5004/"  # 파일을 업로드할 서버의 URL

    question = '집가고 싶다'
    style = "choding"

    print('chatbot start')
    result = get_answer(server_url_1, question, style)
    print(result['answer'])
    print('voice start')
    file_res = get_voice(server_url_2, result['answer'][0], result['emotion'])
    print("---------------")
    # print('vedio start')
    # file_path = file_res
    # get_video(server_url_3, file_path)

    # result = temp_tts("http://192.168.0.32:8000/api/media_file", question, style)