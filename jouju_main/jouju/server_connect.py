import requests
import json
import os


def get_answer(url, question, style):
    data = {
        "text": question, #STT 결과
        "target_style_name": style}
    nlp_response = requests.get(url, params=data)

    if nlp_response.status_code == 200:
        nlp_result = nlp_response.json()
        return nlp_result
    else:
        return nlp_response.status_code
    
    
def temp_tts(url, question, style):
    data = {
        "text": question, #STT 결과
        "target_style_name": style}

    file_response = requests.get(url, params=data)

    if file_response.status_code == 200:
        name = "static/audio/temp.wav"
        with open(name, 'wb') as file:
            file.write(file_response.content)
        return name
    else:
        return file_response.status_code
    

def get_voice(url, answer, emotion):
    data = {'result': answer}
    headers = {"Content-Type": "application/json"}

    file_response = requests.post(url, headers=headers, data=json.dumps(data))

    if file_response.status_code == 200:
        # name = f"{emotion}.wav"
        name = "static/audio/temp.wav"
        os.remove(name)
        with open(name, 'wb') as file:
            file.write(file_response.content)
            print('voice generate done')
        return name
    else:
        return file_response.status_code

def get_video(url, file_path):
    files = open(file_path, 'rb')

    upload = {'file': files}

    file_response = requests.post(url, files = upload)

    if file_response.status_code == 200:
        with open("test.mp4", 'wb') as file:
            file.write(file_response.content)
    else:
        return file_response.status_code




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