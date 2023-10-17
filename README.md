# JUJUbot  
  
  
어린이 대화 친구 AI 프로젝트  
사용자 보이스 대화를 입력 받아 캐릭터가 대답하는 영상을 송출한다.  
  
  
## Branch   
   
### 1 jouju_main (서버 0) 
      - 사용 모델  : Flask, GPT4(랭체인)
      - 서비스 기능 : 시크릿 주주와 음성 통화 및 영상 통화
  
### 2 gpt+bert (서버 1) : 텍스트를 인풋으로 받아 감정 분류 결과를 메인 서버에 전달한다.
      - 사용 모델  : koBERT ( GPT4 -> 메인 서버로 이동 )


### 3 tts_server (서버 2) : 텍스트를 인풋으로 받아 캐릭터의 음성을 메인 서버에 전달한다.
      - 사용 모델 : VITS (text -> Speech), RVC (데이터 가공 및 수집)

 
### 4 wj (서버 3) : 보이스(wav)와 감정(str)을 인풋으로 받아 캐릭터 영상을 메인 서버에 전달한다.  
      - 사용 모델  : Wav2Lip ( 보이스.wav + 캐릭터.mp4  => 입 합성 영상 ) 
      - 환경 : python3.7 CUDA 11.0
        환경 세팅 관련 노션 페이지
        https://brash-visitor-06b.notion.site/Wav2Lip-4dfa9b0d059144a789445dc0ceeac027?pvs=4
        Flask에 올리기 위한 세팅 관련 노션 페이지
        https://brash-visitor-06b.notion.site/Web-6a2df60d72bb499a9eead452fcc00472?pvs=4
      - 모델 : Wav2Lip + GAN,  s3fd-619a316812.pth(face_detection)
      
      
