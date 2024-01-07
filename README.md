# 👸 어린이 대화 친구 쥬쥬 : 어린이들을 위한 대화형 챗봇 서비스

메타버스 아카데미 8월 프로젝트

#### 🎥 시연 영상 보러가기([Click]())
#### 📙 발표자료 보러가기([Click]())


<br/>

# :family: 팀원 소개 및 역할

**개발기간: 2023.08.01 ~ 2023.08.30**

### 세부 역할 분담

<table>
    <tbody>
        <tr>
            <td><b>김해니</b></td>
            <td>11</td>
        </tr>
        <tr>
            <td><b>김우정</b></td>
            <td>11</td>
        </tr>
        <tr>
            <td><b>라경훈</b></td>
            <td>11</td>
        </tr>
        <tr>
            <td><b>이승현</b></td>
            <td>11</td>
        </tr>
        <tr>
            <td><b>정민교</b></td>
            <td>11</td>
        </tr>
    </tbody>
</table>

<br/>

# 🤝 융합 구조도

<br/>

# 💡 프로젝트 소개

어린이 대화 친구 AI 프로젝트  
사용자 보이스 대화를 입력 받아 캐릭터가 대답하는 영상을 송출한다.

### 목적 및 필요성

인공지능 스피커와 아동 간의 상호작용은 언어 표현 능력에 영향을 미치는 것으로 나타났다. 더 나아가 아동들이 자신이 좋아하는 캐릭터의 음성과 영상과 상호작용한다면, 이를 통해 흥미와 적극적인 참여를 이끌어 언어 발달에 긍정적인 영향을 미칠 것이다. 아동들이 선호하는 캐릭터와 실시간으로 소통하는 특별한 경험을 제공하며, 추후 교육 도구로 활용할 수 있는 가치가 있다.

<br/>

# :scroll: 주요 내용

### 주요 기능

- Speech to Text를 통해 아이들의 음성을 챗봇의 입력값으로 받음 
- 챗봇 답변을 Text to Voice를 통해 학습시킨 캐릭터의 목소리도 변환 (음성 전화)
- 캐릭터 영상에 음성을 입혀 Voice to Video로 영상 송출 (영상 전화) 


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
      
<br/>

# 🛠 기술 스택

### - 언어
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

### - 주요 라이브러리
 <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white"> <img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white"> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/openai-412991?style=for-the-badge&logo=openai&logoColor=white"> <img src="https://img.shields.io/badge/langchain-EC1C24?style=for-the-badge&logo=langchain&logoColor=white">

### - 개발 툴
<img src="https://img.shields.io/badge/VS code-2F80ED?style=for-the-badge&logo=VS code&logoColor=white"> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white">

### - 협업 툴
<img src="https://img.shields.io/badge/Github-181717?style=for-the-badge&logo=Github&logoColor=white"> <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white">

# 🔍 참고자료

### Papers

1. 

### GitHub

1. 
2. 


