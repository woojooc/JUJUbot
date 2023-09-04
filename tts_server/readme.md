아래 install(torch 버전은 안맞춰도 될 수 있음):
# pip install git+https://github.com/huggingface/transformers sentencepiece datasets
# pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

임시로 허깅페이스 모델을 가져다 써서 처음 실행할 때 데이터셋을 다운로드 받음.
깜빡하고 json이 아니라 form으로 받도록 짜버림;