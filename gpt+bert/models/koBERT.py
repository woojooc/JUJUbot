import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model




class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 7,   # 감정 클래스 수로 조정
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class EmotionClassification():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

        tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


        # BERT  모델 불러오기
        self.model = BERTClassifier(bertmodel,  dr_rate = 0.5).to(self.device)
        # self.model = torch.load("models/model_3_0.8999485596707815.pt", map_location=self.device)
        self.model.load_state_dict(torch.load('resource/models/koBERT_best.pt', map_location=self.device))

    def predict(self, predict_sentence): # input = 감정분류하고자 하는 sentence
        max_len = 64
        batch_size = 24

        data = [predict_sentence, '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, self.tok, max_len, True, False) # 토큰화한 문장
        
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 0) # torch 형식 변환
        
        self.model.eval() 

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)

            valid_length = valid_length
            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)


            test_eval = []
            for i in out: # out = model(token_ids, valid_length, segment_ids)
                logits = i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("fear")
                elif np.argmax(logits) == 1:
                    test_eval.append("surprise")
                elif np.argmax(logits) == 2:
                    test_eval.append("angry")
                elif np.argmax(logits) == 3:
                    test_eval.append("sadness")
                elif np.argmax(logits) == 4:
                    test_eval.append("neutral")
                elif np.argmax(logits) == 5:
                    test_eval.append("happiness")
                elif np.argmax(logits) == 6:
                    test_eval.append("disgust")

            return test_eval[0]





if __name__ == '__main__':
    emotion_classification = EmotionClassification()
    # 질문에 0 입력 시 종료
    end = 1
    while end == 1 :
        sentence = input("하고싶은 말을 입력해주세요 : ")
        if sentence == "0" :
            break
        print(emotion_classification.predict(sentence))
        print("\n")
