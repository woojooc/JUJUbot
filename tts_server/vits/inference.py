import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import re
import sys
import soundfile as sf

model_name = 'jouju'
device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda:0'

def get_text(text, hps):
    text = re.sub('[\s・·+]', ' ', text).replace('…', '.').strip()
    text = re.sub(r'[^가-힣.!?~,0-9a-zA-Z]+', ' ', text)
    print(f'터미널 필터:{text}')
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file(f"./configs/{model_name}.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(f"./logs/{model_name}/G_109000.pth", net_g, None)

text = sys.argv[1]

stn_tst = get_text(text, hps)

with torch.no_grad():
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    sf.write("../tts.wav", audio, hps.data.sampling_rate)

