import sys
import os

wav_lib = "Wav2Lip"
absolute_path = os.path.abspath(wav_lib)
sys.path.append(absolute_path)

print("000===", absolute_path)
print("001===",sys.path)