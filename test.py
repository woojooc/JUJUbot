import os

# os.system('d:')
# os.system('cd D:\GitHub\JUJUbot\Wav2Lip')

os.system('python Wav2Lip/inference.py --checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth --face "data/01.mp4" --audio "data/wav00.wav"')