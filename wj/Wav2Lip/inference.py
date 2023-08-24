from os import listdir, path
import numpy as np
# import os
# print("======", os.getcwd())
import scipy, cv2, os, sys, argparse

from . import audio
try:
	from . import face_detection
	print("Imported well")
except ImportError as e:
	print("Import Error", e)

import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch

from .models import Wav2Lip

import platform

import shutil

# Multi GPU
import torch
import torch.nn as nn

# Video load
from .config import E_emo

class CLI_Parser:
	def __init__(self):
		self.h= 'Inference code to lip-sync videos in the wild using Wav2Lip models'
		self.checkpoint_path =r"checkpoints/wav2lip_gan.pth" # Name of saved checkpoint to load weights from
		self.face=r"D:\GitHub\JUJUbot\wj\flask_\static\video\neu.mp4" # Filepath of video/image that contains faces to use
		self.audio=r"D:\GitHub\JUJUbot\wj\flask_\static\audio\wav00.wav" # Filepath of video/audio file to use as raw audio source
		self.outfile=r'results/result_voice.mp4' # Video path to save result. See default for an e.g.
		self.static=False # If True, then use only first video frame for inference
		self.fps=25. # Can be specified only if input is a static image (default: 25)
		self.pads=[0, 10, 0, 0] # Padding (top, bottom, left, right). Please adjust to include chin at least
		self.face_det_batch_size=16 # Batch size for face detection
		self.wav2lip_batch_size=128 # Batch size for Wav2Lip model(s)
		self.resize_factor=1 # Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p
		self.crop=[0, -1, 0, -1] # Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg.
		# Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width
		self.box=[-1, -1, -1, -1] # Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.
		# Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
		self.rotate=False # Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.
		# Use if you get a flipped result, despite feeding a normal looking video
		self.nosmooth=False # Prevent smoothing face detections over a short temporal window'

		self.img_size = 96

		self.video_num = 2
	
args = CLI_Parser()
model = None
checkpoint = None
detector = None

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA IS " ,torch.cuda.is_available())
print('Using {} for inference.'.format(device))
print('Current cuda device:', torch.cuda.current_device())
num_gpus = torch.cuda.device_count()
print('Count of using GPUs:', num_gpus)

# video load
# param org
full_frames = []
fps = args.fps
# param custom
full_frames_ch = {}
face_det_ch = {}

if num_gpus > 0:
    print("사용 가능한 GPU 인덱스:")
    for gpu_idx in range(num_gpus):
        print(f"GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")


def addparser(model_path, face_path, audio_path, v_num):
	args.checkpoint_path = model_path
	args.face = face_path
	args.audio = audio_path
	args.video_num = v_num

	## 추가
	root = os.getcwd()
	res_path = os.path.join(root,"Wav2Lip", "results", "result_voice.mp4")  #'results/result_voice.mp4'
	args.outfile = res_path

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

# 추가 미리 로드하기
def load_detector():
	global detector
	print("called load_detector function")
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

def face_detect(images):
	global detector
	#print("facedetection====",face_detection)
	#detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
	#										flip_input=False, device=device)
	
	print(detector)
	if detector is None:
		print("detetor is None")
		load_detector()

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	# 모델 한 번만 로드하게 주석처리함.
	#del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	# 추가
	frames = frames[:len(mels)]
	face_det_results = face_det_ch[args.video_num]
	face_det_results = face_det_results[:len(mels)]

	'''
	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
	'''

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path):
	global checkpoint
	if checkpoint is None:
		print('===checkpoints loading====')
		if device == 'cuda':
			checkpoint = torch.load(checkpoint_path)
		else:
			checkpoint = torch.load(checkpoint_path,
									map_location=lambda storage, loc: storage)
	else:
		print('===checkpoints aleady====')
	return checkpoint

def load_model(path):
	global model, checkpoint

	if model is None:
		model = Wav2Lip()
		print("Load checkpoint from: {}".format(path))
		checkpoint = _load(path)
		s = checkpoint["state_dict"]
		new_s = {}
		for k, v in s.items():
			new_s[k.replace('module.', '')] = v
		model.load_state_dict(new_s)

		# 추가 _ 멀티 GPU
		if(device =='cuda') and (torch.cuda.device_count()>1):
			print('Multi GPU possivle', torch.cuda.device_count())
			model = nn.DataParallel(model)

		model = model.to(device)

		print('====== Loaded model =======')

	else:
		print('====aleady model loaded====')

	return model.eval()

# 추가
def load_video():
	global full_frames, fps

	root = os.getcwd()
	paths = []
	paths.append(os.path.join(root, "flask_", "static", "video", "ang.mp4"))
	paths.append(os.path.join(root, "flask_", "static", "video", "hap.mp4"))
	paths.append(os.path.join(root, "flask_", "static", "video", "neu.mp4"))
	paths.append(os.path.join(root, "flask_", "static", "video", "sad.mp4"))

	for i in range(len(paths)):

		if not os.path.isfile(paths[i]):
			print("none file path")
			print(paths[i], args.audio)
			raise ValueError('--face argument must be a valid path to video/image file')

		elif paths[i].split('.')[1] in ['jpg', 'png', 'jpeg']:
			full_frames = [cv2.imread(paths[i])]
			fps = args.fps

		else:
			video_stream = cv2.VideoCapture(paths[i])
			fps = video_stream.get(cv2.CAP_PROP_FPS)

			print('Reading video frames...')

			temp = []
			while 1:
				still_reading, frame = video_stream.read()
				if not still_reading:
					video_stream.release()
					break
				if args.resize_factor > 1:
					frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

				if args.rotate:
					frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

				y1, y2, x1, x2 = args.crop
				if x2 == -1: x2 = frame.shape[1]
				if y2 == -1: y2 = frame.shape[0]

				frame = frame[y1:y2, x1:x2]

				temp.append(frame)
			full_frames_ch[i] = temp
		print ("Number of frames available for inference: "+str(len(temp)))
	print("Number of Video," ,len(full_frames_ch))

	for i in range(len(paths)):
		if args.box[0] == -1:
			if not args.static:
				face_det_results = face_detect(full_frames_ch[i]) # BGR2RGB for CNN face detection
			else:
				face_det_results = face_detect([full_frames_ch[i][0]])
		else:
			print('Using the specified bounding box instead of face detection...')
			y1, y2, x1, x2 = args.box
			face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames_ch[i]]
		
		face_det_ch[i] = face_det_results
	print("Number of face det ch, " ,len(face_det_ch))


def get_loaded_video(idx):
	
	print(full_frames_ch)

	if idx in ( E_emo.angry.value , E_emo.disgust.value):
		idx = 0
	elif idx == E_emo.surprise.value :
		idx = 2

	return full_frames_ch[idx]

def main():
	global full_frames, fps
	# video load 삭제

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	# 교체 
	full_frames = get_loaded_video(args.video_num)
	#full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			##### 원래 모델 로드 하는 부분 =============
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]

			## 추가
			root = os.getcwd()
			res_path = os.path.join(root,"Wav2Lip", "temp", "result.avi")  #'temp/result.avi'

			out = cv2.VideoWriter(res_path, 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, res_path, args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

	# 추가 결과 파일 Flask로 이동
	if os.path.exists(args.outfile):
		src_path = args.outfile
		dst_path = os.path.join("flask_","static","video", "result_voice.mp4")
		shutil.copy(src_path, dst_path)
		print("Result video copied to:", dst_path)


if __name__ == '__main__':
	main()
