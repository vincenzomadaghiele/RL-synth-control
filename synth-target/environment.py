import os
import time
import json
import librosa
import random
import itertools
import joblib
import numpy as np
import soundfile as sf
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import torch
# import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

from synthesizers import *


register(
	id="synthgen-v0",
	entry_point="environment:SynthGenEnv",
)

def getFeatureNames(features, mfcc_N=13, chroma_N=12):
	feature_names = [feat for feat in features if feat not in ['MFCC', 'chroma']]
	if 'MFCC' in features:
		for i in range(mfcc_N):
			feature_names.append(f'MFCC{i}')
	if 'chroma' in features:
		for i in range(chroma_N):
			feature_names.append(f'chroma{i}')
	return feature_names

def computeFeatures(signal, 
					features4training=['rms', 'pitch', 'cent', 'flatness', 'rolloff', 'MFCC', 'chroma'], 
					sample_rate=44100, 
					FFT_window_size=2048, 
					hop_size=1024, 
					mfcc_N=13, 
					chroma_N=12):
	features = []
	if 'rms' in features4training:
		rms = librosa.feature.rms(y=signal, 
									frame_length=FFT_window_size, 
									hop_length=hop_size)
		features.append(rms)
	if 'pitch' in features4training:
		pitch = librosa.yin(signal, 
							fmin=100, 
							fmax=22000, 
							sr=sample_rate, 
							frame_length=FFT_window_size, 
							hop_length=hop_size)
		features.append(pitch.reshape(1,-1))
	if 'cent' in features4training:
		cent = librosa.feature.spectral_centroid(y=signal, 
										   sr=sample_rate, 
										   n_fft=FFT_window_size, 
										   hop_length=hop_size)
		features.append(cent)
	if 'flatness' in features4training:
		flatness = librosa.feature.spectral_flatness(y=signal, 
													n_fft=FFT_window_size, 
													hop_length=hop_size)
		features.append(flatness)
	if 'rolloff' in features4training:
		rolloff = librosa.feature.spectral_rolloff(y=signal, 
													sr=sample_rate, 
													n_fft=FFT_window_size, 
													hop_length=hop_size)
		features.append(rolloff)
	if 'MFCC' in features4training:
		mfcc = librosa.feature.mfcc(y=signal, 
									sr=sample_rate, 
									n_mfcc=mfcc_N, 
									n_fft=FFT_window_size, 
									hop_length=hop_size)
		for mfcc_component in mfcc.tolist():
			features.append(np.array(mfcc_component).reshape(1,-1))
	if 'chroma' in features4training:
		chroma = librosa.feature.chroma_stft(y=signal, 
											sr=sample_rate,
											n_chroma=chroma_N, 
											n_fft=FFT_window_size, 
											hop_length=hop_size)
		for chroma_component in chroma.tolist():
			features.append(np.array(chroma_component).reshape(1,-1))
	features = np.concatenate(features, axis=0) * rms
	return features.T

def compensateSignalLoudness(signal, FFT_window_size, hop_size):
	rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
	if rms.max() < 0.7: # normalize amplitude
		signal *= 0.6/signal.max()
	return signal

def generateSynthEvenlySampledFeatures(synth, 
										N_synth_parameters_src, 
										N_synth_parameters_tgt, 
										features, 
										param_step_for_corpus_gen=10,
										sample_rate=44100, 
										FFT_window_size=2048, 
										hop_size=1024, 
										mfcc_N=13, 
										chroma_N=12):
	# generate synth normalizer by evenly sampling parameters
	print('Sampling synth for normalization...')
	elements = np.array(range(0, param_step_for_corpus_gen)) / param_step_for_corpus_gen
	permutations = [p for p in itertools.product(elements, repeat=N_synth_parameters_src)]
	synth_samples_features_src = []
	for param_list in permutations:
		param_list = np.array(param_list).tolist()
		synth.resetParametersSRC(param_list)
		window_array_src, _ = synth.forward()
		features_values_src = computeFeatures(window_array_src, features,
											sample_rate=sample_rate, 
											FFT_window_size=FFT_window_size, 
											hop_size=hop_size, 
											mfcc_N=mfcc_N, 
											chroma_N=chroma_N).mean(axis=0)
		synth_samples_features_src.append(features_values_src)
	synth_feats_df_src = pd.DataFrame(data=np.array(synth_samples_features_src), columns=getFeatureNames(features, mfcc_N, chroma_N))
	indixes = []
	for p in permutations:
		indixes.append('-'.join([str(pp) for pp in p]))
	synth_feats_df_src['parameters'] = indixes
	synth_feats_df_src = synth_feats_df_src.set_index('parameters')
	# compute target
	permutations = [p for p in itertools.product(elements, repeat=N_synth_parameters_tgt)]
	synth_samples_features_tgt = []
	for param_list in permutations:
		param_list = np.array(param_list).tolist()
		synth.resetParametersTGT(param_list)
		_, window_array_tgt = synth.forward()
		features_values_src = computeFeatures(window_array_tgt, features,
											sample_rate=sample_rate, 
											FFT_window_size=FFT_window_size, 
											hop_size=hop_size, 
											mfcc_N=mfcc_N, 
											chroma_N=chroma_N).mean(axis=0)
		synth_samples_features_tgt.append(features_values_src)
	synth_feats_df_tgt = pd.DataFrame(data=np.array(synth_samples_features_tgt), columns=getFeatureNames(features, mfcc_N, chroma_N))
	indixes = []
	for p in permutations:
		indixes.append('-'.join([str(pp) for pp in p]))
	synth_feats_df_tgt['parameters'] = indixes
	synth_feats_df_tgt = synth_feats_df_tgt.set_index('parameters')
	return synth_feats_df_src, synth_feats_df_tgt


class SynthGenEnv(gym.Env):
	metadata = {"render_modes": ["human"], "render_fps": 4}

	def __init__(
		self,
		features: list,
		src_synth_type='Theremin',
		tgt_synth_type='Theremin',
		sample_rate=44100,
		FFT_window_size=2048,
		hop_size=512,
		max_episode_duration=3000,
		N_steps_memory=10,
		N_reward_memory=4,
		step_size=1,
		rewards = ['featureDifference'],
		target_type = 'synth', # synth OR corpus
		normalization_mode = 'source',
		episode_mode = 'static',
		synths_info_dir='01_synthesizers',
		save_folder='00_model_logs',
		seed=None,
		train=True,
		render_mode=None
	):
		## TODO:
		# - AUTOMATE TARGET PARAMETER CHANGE
		# - AUDIO TRACK INPUT

		## GLOBAL ENV PARAMETERS
		self.src_synth_type = src_synth_type
		self.tgt_synth_type = tgt_synth_type
		self.sample_rate = sample_rate
		self.FFT_window_size = FFT_window_size
		self.hop_size = hop_size
		self.render_mode = render_mode
		self.max_episode_duration = max_episode_duration
		self.seed = seed
		self.save_folder = save_folder
		self.train = train

		# DEFINE FEATURES
		self.mfcc_N = 13
		self.chroma_N = 12
		# normalization_mode = 'source' # one of: ['source', 'target', 'shared', 'separate']
		self.rewards = rewards # a list of ['mrSTFT', 'featureDifference', 'featureDistance', 'RMSdifference']
		self.episode_mode = episode_mode # one of: ['static', 'moving_target', 'corpus']

		# DEFINE ALL FEATURES FOR NORMALIZATION TABLE COMPUTATION
		self.allfeatures = ['rms', 'pitch', 'cent', 'flatness', 'rolloff', 'MFCC', 'chroma']
		self.all_feature_names = getFeatureNames(self.allfeatures, self.mfcc_N, self.chroma_N)

		# DEFINE FEATURES TO USE DURING TRAINING
		self.features4training = features
		self.feature_names = getFeatureNames(features, self.mfcc_N, self.chroma_N)

		# load synth
		if self.train:
			self.synth = DoubleSynth(src_synth_type, tgt_synth_type, self.sample_rate, self.FFT_window_size, self.hop_size)
			self.N_src_synth_parameters = len(list(self.synth.getInputsSRC().keys()))
			self.N_tgt_synth_parameters = len(list(self.synth.getInputsTGT().keys()))
			print()
			print(f"Initializing source synth {src_synth_type} with {self.N_src_synth_parameters} parameters...")
			print(f"Initializing target synth {tgt_synth_type} with {self.N_tgt_synth_parameters} parameters...")
			print()
			# initialize synthesizer parameter values
			src_parameter_values = [self.np_random.random() for _ in range(self.N_src_synth_parameters)]
			self.synth.resetParametersSRC(src_parameter_values)
			tgt_parameter_values = [self.np_random.random() for _ in range(self.N_tgt_synth_parameters)]
			self.synth.resetParametersTGT(tgt_parameter_values)

			# search for precomputed scaler, if not generate scaler
			if not os.path.isfile(f'{synths_info_dir}/{src_synth_type}/{src_synth_type}.csv'):
				if not os.path.isdir(f'./{synths_info_dir}/{src_synth_type}'):
					os.makedirs(f'./{synths_info_dir}/{src_synth_type}')
				synth_feats_df_src, synth_feats_df_tgt = generateSynthEvenlySampledFeatures(self.synth, self.N_src_synth_parameters, 
																			self.N_tgt_synth_parameters, self.allfeatures, 
																			param_step_for_corpus_gen=10, sample_rate=self.sample_rate, 
																			FFT_window_size=int(self.FFT_window_size/2), hop_size=int(self.hop_size/2), 
																			mfcc_N=self.mfcc_N, chroma_N=self.chroma_N)
				synth_feats_df_src.to_csv(f'{synths_info_dir}/{src_synth_type}/{src_synth_type}.csv')
				synth_feats_df_tgt.to_csv(f'{synths_info_dir}/{tgt_synth_type}/{tgt_synth_type}.csv')
			else:
				# load pre-computed dataset
				synth_feats_df_src = pd.read_csv(f'{synths_info_dir}/{src_synth_type}/{src_synth_type}.csv')
				synth_feats_df_src = synth_feats_df_src.set_index('parameters')
				synth_feats_df_tgt = pd.read_csv(f'{synths_info_dir}/{tgt_synth_type}/{tgt_synth_type}.csv')
				synth_feats_df_tgt = synth_feats_df_tgt.set_index('parameters')
			
			# make scaler to normalize features
			if normalization_mode == 'source':
				synth_feats_df_src = synth_feats_df_src[self.feature_names]
				synth_samples_features = synth_feats_df_src.values
				self.src_synth_scaler = StandardScaler()
				self.src_synth_scaler.fit(np.array(synth_samples_features))
				self.tgt_synth_scaler = self.src_synth_scaler
			elif normalization_mode == 'target':
				synth_feats_df_tgt = synth_feats_df_tgt[self.feature_names]
				synth_samples_features = synth_feats_df_tgt.values
				self.tgt_synth_scaler = StandardScaler()
				self.tgt_synth_scaler.fit(np.array(synth_samples_features))
				self.src_synth_scaler = self.tgt_synth_scaler
			elif normalization_mode == 'shared':
				synth_feats_df_src = synth_feats_df_src[self.feature_names]
				synth_src_features = synth_feats_df_src.values
				synth_feats_df_tgt = synth_feats_df_tgt[self.feature_names]
				synth_tgt_features = synth_feats_df_tgt.values
				synth_samples_features = np.concatenate((synth_src_features, synth_tgt_features), axis=1)
				self.src_synth_scaler = StandardScaler()
				self.src_synth_scaler.fit(np.array(synth_samples_features))
				self.tgt_synth_scaler = self.src_synth_scaler
			elif normalization_mode == 'separate':
				synth_feats_df_src = synth_feats_df_src[self.feature_names]
				synth_samples_features = synth_feats_df_src.values
				self.src_synth_scaler = StandardScaler()
				self.src_synth_scaler.fit(np.array(synth_samples_features))
				synth_feats_df_tgt = synth_feats_df_tgt[self.feature_names]
				synth_samples_features = synth_feats_df_tgt.values
				self.tgt_synth_scaler = StandardScaler()
				self.tgt_synth_scaler.fit(np.array(synth_samples_features))
			joblib.dump(self.src_synth_scaler, f'{self.save_folder}/src_synth_scaler.pkl')
			joblib.dump(self.tgt_synth_scaler, f'{self.save_folder}/tgt_synth_scaler.pkl')

		else:
			# MOVE THESE TO JSON
			if src_synth_type == 'Benjolin':
				self.N_src_synth_parameters = 4
			elif src_synth_type == 'Granular':
				self.N_src_synth_parameters = 4
			elif src_synth_type == 'FM':
				self.N_src_synth_parameters = 8
			else:
				self.N_src_synth_parameters = 2
			if tgt_synth_type == 'Benjolin':
				self.N_tgt_synth_parameters = 4
			elif tgt_synth_type == 'Granular':
				self.N_tgt_synth_parameters = 4
			elif tgt_synth_type == 'FM':
				self.N_tgt_synth_parameters = 8
			else:
				self.N_tgt_synth_parameters = 2


		# SET ENVIRONMENT CONFIGURATIONS
		self.N_features = len(self.feature_names)
		self.step_size = step_size # max param change per step
		# action space
		self.action_space = spaces.Box(
			low = -self.step_size, 
			high = self.step_size, 
			shape = (self.N_src_synth_parameters,), 
			dtype = np.float32
		)
		# observation space
		self.observation_space = spaces.Box(
			low = -np.inf,
			high = np.inf,
			shape = (self.N_features * 3 + self.N_src_synth_parameters,),
			dtype = np.float32,
		)

		# # observation space
		# self.observation_space = spaces.Box(
		# 	low = -np.inf,
		# 	high = np.inf,
		# 	shape = (self.N_features * 2 * N_steps_memory + self.N_src_synth_parameters,),
		# 	dtype = np.float32,
		# )


		# initialize state variables
		self.target_features = np.zeros((self.N_features,))
		self.previous_target_features = np.zeros((self.N_features,))
		self.previous_synth_features = np.zeros((self.N_features,))
		self.synthesis_parameters = np.zeros((self.N_src_synth_parameters,))
		self.src_synthesis_parameters = np.zeros((self.N_src_synth_parameters,))
		self.episode_duration = max_episode_duration
		if self.episode_mode == 'moving_target':
			self.fading = False
			self.fade_counter = 0
		print(f'max episode duration: {self.max_episode_duration}')

		self.N_steps_memory = N_steps_memory
		self.N_reward_memory = N_reward_memory
		self.N_memory_size = max(self.N_steps_memory, self.N_reward_memory) + 1 + 2
		self.memory_src = np.zeros(self.N_memory_size*self.hop_size)
		self.memory_tgt = np.zeros(self.N_memory_size*self.hop_size)

		if self.render_mode == "human":
			self.src_synth_audiotrack = np.zeros([librosa.frames_to_samples(self.max_episode_duration, 
																			hop_length=self.hop_size, 
																			n_fft=self.FFT_window_size)], dtype=float, order='C')
			self.tgt_synth_audiotrack = np.zeros([librosa.frames_to_samples(self.max_episode_duration, 
																   			hop_length=self.hop_size, 
																			n_fft=self.FFT_window_size)], dtype=float, order='C')
			self.episode_src_features_list = []
			self.episode_tgt_features_list = []
			self.episode_src_parameters_list = []
			self.episode_tgt_parameters_list = []

			self.episode_evaluation = {}
			self.mrSTFT_values = []
			self.dMSE_values = []
			self.diffMSE_values = []

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		print()
		print('Starting episode')
		print('-'*50)
		# RESET GRAPH
		self.synth.resetGraph()
		
		self.memory_src = np.zeros(self.N_memory_size*self.hop_size)
		self.memory_tgt = np.zeros(self.N_memory_size*self.hop_size)

		# INITIALIZE SOURCE AND TARGET SYNTH
		self.tgt_synthesis_parameters = np.array([self.np_random.random() for _ in range(self.N_tgt_synth_parameters)])
		self.synth.resetParametersTGT(self.tgt_synthesis_parameters.tolist())
		self.src_synthesis_parameters = np.array([self.np_random.random() for _ in range(self.N_src_synth_parameters)])
		self.synth.resetParametersSRC(self.src_synthesis_parameters.tolist())
		src_synth_sound, tgt_synth_sound = self.synth.forward()
		self.memory_src[:-self.hop_size] = self.memory_src[self.hop_size:] 
		self.memory_src[-self.hop_size:] = src_synth_sound[-self.hop_size:] 
		self.memory_tgt[:-self.hop_size] = self.memory_tgt[self.hop_size:]
		self.memory_tgt[-self.hop_size:] = tgt_synth_sound[-self.hop_size:]
		previous_tgt_features = computeFeatures(self.memory_tgt, self.features4training, sample_rate=self.sample_rate, 
												FFT_window_size=self.FFT_window_size, hop_size=self.hop_size, 
												mfcc_N=self.mfcc_N, chroma_N=self.chroma_N)
		self.previous_tgt_features = self.tgt_synth_scaler.transform(previous_tgt_features)
		self.tgt_synthesis_parameters = np.array([self.np_random.random() for _ in range(self.N_tgt_synth_parameters)])
		print(f'Objective parameters: {self.tgt_synthesis_parameters}')
		self.synth.fadeParametersTGT(self.tgt_synthesis_parameters.tolist())
		src_synth_sound, tgt_synth_sound = self.synth.forward()
		self.memory_src[:-self.hop_size] = self.memory_src[self.hop_size:] 
		self.memory_src[-self.hop_size:] = src_synth_sound[-self.hop_size:] 
		self.memory_tgt[:-self.hop_size] = self.memory_tgt[self.hop_size:]
		self.memory_tgt[-self.hop_size:] = tgt_synth_sound[-self.hop_size:]
		tgt_features = computeFeatures(self.memory_tgt, self.features4training,sample_rate=self.sample_rate, 
										FFT_window_size=self.FFT_window_size, hop_size=self.hop_size, 
										mfcc_N=self.mfcc_N, chroma_N=self.chroma_N)
		self.tgt_features = self.tgt_synth_scaler.transform(tgt_features)
		previous_src_features = computeFeatures(self.memory_src, self.features4training,sample_rate=self.sample_rate, 
												FFT_window_size=self.FFT_window_size, hop_size=self.hop_size, 
												mfcc_N=self.mfcc_N, chroma_N=self.chroma_N)
		self.previous_src_features = self.tgt_synth_scaler.transform(previous_src_features)
		self.this_episode_influence = tgt_synth_sound

		# CONSTRUCT OBSERVATION
		obs = np.concatenate((
				self.tgt_features[-2,:].reshape(-1),
				self.previous_tgt_features[-2,:].reshape(-1),
				self.previous_src_features[-2,:].reshape(-1),
				self.src_synthesis_parameters))

		# # update observed state
		# obs = np.concatenate((self.tgt_features[1:-1,:][-self.N_steps_memory:,:].reshape(-1), 
		# 					# self.previous_tgt_features[1:-1,:][-self.N_steps_memory:,:].reshape(-1), 
		# 					self.previous_src_features[1:-1,:][-self.N_steps_memory:,:].reshape(-1),
		# 					self.src_synthesis_parameters))


		if self.episode_mode == 'moving_target':
			self.fading = False
			self.fade_counter = 0

		info = {}
		self.episode_step = 0
		self.cumulative_reward = 0
		if self.render_mode == "human":
			curr_savedir = f'{self.save_folder}/evaluation/{int(time.time())}'
			os.makedirs(f'{curr_savedir}/sounds', exist_ok=True)
			os.makedirs(f'{curr_savedir}/plots', exist_ok=True)

			# save output from previous episode
			out_signal = np.concatenate((self.src_synth_audiotrack.reshape(-1,1), self.tgt_synth_audiotrack.reshape(-1,1)), axis=1)
			sf.write(f'{curr_savedir}/sounds/SRC_{self.src_synth_type}.wav', self.src_synth_audiotrack, self.sample_rate, subtype='PCM_24')
			sf.write(f'{curr_savedir}/sounds/TGT_{self.tgt_synth_type}.wav', self.tgt_synth_audiotrack, self.sample_rate, subtype='PCM_24')
			sf.write(f'{curr_savedir}/sounds/{self.src_synth_type}-{self.tgt_synth_type}.wav', out_signal, self.sample_rate, subtype='PCM_24')			
			
			# PLOT FEATURE MATCHINGS AND SAVE
			fig, ax = plt.subplots(2, figsize=(20, 8))
			librosa.display.waveshow(y=self.src_synth_audiotrack, sr=self.sample_rate, x_axis='time', ax=ax[0], alpha=0.5)
			librosa.display.waveshow(y=self.tgt_synth_audiotrack, sr=self.sample_rate, x_axis='time', ax=ax[1], alpha=0.5)
			ax[0].set_xlabel('time [s]')
			ax[0].set_title(f'SRC_{self.src_synth_type}')
			ax[1].set_xlabel('time [s]')
			ax[1].set_title(f'TGT_{self.tgt_synth_type}')
			plt.savefig(f'{curr_savedir}/plots/{self.tgt_synth_type}.png')
			self.episode_src_features_list = np.array(self.episode_src_features_list)
			self.episode_tgt_features_list = np.array(self.episode_tgt_features_list)
			if self.episode_src_features_list.shape[0] > 0 and self.episode_tgt_features_list.shape[0] > 0:
				for kk, feat_name in enumerate(self.feature_names):
					fig, ax = plt.subplots(1, figsize=(15, 4))
					ax.plot(self.episode_src_features_list[50:,kk], label='src')
					ax.plot(self.episode_tgt_features_list[50:,kk], label='tgt')
					ax.set_xlabel('steps')
					ax.set_ylabel(feat_name)
					ax.set_title(f'{feat_name}')
					# ax.set_ylim([-2,2])
					ax.legend()
					plt.savefig(f'{curr_savedir}/plots/{feat_name}.png')
			self.episode_src_parameters_list = np.array(self.episode_src_parameters_list)
			self.episode_tgt_parameters_list = np.array(self.episode_tgt_parameters_list)
			if self.episode_src_parameters_list.shape[0] > 0 and self.episode_tgt_parameters_list.shape[0] > 0:
				for kk in range(max(self.N_src_synth_parameters, self.N_tgt_synth_parameters)):
					fig, ax = plt.subplots(1, figsize=(15, 4))
					if kk < self.N_src_synth_parameters:
						ax.plot(self.episode_src_parameters_list[50:,kk], label='src')
					if kk < self.N_tgt_synth_parameters:
						ax.plot(self.episode_tgt_parameters_list[50:,kk], label='tgt')
					ax.set_xlabel('steps')
					ax.set_ylabel(f'p{kk}')
					ax.set_title(f'p{kk}')
					ax.set_ylim([0,1])
					ax.legend()
					plt.savefig(f'{curr_savedir}/plots/p{kk}.png')

			# initialize new arrays
			self.src_synth_audiotrack = np.zeros([librosa.frames_to_samples(self.max_episode_duration, 
																			hop_length=self.hop_size, 
																			n_fft=self.FFT_window_size)], dtype=float, order='C')
			self.tgt_synth_audiotrack = np.zeros([librosa.frames_to_samples(self.max_episode_duration, 
																   			hop_length=self.hop_size, 
																			n_fft=self.FFT_window_size)], dtype=float, order='C')
			self.episode_src_features_list = []
			self.episode_tgt_features_list = []
			self.episode_src_parameters_list = []
			self.episode_tgt_parameters_list = []

			self.episode_evaluation['mrSTFT_mean'] = np.array(self.mrSTFT_values).mean()
			self.episode_evaluation['mrSTFT_std'] = np.array(self.mrSTFT_values).std()
			self.episode_evaluation['dMSE_mean'] = np.array(self.dMSE_values).mean()
			self.episode_evaluation['dMSE_std'] = np.array(self.dMSE_values).std()
			self.episode_evaluation['diffMSE_mean'] = np.array(self.diffMSE_values).mean()
			self.episode_evaluation['diffMSE_std'] = np.array(self.diffMSE_values).std()
			with open(f'{curr_savedir}/evaluation.json', 'w', encoding='utf-8') as f:
				json.dump(self.episode_evaluation, f, ensure_ascii=False, indent=4)

			self.episode_evaluation = {}
			self.mrSTFT_values = []
			self.dMSE_values = []
			self.diffMSE_values = []

			self.render()
		return np.array(obs.tolist(), dtype=np.float32), info

	def step(self, action):
		# perform action --> get f(g(theta_pre + theta_mod))
		self.src_synthesis_parameters += action
		self.src_synthesis_parameters = np.clip(self.src_synthesis_parameters, a_min=0, a_max=1)
		# save latest target feats
		self.previous_tgt_features = self.tgt_features
		# compute sound
		self.synth.fadeParametersSRC(self.src_synthesis_parameters.tolist())
		if self.episode_mode == 'moving_target':
			if self.np_random.random() < 0.2 and not self.fading:
				if self.np_random.random() < 0.5:
					num_frames_fade = np.random.randint(low=0, high=100)
					self.tgt_synthesis_parameters = np.array([self.np_random.random() for _ in range(self.N_tgt_synth_parameters)])
					self.synth.fadeParametersTGT(self.tgt_synthesis_parameters.tolist(), fade_time=num_frames_fade*self.hop_size/self.sample_rate)
					self.fade_counter = num_frames_fade
					self.fading = True
				else:
					self.tgt_synthesis_parameters = np.array([self.np_random.random() for _ in range(self.N_tgt_synth_parameters)])
					self.synth.fadeParametersTGT(self.tgt_synthesis_parameters.tolist())
			elif self.fading:
				self.fade_counter -= 1
				if self.fade_counter <= 0:
					self.fading = False
		synth_sound_src, synth_sound_tgt = self.synth.forward()
		# update sound memory
		self.memory_src[:-self.hop_size] = self.memory_src[self.hop_size:] 
		self.memory_src[-self.hop_size:] = synth_sound_src[-self.hop_size:] 
		self.memory_tgt[:-self.hop_size] = self.memory_tgt[self.hop_size:]
		self.memory_tgt[-self.hop_size:] = synth_sound_tgt[-self.hop_size:]
		# compute features
		src_features = computeFeatures(self.memory_src, self.features4training, sample_rate=self.sample_rate, 
										FFT_window_size=self.FFT_window_size, hop_size=self.hop_size, 
										mfcc_N=self.mfcc_N, chroma_N=self.chroma_N)#.mean(axis=0)
		tgt_features = computeFeatures(self.memory_tgt, self.features4training, sample_rate=self.sample_rate, 
										FFT_window_size=self.FFT_window_size, hop_size=self.hop_size, 
										mfcc_N=self.mfcc_N, chroma_N=self.chroma_N)#.mean(axis=0)
		src_features = self.src_synth_scaler.transform(src_features)
		tgt_features = self.tgt_synth_scaler.transform(tgt_features)

		# REWARDS
		# spectrum-based reward
		reward = 0
		if 'mrSTFT' in self.rewards:
			# reward += self.mrSTFTReward(self.memory_tgt[-(self.N_reward_memory+1)*self.hop_size:-self.hop_size], self.memory_src[-self.N_reward_memory*self.hop_size:])
			reward += self.mrSTFTReward(self.memory_tgt[-2*self.hop_size:-self.hop_size], self.memory_src[-self.hop_size:])
		elif 'featureDifference' in self.rewards:
			reward += self.featureDifferenceReward(self.previous_tgt_features[-2,:], src_features[-2,:])
			# reward += self.featureDifferenceReward(self.previous_tgt_features[1:-1,:][-self.N_reward_memory:,:], src_features[1:-1,:][-self.N_reward_memory:,:])
			# reward /= 2
		elif 'featureDistance' in self.rewards:
			# reward += self.featureDirectionReward(tgt_features[-(self.N_reward_memory+1):-1,:], src_features[-self.N_reward_memory:,:])
			reward += self.featureDirectionReward(self.previous_tgt_features[-2,:], src_features[-2,:])
		elif 'RMSdifference' in self.rewards:
			# reward += self.RMSDifferenceReward(tgt_features[-(self.N_reward_memory+1):-1,:], src_features[-self.N_reward_memory:,:])
			reward += self.RMSDifferenceReward(self.previous_tgt_features[-2,:], src_features[-2,:])
		reward /= len(self.rewards)
		
		self.cumulative_reward += reward

		if self.render_mode == "human":
			self.mrSTFT_values.append(self.mrSTFTReward(self.memory_tgt[-2*self.hop_size:-self.hop_size], self.memory_src[-self.hop_size:]))
			self.dMSE_values.append(self.featureDifferenceReward(self.previous_tgt_features[-2,:], src_features[-2,:]))
			self.diffMSE_values.append(self.featureDirectionReward(self.previous_tgt_features[-2,:], src_features[-2,:]))

		info = {}
		info["episode_step"] = self.episode_step
		info["action"] = [float(f"{p:.2f}") for p in action.tolist()]
		info["src_synthesis_parameters"] = [float(f"{p:.2f}") for p in self.src_synthesis_parameters.tolist()]
		info["tgt_synthesis_parameters"] = [float(f"{p:.2f}") for p in self.tgt_synthesis_parameters.tolist()]
		info["previous_synth_features"] = self.previous_src_features.tolist()
		info["src_synth_features"] = [float(f"{p:.2f}") for p in src_features[-2,:].reshape(-1).tolist()]
		info["tgt_synth_features"] = [float(f"{p:.2f}") for p in tgt_features[-2,:].reshape(-1).tolist()]
		info["previous_target_features"] = [float(f"{p:.2f}") for p in self.previous_tgt_features[-2,:].reshape(-1).tolist()]
		info["reward"] = reward
		if self.episode_step % 50 == 0:
			# print(f"Step {self.episode_step} | Reward {reward:.3f} | Action {info["action"]} | Parameters {info["src_synthesis_parameters"]} ")
			print(f'TGT feats {info["previous_target_features"]} | SRC feats {info["src_synth_features"]} | NEW TGT feats {info["tgt_synth_features"]}')
		
		# extract new target from corpus
		self.previous_src_features = src_features
		self.tgt_features = tgt_features
		
		# update observed state
		obs = np.concatenate((self.tgt_features[-2,:].reshape(-1), 
							self.previous_tgt_features[-2,:].reshape(-1), 
							self.previous_src_features[-2,:].reshape(-1),
							self.src_synthesis_parameters))

		# update observed state
		# obs = np.concatenate((self.tgt_features[1:-1,:][-self.N_steps_memory:,:].reshape(-1), 
		# 					# self.previous_tgt_features[1:-1,:][-self.N_steps_memory:,:].reshape(-1), 
		# 					self.previous_src_features[1:-1,:][-self.N_steps_memory:,:].reshape(-1),
		# 					self.src_synthesis_parameters))


		if self.render_mode == "human":
			self.src_synth_audiotrack[self.episode_step*self.hop_size:(self.episode_step+1)*self.hop_size] = synth_sound_src[-self.hop_size:]
			self.tgt_synth_audiotrack[self.episode_step*self.hop_size:(self.episode_step+1)*self.hop_size] = synth_sound_tgt[-self.hop_size:]
			self.episode_src_features_list.append(src_features[-2,:])
			self.episode_tgt_features_list.append(self.previous_tgt_features[-2,:])
			self.episode_src_parameters_list.append(self.src_synthesis_parameters.tolist())
			self.episode_tgt_parameters_list.append(self.tgt_synthesis_parameters.tolist())
			self.render()

		# update episode step
		if self.episode_step >= self.max_episode_duration - 1:
			terminated = True
			print(f'Episode ended. Total reward: {self.cumulative_reward:.3f}')
			print()
			self.synth.close()
		else:
			self.episode_step += 1
			terminated = False
		return np.array(obs.tolist(), dtype=np.float32), reward, terminated, False, info

	def render(self):
		return

	# REWARDS
	def featureDifferenceReward(self, target_state, agent_state):
		L2 = np.linalg.norm((agent_state - target_state), ord=1)
		return -L2
	def RMSDifferenceReward(self, target_state, agent_state): 
		L2 = np.linalg.norm((agent_state[:,0] - target_state[:,0]), ord=1) # assumes RMS is always first feature
		return -L2
	def featureDirectionReward(self, target_state, agent_state):
		y = target_state[1:] - target_state[:-1]
		y_hat = agent_state[1:] - agent_state[:-1]
		L2 = np.linalg.norm((y_hat - y), ord=1)
		return -L2
	def mrSTFTReward(self, y_true, y_pred):
		min_len = min(y_true.shape[0], y_pred.shape[0])
		y_true = torch.tensor(y_true[:min_len]).type(torch.FloatTensor)
		y_pred = torch.tensor(y_pred[:min_len]).type(torch.FloatTensor)
		loss = 0
		# Multiple FFT window sizes
		for n_fft in [int(self.hop_size/4), int(self.hop_size/2), self.hop_size]:
			# Spectral convergence loss
			stft_true = torch.stft(y_true, n_fft=n_fft, return_complex=True)
			stft_pred = torch.stft(y_pred, n_fft=n_fft, return_complex=True)
			sc_loss = torch.norm(stft_true - stft_pred, 'fro') / torch.norm(stft_true, 'fro')
			# Log magnitude loss
			mag_true = torch.abs(stft_true)
			mag_pred = torch.abs(stft_pred)
			logmag_loss = F.l1_loss(torch.log(mag_true + 1e-7), torch.log(mag_pred + 1e-7))
			loss += sc_loss + logmag_loss
		return max(-loss.detach().item(),-1000) # to avoid infinity in backprop




if __name__ == "__main__":

	# define environment
	training_parameters = {}
	training_parameters["environment"] = "synthgen-v0"
	training_parameters["corpus_csv_path"] = "00_corpus/00_moisesdb/bass"
	training_parameters["src_synth_type"] = "Theremin"
	training_parameters["tgt_synth_type"] = "Theremin"
	training_parameters["features"] = ['rms', 'cent']
	training_parameters["sample_rate"] = 44100
	training_parameters["FFT_window_size"] = 4096
	training_parameters["hop_size"] = 2048
	training_parameters["max_episode_duration"] = 1000
	training_parameters["N_steps_memory"] = 8
	training_parameters["N_reward_memory"] = 8
	training_parameters["step_size"] = 0.1
	training_parameters["rewards"] = ['featureDifference']
	training_parameters["normalization_mode"] = 'source'
	training_parameters["episode_mode"] = 'moving_target'

	env = gym.make(
		training_parameters["environment"],
		features=training_parameters["features"],
		src_synth_type=training_parameters["src_synth_type"],
		tgt_synth_type=training_parameters["tgt_synth_type"],
		sample_rate = training_parameters["sample_rate"],
		FFT_window_size = training_parameters["FFT_window_size"],
		hop_size = training_parameters["hop_size"],
		N_steps_memory=training_parameters["N_steps_memory"],
		N_reward_memory=training_parameters["N_reward_memory"],
		max_episode_duration=training_parameters["max_episode_duration"],
		step_size=training_parameters["step_size"],
		rewards = training_parameters["rewards"],
		normalization_mode = training_parameters["normalization_mode"],
		episode_mode = training_parameters["episode_mode"],
		synths_info_dir='RL_continuous/in-domain/01_synthesizers',
		save_folder='RL_continuous/in-domain/00_model_logs/prova_env',
		render_mode="human",
	)
	print(env.observation_space)

	# Use this to check our custom environment
	print("Check environment begin")
	check_env(env.unwrapped)
	print("Check environment end")

	# Reset environment
	obs = env.reset()[0]

	# Run 5 episodes
	count_episodes = 0
	while count_episodes < 5:
		rand_action = env.action_space.sample()
		obs, reward, terminated, _, _ = env.step(rand_action)

		if terminated:
			obs = env.reset()[0]
			count_episodes += 1
