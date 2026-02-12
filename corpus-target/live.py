from signalflow import *
import numpy as np
import soundfile as sf
import time
import json
import librosa
import joblib

from stable_baselines3 import A2C, PPO, TD3, SAC
import gymnasium as gym
from environment import SynthGenEnv

from synthesizers import *


def computeFeaturesCorpus(signal, features4training, sample_rate, FFT_window_size, hop_size):

    features = []
    mfcc_N = 13
    chroma_N = 12
    if 'rms' in features4training:
        rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
        features.append(rms)
    if 'pitch' in features4training:
        pitch = librosa.yin(signal, fmin=100, fmax=22000, sr=sample_rate, frame_length=FFT_window_size, hop_length=hop_size)
        features.append(pitch.reshape(1,-1))
    if 'cent' in features4training:
        cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=FFT_window_size, hop_length=hop_size)
        features.append(cent)
    if 'flatness' in features4training:
        flatness = librosa.feature.spectral_flatness(y=signal, n_fft=FFT_window_size, hop_length=hop_size)
        features.append(flatness)
    if 'rolloff' in features4training:
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=FFT_window_size, hop_length=hop_size)
        features.append(rolloff)
    if 'MFCC' in features4training:
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=mfcc_N, n_fft=FFT_window_size, hop_length=hop_size)
        for mfcc_component in mfcc.tolist():
            features.append(np.array(mfcc_component).reshape(1,-1))
    if 'chroma' in features4training:
        chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate, n_chroma=chroma_N, n_fft=FFT_window_size, hop_length=hop_size)
        for chroma_component in chroma.tolist():
            features.append(np.array(chroma_component).reshape(1,-1))
    features = np.concatenate(features, axis=0)
    return features


if __name__ == "__main__":

	# load model
	MODEL_NAME = '2025-12-04/1764862455-TD3'
	log_dir = f'RL_continuous/00_model_logs/{MODEL_NAME}'

	with open(f'{log_dir}/training_config.json', 'r') as f:
		training_parameters = json.load(f)

	sr = training_parameters["sample_rate"]
	window_size = training_parameters["FFT_window_size"]
	hop_size = training_parameters["hop_size"]

	config = AudioGraphConfig()
	config.sample_rate = sr
	config.output_buffer_size = 512
	config.output_device_name = "UltraLite-mk5"
	config.input_device_name = "UltraLite-mk5"
	graph = AudioGraph(config)
	# graph.poll(2)
	audio_in = AudioIn() * 3
	right_output = StereoPanner(audio_in, 1)

	if training_parameters["synth_type"] == 'Benjolin':
		synth = Benjolin()
	elif training_parameters["synth_type"] == 'FM':
		synth = FM()
	elif training_parameters["synth_type"] == 'Granulator':
		synth = Granular()
	else:
		synth = Theremin()


	N_params = len(synth.inputs)
	synth_parameters = np.zeros(N_params).tolist()
	print(synth_parameters)
	for i, param_name in enumerate(list(synth.inputs.keys())):
		current_p = synth.inputs[param_name].value
		# p_fade = Line(current_p, synth_parameters[i], synth.fade_time)
		synth.set_input(param_name, synth_parameters[i]) 

	# graph.play([synth, audio_in]) 
	graph.play([audio_in, synth]) 


	corpus_scaler = joblib.load(f'{log_dir}/corpus_scaler.pkl')
	synth_scaler = joblib.load(f'{log_dir}/synth_scaler.pkl')

	# create env
	# define environment
	env = gym.make(
		training_parameters["environment"],
		corpus_path=training_parameters["corpus_csv_path"],
		features=training_parameters["features"],
		synth_type=training_parameters["synth_type"],
		sample_rate = training_parameters["sample_rate"],
		FFT_window_size = training_parameters["FFT_window_size"],
		hop_size = training_parameters["hop_size"],
		max_episode_duration=training_parameters["max_episode_duration"],
		param_in_state=training_parameters["param_in_state"],
		feature_difference=training_parameters["feature_difference"],
		synths_info_dir='RL_continuous/01_synthesizers',
		save_folder=log_dir,
		train=False,
		render_mode=None
	)
	print(env.observation_space)

	# load model
	AGENT_TYPE = training_parameters["AGENT_TYPE"]
	MODEL_DIR = f'{log_dir}/best'
	if AGENT_TYPE == 'PPO':
		model = PPO.load(MODEL_DIR, env=env)
	elif AGENT_TYPE == 'A2C':
		model = A2C.load(MODEL_DIR, env=env)
	elif AGENT_TYPE == 'TD3':
		model = TD3.load(MODEL_DIR, env=env)
	elif AGENT_TYPE == 'SAC':
		model = SAC.load(MODEL_DIR, env=env)


	synth_buffer = Buffer(1, window_size)
	input_buffer = Buffer(1, window_size)

	# add delay lines for responses (how to compensate for latency?)
	min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
	graph.add_node(FeedbackBufferWriter(synth_buffer, synth, min_feedback_writer_delay))
	graph.add_node(FeedbackBufferWriter(input_buffer, audio_in, min_feedback_writer_delay))


	update_interval_s = hop_size / sr
	input_sound = input_buffer.data[0,:]
	# make environment for live
	prev_audioin_features = computeFeaturesCorpus(input_sound, training_parameters["features"], sr, window_size, hop_size)
	prev_audioin_features = corpus_scaler.transform(np.array(prev_audioin_features)[:,-1].reshape(1, -1))
	time.sleep(update_interval_s)
	while True:
		input_sound = input_buffer.data[0,:]
		audioin_features = computeFeaturesCorpus(input_sound, training_parameters["features"], sr, window_size, hop_size)
		audioin_features = corpus_scaler.transform(np.array(audioin_features)[:,-1].reshape(1, -1))

		if training_parameters["param_in_state"]:
			if training_parameters["feature_difference"]:
				observation = np.concatenate((np.array(audioin_features).reshape(-1) - np.array(prev_audioin_features).reshape(-1),
											np.array(synth_parameters).reshape(-1))).astype(np.float32)
			else:
				observation = np.concatenate((np.array(audioin_features).reshape(-1), 
											np.array(prev_audioin_features).reshape(-1),
											np.array(synth_parameters).reshape(-1))).astype(np.float32)
		else:
			if training_parameters["feature_difference"]:
				observation = (np.array(audioin_features).reshape(-1) - np.array(prev_audioin_features).reshape(-1)).astype(np.float32)
			else:
				observation = np.concatenate((np.array(audioin_features).reshape(-1), 
											np.array(prev_audioin_features).reshape(-1))).astype(np.float32)


		action, _ = model.predict(observation=observation, deterministic=True)
		# print(audioin_features)
		# print(action)
		synth_parameters += action
		synth_parameters = np.clip(synth_parameters, a_min=0, a_max=1)

		# update synth parameters
		# synth_parameters = np.random.rand(N_params).tolist()
		print('-' * 20)
		for i, param_name in enumerate(list(synth.inputs.keys())):
			print(f'{param_name}: {synth.inputs[param_name].value:.3f}')
			# current_p = synth.inputs[param_name].value
			# p_fade = Line(current_p, synth_parameters[i], synth.fade_time)
			synth.set_input(param_name, synth_parameters[i]) 

		prev_audioin_features = audioin_features
		time.sleep(update_interval_s)



