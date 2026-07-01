from signalflow import *
import numpy as np
import soundfile as sf
import time
import json
import librosa
import joblib

from stable_baselines3 import A2C, PPO, TD3, SAC
import gymnasium as gym
from environment import SynthGenEnv, computeFeatures

from synthesizers import *



if __name__ == "__main__":

	# load model
	MODEL_NAME = '2026-06-23/1782219887-SAC'
	log_dir = f'./corpus-target/01_model_logs/{MODEL_NAME}'

	with open(f'{log_dir}/training_config.json', 'r') as f:
		training_parameters = json.load(f)

	sr = training_parameters["sample_rate"]
	window_size = training_parameters["FFT_window_size"]
	hop_size = training_parameters["hop_size"]

	config = AudioGraphConfig()
	config.sample_rate = sr
	config.output_buffer_size = 256
	# config.output_device_name = "UltraLite-mk5"
	config.input_device_name = "Scarlett 2i2 USB" 
	config.output_device_name = "Scarlett 2i2 USB"
	graph = AudioGraph(config)
	# graph.poll(2)
	audio_in = AudioIn() * 1
	# right_output = StereoPanner(audio_in, 1)

	if training_parameters["src_synth_type"] == 'Benjolin':
		synth = Benjolin(graph)
	elif training_parameters["src_synth_type"] == 'FM':
		synth = FM()
	elif training_parameters["src_synth_type"] == 'Granular':
		synth = Granular()
	else:
		synth = Theremin()
	synth.play()

	N_params = len(synth.inputs)
	synth_parameters = np.zeros(N_params).tolist()
	print(synth_parameters)
	for i, param_name in enumerate(list(synth.inputs.keys())):
		current_p = synth.inputs[param_name].value
		# p_fade = Line(current_p, synth_parameters[i], synth.fade_time)
		synth.set_input(param_name, synth_parameters[i]) 

	# graph.play([synth, audio_in]) 
	# graph.play([audio_in, synth]) 
	# graph.play(StereoPanner(audio_in, 1)) 
	# graph.play(StereoPanner(synth, -1)) 
	

	corpus_scaler = joblib.load(f'{log_dir}/tgt_synth_scaler.pkl')
	synth_scaler = joblib.load(f'{log_dir}/src_synth_scaler.pkl')

	# create env
	env = gym.make(
		training_parameters["environment"],
		features=training_parameters["features"],
		src_synth_type=training_parameters["src_synth_type"],
		corpus_path=training_parameters["corpus_path"],
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
		synths_info_dir='./corpus-target/02_synthesizers',
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


	synth_buffer = Buffer(1, window_size*3)
	input_buffer = Buffer(1, window_size*3)

	# add delay lines for responses
	min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
	graph.add_node(FeedbackBufferWriter(synth_buffer, synth, min_feedback_writer_delay))
	graph.add_node(FeedbackBufferWriter(input_buffer, audio_in, min_feedback_writer_delay))

	fade_time = hop_size / sr / 8

	update_interval_s = hop_size / sr
	input_sound = input_buffer.data[0,:]
	synth_sound = synth_buffer.data[0,:]
	# make environment for live
	# prev_audioin_features = computeFeaturesCorpus(input_sound, training_parameters["features"], sr, window_size, hop_size)
	prev_audioin_features = computeFeatures(input_sound, training_parameters["features"], sr, window_size, hop_size)
	prev_audioin_features = corpus_scaler.transform(np.array(prev_audioin_features))
	previous_src_features = computeFeatures(synth_sound, training_parameters["features"], sr, window_size, hop_size)
	prev_audioin_features = synth_scaler.transform(np.array(previous_src_features))
	time.sleep(update_interval_s)

	# graph.play([audio_in, synth]) 
	# graph.play([0,0,audio_in]) 
	# audio_in.play()
	# graph.play(StereoPanner(audio_in, 1)) 
	# StereoPanner(synth, -1).play()
	# graph.play(StereoPanner(synth, -1)) 
	# StereoPanner(synth, -1).play()
	# StereoPanner(audio_in, -1).play()

	while True:
		input_sound = input_buffer.data[0,:]
		synth_sound = synth_buffer.data[0,:]
		# audioin_features = computeFeaturesCorpus(input_sound, training_parameters["features"], sr, window_size, hop_size)
		audioin_features = computeFeatures(input_sound, training_parameters["features"], sr, window_size, hop_size)
		audioin_features = corpus_scaler.transform(np.array(audioin_features))
		previous_src_features = computeFeatures(synth_sound, training_parameters["features"], sr, window_size, hop_size)
		previous_src_features = synth_scaler.transform(np.array(previous_src_features))

		# update observed state
		observation = np.concatenate((np.array(audioin_features)[-2,:].reshape(-1), 
									np.array(prev_audioin_features)[-2,:].reshape(-1), 
									np.array(previous_src_features)[-2,:].reshape(-1),
									# np.array([0,0,0,0]).reshape(-1),
									np.array(synth_parameters).reshape(-1)))


		action, _ = model.predict(observation=observation, deterministic=True)
		# print(audioin_features)
		# # print(np.array(previous_src_features)[-2,:].reshape(-1))
		# print(action)
		synth_parameters += action
		synth_parameters = np.clip(synth_parameters, a_min=0, a_max=1)

		# update synth parameters
		# synth_parameters = np.random.rand(N_params).tolist()
		print('-' * 20)
		# for i, param_name in enumerate(list(synth.inputs.keys())):
		# 	print(f'{param_name}: {synth.inputs[param_name].value:.3f}')
		# 	# current_p = synth.inputs[param_name].value
		# 	# p_fade = Line(current_p, synth_parameters[i], synth.fade_time)
		# 	synth.set_input(param_name, synth_parameters[i]) 

		for i, param_name in enumerate(list(synth.inputs.keys())):
			print(f'{param_name}: {synth.inputs[param_name].value:.3f}')
			current_p = synth.inputs[param_name].value
			p_fade = Line(current_p, synth_parameters[i], fade_time)
			synth.set_input(param_name, p_fade) 


		prev_audioin_features = audioin_features
		time.sleep(update_interval_s)

