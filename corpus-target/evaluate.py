import json
import numpy as np
from stable_baselines3 import A2C, PPO, TD3, SAC
import gymnasium as gym
from environment import SynthGenEnv
import os

def evaluateRLmodel(log_dir, N_eval_episodes):

	# Opening and reading the JSON file
	with open(f'{log_dir}/training_config.json', 'r') as f:
		# Parsing the JSON file into a Python dictionary
		training_parameters = json.load(f)

	# create env
	# define environment
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
		synths_info_dir='corpus-target/02_synthesizers',
		render_mode="human",
		save_folder=log_dir
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

	# Run 2 episodes
	obs = env.reset()[0]
	count_episodes = 0
	cumulative_reward = 0
	rewards = []
	while count_episodes < N_eval_episodes:
		action, _ = model.predict(observation=obs, deterministic=True) 
		# Turn on deterministic, so predict always returns the same behavior
		obs, reward, terminated, _, _ = env.step(action)
		cumulative_reward += reward

		if terminated:
			obs = env.reset()[0]
			count_episodes += 1
			rewards.append(cumulative_reward/training_parameters["max_episode_duration"])
			cumulative_reward = 0

	return np.array(rewards).mean()


if __name__ == "__main__":
	
	MODEL_NAME = '2026-05-28/1779967918-SAC'
	log_dir = f'corpus-target/01_model_logs/{MODEL_NAME}'
	reward = evaluateRLmodel(log_dir, 10)
	print(f'Evaluated model {MODEL_NAME} | Reward: {reward}')


	log_dir = f'corpus-target/01_model_logs/{MODEL_NAME}/evaluation'
	dirs_list = [name for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name))]
	mrSTFT_mean = 0
	mrSTFT_std = 0
	dMSE_mean = 0
	dMSE_std = 0
	diffMSE_mean = 0
	diffMSE_std = 0
	specMAE_mean = 0
	specMAE_std = 0
	specConv_mean = 0
	specConv_std = 0
	mfccMAE_mean = 0
	mfccMAE_std = 0
	mfccMAE40_mean = 0
	mfccMAE40_std = 0
	# SOT_mean = 0
	# SOT_std = 0
	for dir in dirs_list:
		with open(f'{log_dir}/{dir}/evaluation.json', 'r', encoding='utf-8') as file:
			data = json.load(file)
		mrSTFT_mean += data['mrSTFT_mean'] if str(data['mrSTFT_mean']) != "nan" else 0 
		mrSTFT_std += data['mrSTFT_std'] if str(data['mrSTFT_std']) != "nan" else 0 
		dMSE_mean += data['dMSE_mean'] if str(data['dMSE_mean']) != "nan" else 0 
		dMSE_std += data['dMSE_std'] if str(data['dMSE_std']) != "nan" else 0 
		diffMSE_mean += data['diffMSE_mean'] if str(data['diffMSE_mean']) != "nan" else 0 
		diffMSE_std += data['diffMSE_std'] if str(data['diffMSE_std']) != "nan" else 0 
		specMAE_mean += data['specMAE_mean'] if str(data['specMAE_mean']) != "nan" else 0 
		specMAE_std += data['specMAE_std'] if str(data['specMAE_std']) != "nan" else 0 
		specConv_mean += data['specConv_mean'] if str(data['specConv_mean']) != "nan" else 0 
		specConv_std += data['specConv_std'] if str(data['specConv_std']) != "nan" else 0 
		mfccMAE_mean += data['mfccMAE_mean'] if str(data['mfccMAE_mean']) != "nan" else 0 
		mfccMAE_std += data['mfccMAE_std'] if str(data['mfccMAE_std']) != "nan" else 0 
		mfccMAE40_mean += data['mfccMAE40_mean'] if str(data['mfccMAE40_mean']) != "nan" else 0 
		mfccMAE40_std += data['mfccMAE40_std'] if str(data['mfccMAE40_std']) != "nan" else 0 
		# SOT_mean += data['SOT_mean'] if str(data['SOT_mean']) != "nan" else 0 
		# SOT_std += data['SOT_std'] if str(data['SOT_std']) != "nan" else 0 
	mrSTFT_mean /= len(dir) - 1
	mrSTFT_std /= len(dir) - 1
	dMSE_mean /= len(dir) - 1
	dMSE_std /= len(dir) - 1
	diffMSE_mean /= len(dir) - 1
	diffMSE_std /= len(dir) - 1
	specMAE_mean /= len(dir) - 1
	specMAE_std /= len(dir) - 1
	specConv_mean /= len(dir) - 1
	specConv_std /= len(dir) - 1
	mfccMAE_mean /= len(dir) - 1
	mfccMAE_std /= len(dir) - 1
	mfccMAE40_mean /= len(dir) - 1
	mfccMAE40_std /= len(dir) - 1
	# SOT_mean /= len(dir) - 1
	# SOT_std /= len(dir) - 1

	tot_evaluation = {}
	tot_evaluation['mrSTFT_mean'] = mrSTFT_mean
	tot_evaluation['mrSTFT_std'] = mrSTFT_std
	tot_evaluation['dMSE_mean'] = dMSE_mean
	tot_evaluation['dMSE_std'] = dMSE_std
	tot_evaluation['diffMSE_mean'] = diffMSE_mean
	tot_evaluation['diffMSE_std'] = diffMSE_std
	tot_evaluation['specMAE_mean'] = specMAE_mean
	tot_evaluation['specMAE_std'] = specMAE_std
	tot_evaluation['specConv_mean'] = specConv_mean
	tot_evaluation['specConv_std'] = specConv_std
	tot_evaluation['mfccMAE_mean'] = mfccMAE_mean
	tot_evaluation['mfccMAE_std'] = mfccMAE_std
	tot_evaluation['mfccMAE40_mean'] = mfccMAE40_mean
	tot_evaluation['mfccMAE40_std'] = mfccMAE40_std
	# tot_evaluation['SOT_mean'] = SOT_mean
	# tot_evaluation['SOT_std'] = SOT_std
	with open(f'{log_dir}/evaluation.json', 'w', encoding='utf-8') as f:
		json.dump(tot_evaluation, f, ensure_ascii=False, indent=4)
