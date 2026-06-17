import time
import os
import json
import warnings
from datetime import date

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO, TD3, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from environment import SynthGenEnv
from evaluate import evaluateRLmodel

# tensorboard --logdir ./model_logs

warnings.filterwarnings("ignore")

if __name__ == "__main__":

	today = date.today()

	# define environment
	training_parameters = {}
	training_parameters["environment"] = "synthgen-v0"
	training_parameters["src_synth_type"] = "Benjolin"
	training_parameters["corpus_path"] = './corpus-target/00_corpus/moisesdb_guitar'
	training_parameters["features"] = ['rms','flatness','rolloff','cent']
	training_parameters["sample_rate"] = 44100
	training_parameters["FFT_window_size"] = 4096
	training_parameters["hop_size"] = 2048
	training_parameters["max_episode_duration"] = 1000
	training_parameters["N_steps_memory"] = 2
	training_parameters["N_reward_memory"] = 8
	training_parameters["step_size"] = 1
	training_parameters["rewards"] = ['featureDifference','featureDistance','parameters_MAE']
	training_parameters["normalization_mode"] = 'shared'
	training_parameters["episode_mode"] = 'static'
	training_parameters["AGENT_TYPE"] = 'SAC'
	training_parameters["iterations"] = 10
	training_parameters["timesteps"] = 1000
	training_parameters["n_eval_episodes"] = 3
	training_parameters["eval_interval"] = 5

	# define model
	AGENT_TYPE = training_parameters["AGENT_TYPE"]
	agent_name = f'{int(time.time())}-{AGENT_TYPE}'
	logdir = f'./corpus-target/01_model_logs/{str(today)}/{agent_name}'
	os.makedirs(logdir, exist_ok=True)
	with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
		json.dump(training_parameters, f, ensure_ascii=False, indent=4)

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
		save_folder=logdir,
		train=True,
		render_mode=None
	)
	print(env.observation_space)


	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	policy_kwargs = dict(net_arch=[128, 128])
	if AGENT_TYPE == 'PPO':
		model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, learning_rate=0.0003, verbose=1, device=device, tensorboard_log=logdir)
	elif AGENT_TYPE == 'A2C':
		model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, learning_rate=0.0003, verbose=1, device=device, tensorboard_log=logdir)
	elif AGENT_TYPE == 'TD3':
		model = TD3('MlpPolicy', env, 
				#policy_kwargs=policy_kwargs, 
				learning_rate=0.001, verbose=1, device=device, tensorboard_log=logdir)
	elif AGENT_TYPE == 'SAC':
		model = SAC('MlpPolicy', env, policy_kwargs=policy_kwargs, learning_rate=0.001, verbose=1, device=device, tensorboard_log=logdir)

	# train
	ITERATIONS = training_parameters["iterations"]
	TIMESTEPS = training_parameters["timesteps"]
	N_EVAL_EPISODES = training_parameters["n_eval_episodes"]
	eval_interval = training_parameters["eval_interval"]
	best_model_rew = -np.inf
	for i in range(ITERATIONS):
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=AGENT_TYPE)
		if i % eval_interval:
			mean_rew, std_rew = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
			if mean_rew > best_model_rew:
				best_model_rew = mean_rew
				print(f'Best model reward: {best_model_rew}')
				model.save(f"{logdir}/best")

	reward = evaluateRLmodel(logdir, 10)
	print(f'Evaluated model {agent_name} | Reward: {reward}')


	log_dir = f'corpus-target/01_model_logs/{str(today)}/{agent_name}/evaluation'
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
	with open(f'{log_dir}/evaluation.json', 'w', encoding='utf-8') as f:
		json.dump(tot_evaluation, f, ensure_ascii=False, indent=4)
