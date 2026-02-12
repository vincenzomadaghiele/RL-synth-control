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
	training_parameters["tgt_synth_type"] = "FM"
	training_parameters["features"] = ['rms', 'cent', 'flatness', 'rolloff']
	training_parameters["sample_rate"] = 44100
	training_parameters["FFT_window_size"] = 4096
	training_parameters["hop_size"] = 2048
	training_parameters["max_episode_duration"] = 1000
	training_parameters["N_steps_memory"] = 2
	training_parameters["N_reward_memory"] = 8
	training_parameters["step_size"] = 1
	training_parameters["rewards"] = ['featureDifference', 'featureDistance']
	training_parameters["normalization_mode"] = 'source'
	training_parameters["episode_mode"] = 'moving_target'
	training_parameters["AGENT_TYPE"] = 'SAC'
	training_parameters["iterations"] = 100
	training_parameters["timesteps"] = 10000
	training_parameters["n_eval_episodes"] = 3
	training_parameters["eval_interval"] = 5

	# define model
	AGENT_TYPE = training_parameters["AGENT_TYPE"]
	agent_name = f'{int(time.time())}-{AGENT_TYPE}'
	logdir = f'RL_continuous/in-domain/00_model_logs/{str(today)}/{agent_name}'
	os.makedirs(logdir, exist_ok=True)
	with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
		json.dump(training_parameters, f, ensure_ascii=False, indent=4)

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

	evaluateRLmodel(logdir, 10)


