from signalflow import *
import numpy as np
import soundfile as sf
import time
import json
import librosa
import joblib
import mido

from stable_baselines3 import A2C, PPO, TD3, SAC
import gymnasium as gym
from environment import SynthGenEnv, computeFeatures

from synthesizers import *


if __name__ == "__main__":

    # MIDI INPUT
    MIDI_device_name = "Midi Fighter Twister" # substitute with the name of your MIDI device
    default_input = mido.open_input(MIDI_device_name)
    midi_port_output = mido.open_output(MIDI_device_name)

    red = 80
    orange = 70
    green = 50
    purple = 100
    blue = 10
    black = 0
    for j in range(64):
        if j in [3, 11, 19]:
            midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=green, time=0))
        elif j in [4, 5]:
            midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=purple, time=0))
        elif j in list(range(12,12+4)):
            midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=blue, time=0))
        elif j in list(range(24,24+6)):
            midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=orange, time=0))
        # elif j in list(range(48-16,48+8-16)):
        #     midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=orange, time=0))
        # elif j in list(range(56-16,56+4-16)):
        #     midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=blue, time=0))
        else:
            midi_port_output.send(mido.Message('control_change', channel=1, control=j, value=black, time=0))


    theremin_gain = 0.8
    theremin_offset_MIDI_channel = 4
    midi_port_output.send(mido.Message('control_change', channel=0, control=int(theremin_offset_MIDI_channel), value=int(127*theremin_gain), time=0))
    granular_gain = 0.8
    granular_offset_MIDI_channel = 8+4
    midi_port_output.send(mido.Message('control_change', channel=0, control=int(granular_offset_MIDI_channel), value=int(127*granular_gain), time=0))
    benjolin_gain = 0.8
    benjolin_offset_MIDI_channel = 24
    midi_port_output.send(mido.Message('control_change', channel=0, control=int(benjolin_offset_MIDI_channel), value=int(127*benjolin_gain), time=0))


    # load THEREMIN model
    MODEL_NAME_theremin = '2026-06-26/1782464991-SAC'
    log_dir_theremin = f'./corpus-target/01_model_logs/{MODEL_NAME_theremin}'
    with open(f'{log_dir_theremin}/training_config.json', 'r') as f:
        training_parameters_theremin = json.load(f)

    # load GRANULAR model
    MODEL_NAME_granular = '2026-06-26/1782463243-SAC'
    log_dir_granular = f'./corpus-target/01_model_logs/{MODEL_NAME_granular}'
    with open(f'{log_dir_granular}/training_config.json', 'r') as f:
        training_parameters_granular = json.load(f)

    # load BENJOLIN model
    MODEL_NAME_benjolin = '2026-06-26/1782437554-SAC'
    log_dir_benjolin = f'./corpus-target/01_model_logs/{MODEL_NAME_benjolin}'
    with open(f'{log_dir_benjolin}/training_config.json', 'r') as f:
        training_parameters_benjolin = json.load(f)


    # # load BENJOLIN-GRANULAR model
    # MODEL_NAME_benjolin_to_granular = '2026-06-26/1782462770-SAC'
    # log_dir_benjolin_to_granular = f'./synth-target/01_model_logs/{MODEL_NAME_benjolin_to_granular}'
    # with open(f'{log_dir_benjolin_to_granular}/training_config.json', 'r') as f:
    #     training_parameters_benjolin_to_granular = json.load(f)


    # AUDIO GRAPH PARAMETERS
    sr = training_parameters_theremin["sample_rate"]
    window_size = training_parameters_theremin["FFT_window_size"]
    hop_size = training_parameters_theremin["hop_size"]

    # START AUDIO GRAPH
    config = AudioGraphConfig()
    config.sample_rate = sr
    config.output_buffer_size = 256
    config.input_device_name = "Scarlett 2i2 USB" 
    config.output_device_name = "Scarlett 2i2 USB"
    # config.input_device_name = "MacBook Pro Microphone" 
    # config.output_device_name = "External Headphones"
    graph = AudioGraph(config)
    audio_in = AudioIn() * 1.5

    # LOAD SYNTHESIZERS
    theremin = Theremin()
    granular = Granular()
    benjolin = Benjolin(graph)
    theremin.play()
    granular.play()
    benjolin.play()

    # # LOAD SYNTHS FOR SYNTH-TO-SYNTH
    # benjolin_sth = Benjolin(graph)
    # granular_sth = Granular()
    # benjolin_sth.play()
    # granular_sth.play()


    # initialize params
    N_params = len(theremin.inputs)
    synth_parameters_theremin = np.zeros(N_params).tolist()
    for i, param_name in enumerate(list(theremin.inputs.keys())):
        current_p = theremin.inputs[param_name].value
        theremin.set_input(param_name, synth_parameters_theremin[i]) 

    N_params = len(granular.inputs)
    synth_parameters_granular = np.zeros(N_params).tolist()
    for i, param_name in enumerate(list(granular.inputs.keys())):
        current_p = granular.inputs[param_name].value
        granular.set_input(param_name, synth_parameters_granular[i]) 

    N_params = len(benjolin.inputs)
    synth_parameters_benjolin = np.zeros(N_params).tolist()
    for i, param_name in enumerate(list(benjolin.inputs.keys())):
        current_p = benjolin.inputs[param_name].value
        benjolin.set_input(param_name, synth_parameters_benjolin[i]) 

    # # 4 synth-to-synth
    # N_params = len(granular_sth.inputs)
    # synth_parameters_granular_sth = np.zeros(N_params).tolist()
    # for i, param_name in enumerate(list(granular_sth.inputs.keys())):
    #     current_p = granular_sth.inputs[param_name].value
    #     granular_sth.set_input(param_name, synth_parameters_granular_sth[i]) 

    # N_params = len(benjolin_sth.inputs)
    # synth_parameters_benjolin_sth = np.zeros(N_params).tolist()
    # for i, param_name in enumerate(list(benjolin_sth.inputs.keys())):
    #     current_p = benjolin_sth.inputs[param_name].value
    #     benjolin_sth.set_input(param_name, synth_parameters_benjolin_sth[i]) 


    # load models and scalers
    corpus_scaler_theremin = joblib.load(f'{log_dir_theremin}/tgt_synth_scaler.pkl')
    synth_scaler_theremin = joblib.load(f'{log_dir_theremin}/src_synth_scaler.pkl')
    corpus_scaler_granular = joblib.load(f'{log_dir_granular}/tgt_synth_scaler.pkl')
    synth_scaler_granular = joblib.load(f'{log_dir_granular}/src_synth_scaler.pkl')
    corpus_scaler_benjolin = joblib.load(f'{log_dir_benjolin}/tgt_synth_scaler.pkl')
    synth_scaler_benjolin = joblib.load(f'{log_dir_benjolin}/src_synth_scaler.pkl')

    # corpus_scaler_benjolin_sth = joblib.load(f'{log_dir_benjolin}/tgt_synth_scaler.pkl')
    # synth_scaler_benjolin_sth = joblib.load(f'{log_dir_benjolin}/src_synth_scaler.pkl')


    # CREATE ENVIRONMENTS
    env_theremin = gym.make(
        training_parameters_theremin["environment"], features=training_parameters_theremin["features"],
        src_synth_type=training_parameters_theremin["src_synth_type"], corpus_path=training_parameters_theremin["corpus_path"],
        sample_rate = training_parameters_theremin["sample_rate"],FFT_window_size = training_parameters_theremin["FFT_window_size"],
        hop_size = training_parameters_theremin["hop_size"],N_steps_memory=training_parameters_theremin["N_steps_memory"],
        N_reward_memory=training_parameters_theremin["N_reward_memory"],max_episode_duration=training_parameters_theremin["max_episode_duration"],
        step_size=training_parameters_theremin["step_size"],rewards = training_parameters_theremin["rewards"],
        normalization_mode = training_parameters_theremin["normalization_mode"],episode_mode = training_parameters_theremin["episode_mode"],
        synths_info_dir='./corpus-target/02_synthesizers',save_folder=log_dir_theremin,train=False,render_mode=None
    )
    env_granular = gym.make(
        training_parameters_granular["environment"], features=training_parameters_granular["features"],
        src_synth_type=training_parameters_granular["src_synth_type"], corpus_path=training_parameters_granular["corpus_path"],
        sample_rate = training_parameters_granular["sample_rate"],FFT_window_size = training_parameters_granular["FFT_window_size"],
        hop_size = training_parameters_granular["hop_size"],N_steps_memory=training_parameters_granular["N_steps_memory"],
        N_reward_memory=training_parameters_granular["N_reward_memory"],max_episode_duration=training_parameters_granular["max_episode_duration"],
        step_size=training_parameters_granular["step_size"],rewards = training_parameters_granular["rewards"],
        normalization_mode = training_parameters_granular["normalization_mode"],episode_mode = training_parameters_granular["episode_mode"],
        synths_info_dir='./corpus-target/02_synthesizers',save_folder=log_dir_granular,train=False,render_mode=None
    )
    env_benjolin = gym.make(
        training_parameters_benjolin["environment"], features=training_parameters_benjolin["features"],
        src_synth_type=training_parameters_benjolin["src_synth_type"], corpus_path=training_parameters_benjolin["corpus_path"],
        sample_rate = training_parameters_benjolin["sample_rate"],FFT_window_size = training_parameters_benjolin["FFT_window_size"],
        hop_size = training_parameters_benjolin["hop_size"],N_steps_memory=training_parameters_benjolin["N_steps_memory"],
        N_reward_memory=training_parameters_benjolin["N_reward_memory"],max_episode_duration=training_parameters_benjolin["max_episode_duration"],
        step_size=training_parameters_benjolin["step_size"],rewards = training_parameters_benjolin["rewards"],
        normalization_mode = training_parameters_benjolin["normalization_mode"],episode_mode = training_parameters_benjolin["episode_mode"],
        synths_info_dir='./corpus-target/02_synthesizers',save_folder=log_dir_benjolin,train=False,render_mode=None
    )

    # LOAD MODEL THEREMIN
    AGENT_TYPE = training_parameters_theremin["AGENT_TYPE"]
    MODEL_DIR = f'{log_dir_theremin}/best'
    if AGENT_TYPE == 'PPO':
        model_theremin = PPO.load(MODEL_DIR, env=env_theremin)
    elif AGENT_TYPE == 'A2C':
        model_theremin = A2C.load(MODEL_DIR, env=env_theremin)
    elif AGENT_TYPE == 'TD3':
        model_theremin = TD3.load(MODEL_DIR, env=env_theremin)
    elif AGENT_TYPE == 'SAC':
        model_theremin = SAC.load(MODEL_DIR, env=env_theremin)

    # LOAD MODEL GRANULAR
    AGENT_TYPE = training_parameters_granular["AGENT_TYPE"]
    MODEL_DIR = f'{log_dir_granular}/best'
    if AGENT_TYPE == 'PPO':
        model_granular = PPO.load(MODEL_DIR, env=env_granular)
    elif AGENT_TYPE == 'A2C':
        model_granular = A2C.load(MODEL_DIR, env=env_granular)
    elif AGENT_TYPE == 'TD3':
        model_granular = TD3.load(MODEL_DIR, env=env_granular)
    elif AGENT_TYPE == 'SAC':
        model_granular = SAC.load(MODEL_DIR, env=env_granular)

    # LOAD MODEL BENJOLIN
    AGENT_TYPE = training_parameters_benjolin["AGENT_TYPE"]
    MODEL_DIR = f'{log_dir_benjolin}/best'
    if AGENT_TYPE == 'PPO':
        model_benjolin = PPO.load(MODEL_DIR, env=env_benjolin)
    elif AGENT_TYPE == 'A2C':
        model_benjolin = A2C.load(MODEL_DIR, env=env_benjolin)
    elif AGENT_TYPE == 'TD3':
        model_benjolin = TD3.load(MODEL_DIR, env=env_benjolin)
    elif AGENT_TYPE == 'SAC':
        model_benjolin = SAC.load(MODEL_DIR, env=env_benjolin)

    # CREATE INPUT BUFFERS FOR RECORDING
    input_buffer = Buffer(1, window_size*3)
    theremin_buffer = Buffer(1, window_size*3)
    granular_buffer = Buffer(1, window_size*3)
    benjolin_buffer = Buffer(1, window_size*3)

    # add delay lines for responses
    min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
    graph.add_node(FeedbackBufferWriter(input_buffer, audio_in, min_feedback_writer_delay))
    graph.add_node(FeedbackBufferWriter(theremin_buffer, theremin, min_feedback_writer_delay))
    graph.add_node(FeedbackBufferWriter(granular_buffer, granular, min_feedback_writer_delay))
    graph.add_node(FeedbackBufferWriter(benjolin_buffer, benjolin, min_feedback_writer_delay))


    fade_time = hop_size / sr / 8
    update_interval_s = hop_size / sr
    input_sound = input_buffer.data[0,:]
    theremin_sound = theremin_buffer.data[0,:]
    granular_sound = granular_buffer.data[0,:]
    benjolin_sound = benjolin_buffer.data[0,:]

    prev_audioin_features = computeFeatures(input_sound, training_parameters_theremin["features"], sr, window_size, hop_size)
    prev_audioin_features_theremin = corpus_scaler_theremin.transform(np.array(prev_audioin_features))
    prev_audioin_features_granular = corpus_scaler_granular.transform(np.array(prev_audioin_features))
    prev_audioin_features_benjolin = corpus_scaler_benjolin.transform(np.array(prev_audioin_features))

    previous_src_features_theremin = computeFeatures(theremin_sound, training_parameters_theremin["features"], sr, window_size, hop_size)
    prev_audioin_features_theremin = synth_scaler_theremin.transform(np.array(previous_src_features_theremin))
    previous_src_features_granular = computeFeatures(granular_sound, training_parameters_theremin["features"], sr, window_size, hop_size)
    prev_audioin_features_granular = synth_scaler_granular.transform(np.array(previous_src_features_granular))
    previous_src_features_benjolin = computeFeatures(benjolin_sound, training_parameters_theremin["features"], sr, window_size, hop_size)
    prev_audioin_features_benjolin = synth_scaler_theremin.transform(np.array(previous_src_features_benjolin))
    time.sleep(update_interval_s)

    theremin_on = True
    granular_on = False
    granular.stop()
    midi_port_output.send(mido.Message('control_change', channel=1, control=11, value=red, time=0))
    benjolin_on = False
    benjolin.stop()
    midi_port_output.send(mido.Message('control_change', channel=1, control=19, value=red, time=0))
    with mido.open_input(MIDI_device_name) as port:
        while True:
            for MIDImsg in port.iter_pending():
                if MIDImsg.is_cc(3) and MIDImsg.value == 127:
                    if theremin_on:
                        theremin_on = False
                        theremin.stop()
                        midi_port_output.send(mido.Message('control_change', channel=1, control=3, value=red, time=0))
                    else:
                        theremin_on = True
                        theremin.play()
                        midi_port_output.send(mido.Message('control_change', channel=1, control=3, value=green, time=0))
                if MIDImsg.is_cc(11) and MIDImsg.value == 127:
                    if granular_on:
                        granular_on = False
                        granular.stop()
                        midi_port_output.send(mido.Message('control_change', channel=1, control=11, value=red, time=0))
                    else:
                        granular_on = True
                        granular.play()
                        midi_port_output.send(mido.Message('control_change', channel=1, control=11, value=green, time=0))
                if MIDImsg.is_cc(19) and MIDImsg.value == 127:
                    if benjolin_on:
                        benjolin_on = False
                        benjolin.stop()
                        midi_port_output.send(mido.Message('control_change', channel=1, control=19, value=red, time=0))
                    else:
                        benjolin_on = True
                        benjolin.play()
                        midi_port_output.send(mido.Message('control_change', channel=1, control=19, value=green, time=0))

            input_sound = input_buffer.data[0,:]
            if theremin_on:
                theremin_sound = theremin_buffer.data[0,:]
            if granular_on:
                granular_sound = granular_buffer.data[0,:]
            if benjolin_on:
                benjolin_sound = benjolin_buffer.data[0,:]

            audioin_features = computeFeatures(input_sound, training_parameters_theremin["features"], sr, window_size, hop_size)
            if theremin_on:
                audioin_features_theremin = corpus_scaler_theremin.transform(np.array(audioin_features))
            if granular_on:
                audioin_features_granular = corpus_scaler_granular.transform(np.array(audioin_features))
            if benjolin_on:
                audioin_features_benjolin = corpus_scaler_benjolin.transform(np.array(audioin_features))

            if theremin_on:
                previous_src_features_theremin = computeFeatures(theremin_sound, training_parameters_theremin["features"], sr, window_size, hop_size)
            if granular_on:
                previous_src_features_granular = computeFeatures(granular_sound, training_parameters_theremin["features"], sr, window_size, hop_size)
            if benjolin_on:
                previous_src_features_benjolin = computeFeatures(benjolin_sound, training_parameters_theremin["features"], sr, window_size, hop_size)

            # UPDATE THEREMIN
            if theremin_on:
                observation_theremin = np.concatenate((np.array(audioin_features_theremin)[-2,:].reshape(-1), 
                                            np.array(prev_audioin_features_theremin)[-2,:].reshape(-1), 
                                            np.array(previous_src_features_theremin)[-2,:].reshape(-1),
                                            np.array(synth_parameters_theremin).reshape(-1)))
                action, _ = model_theremin.predict(observation=observation_theremin, deterministic=True)
                synth_parameters_theremin += action
                synth_parameters_theremin = np.clip(synth_parameters_theremin, a_min=0, a_max=1)
                print('-' * 20)
                for i, param_name in enumerate(list(theremin.inputs.keys())):
                    print(f'{param_name}: {theremin.inputs[param_name].value:.3f}')
                    current_p = theremin.inputs[param_name].value
                    p_fade = Line(current_p, synth_parameters_theremin[i], fade_time)
                    theremin.set_input(param_name, p_fade) 


            # UPDATE GRANULAR
            if granular_on:
                observation_granular = np.concatenate((np.array(audioin_features_granular)[-2,:].reshape(-1), 
                                            np.array(prev_audioin_features_granular)[-2,:].reshape(-1), 
                                            np.array(previous_src_features_granular)[-2,:].reshape(-1),
                                            np.array(synth_parameters_granular).reshape(-1)))
                action, _ = model_granular.predict(observation=observation_granular, deterministic=True)
                synth_parameters_granular += action
                synth_parameters_granular = np.clip(synth_parameters_granular, a_min=0, a_max=1)
                print('-' * 20)
                for i, param_name in enumerate(list(granular.inputs.keys())):
                    print(f'{param_name}: {granular.inputs[param_name].value:.3f}')
                    current_p = granular.inputs[param_name].value
                    p_fade = Line(current_p, synth_parameters_granular[i], fade_time)
                    granular.set_input(param_name, p_fade) 


            # UPDATE BENJOLIN
            if benjolin_on:
                observation_benjolin = np.concatenate((np.array(audioin_features_benjolin)[-2,:].reshape(-1), 
                                            np.array(prev_audioin_features_benjolin)[-2,:].reshape(-1), 
                                            np.array(previous_src_features_benjolin)[-2,:].reshape(-1),
                                            np.array(synth_parameters_benjolin).reshape(-1)))
                action, _ = model_benjolin.predict(observation=observation_benjolin, deterministic=True)
                synth_parameters_benjolin += action
                synth_parameters_benjolin = np.clip(synth_parameters_benjolin, a_min=0, a_max=1)
                print('-' * 20)
                for i, param_name in enumerate(list(benjolin.inputs.keys())):
                    print(f'{param_name}: {benjolin.inputs[param_name].value:.3f}')
                    current_p = benjolin.inputs[param_name].value
                    p_fade = Line(current_p, synth_parameters_benjolin[i], fade_time)
                    benjolin.set_input(param_name, p_fade) 

            # UPDATE PARAMETERS ON CONTROLLER
            if theremin_on:
                for pp, param_value in enumerate(synth_parameters_theremin):
                    midi_port_output.send(mido.Message('control_change', channel=0, control=int(pp+theremin_offset_MIDI_channel), value=int(127*param_value), time=0))
            if granular_on:
                for pp, param_value in enumerate(synth_parameters_granular):
                    midi_port_output.send(mido.Message('control_change', channel=0, control=int(pp+granular_offset_MIDI_channel), value=int(127*param_value), time=0))
            if benjolin_on:
                for pp, param_value in enumerate(synth_parameters_benjolin):
                    midi_port_output.send(mido.Message('control_change', channel=0, control=int(pp+benjolin_offset_MIDI_channel), value=int(127*param_value), time=0))

            # MAKE SYNTH-TO-SYNTH

            # UPDATE ARRAY
            if theremin_on:
                prev_audioin_features_theremin = audioin_features_theremin
            if granular_on:
                prev_audioin_features_granular = audioin_features_granular
            if benjolin_on:
                prev_audioin_features_benjolin = audioin_features_benjolin
            time.sleep(update_interval_s)
