from signalflow import *
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import librosa


class Theremin(Patch):
    '''A synthesis patch with N continuous parameters between 0 and 1'''
    def __init__(self, parameter_values: list = [0,0]):
        super().__init__()
        frequency = self.add_input("frequency", parameter_values[0])
        gain = self.add_input("gain", parameter_values[1])
        # frequency = ScaleLinLin(frequency, Constant(0), Constant(1), Constant(50), Constant(220))
        frequency = ScaleLinLin(frequency, 0, 1, 220, 880)
        # gain = ScaleLinLin(gain, Constant(0), Constant(1), Constant(-100), Constant(-5))
        sine = SineOscillator(frequency)
        output = sine * DecibelsToAmplitude(gain)
        # output = sine * gain
        self.set_output(output)
        self.set_auto_free(False)

class Benjolin(Patch):
	def __init__(self, parameter_values: list = [0,0,0,0]):
		super().__init__()

		FRQ01 = self.add_input("parameter-0", parameter_values[0])
		# FRQ02 = self.add_input("parameter-1", parameter_values[1])
		RUN01 = self.add_input("parameter-1", parameter_values[1])
		RUN02 = self.add_input("parameter-2", parameter_values[2])
		# FIL_FRQ = self.add_input("parameter-4", parameter_values[4])
		# FIL_RES = self.add_input("parameter-5", parameter_values[5])
		# FIL_RUN = self.add_input("parameter-6", parameter_values[6])
		# FIL_SWP = self.add_input("parameter-7", parameter_values[7])
		gain = self.add_input("parameter-3", parameter_values[3])

		FRQ02 = 0.1
		FIL_FRQ = 0.5
		FIL_RES = 0.2
		FIL_RUN = 0.4
		FIL_SWP = 0.2

		## PARAMETERS
		FRQ01 = ScaleLinLin(input=FRQ01, a=0, b=1, c=0, d=127)
		FRQ02 = ScaleLinLin(input=FRQ02, a=0, b=1, c=0, d=127)
		RUN01 = ScaleLinLin(input=RUN01, a=0, b=1, c=0, d=127)
		RUN02 = ScaleLinLin(input=RUN02, a=0, b=1, c=0, d=127)
		FIL_FRQ = ScaleLinLin(input=FIL_FRQ, a=0, b=1, c=0, d=127)
		FIL_RES = ScaleLinLin(input=FIL_RES, a=0, b=1, c=0, d=127)
		FIL_RUN = ScaleLinLin(input=FIL_RUN, a=0, b=1, c=0, d=127)
		FIL_SWP = ScaleLinLin(input=FIL_SWP, a=0, b=1, c=0, d=127)

		buf = Buffer(1, 256)
		buf_pulse01 = Buffer(1, 256)
		buf_pulse02 = Buffer(1, 256)

		buf_writer = FeedbackBufferWriter(buf)
		buf_reader = FeedbackBufferReader(buf)
		buf_pulse01_writer = FeedbackBufferWriter(buf_pulse01)
		buf_pulse01_reader = FeedbackBufferReader(buf_pulse01)
		buf_pulse02_writer = FeedbackBufferWriter(buf_pulse02)
		buf_pulse02_reader = FeedbackBufferReader(buf_pulse02)

		sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8 = 0, 0, 0, 0, 0, 0, 0, 0

		## RUNGLER
		XOR = Pow(a=GreaterThan(a=Clip(input=buf_pulse01_reader, min=0, max=1.0), b=0.5), 
			b=buf_reader)
		clock = GreaterThan(a=buf_pulse02_reader, b=0)
		sh1 = SampleAndHold(input=XOR, clock=clock)
		sh2 = SampleAndHold(input=sh1, clock=clock)
		sh3 = SampleAndHold(input=sh2, clock=clock)
		sh4 = SampleAndHold(input=sh3, clock=clock)
		sh5 = SampleAndHold(input=sh4, clock=clock)
		sh6 = SampleAndHold(input=sh5, clock=clock)
		sh7 = SampleAndHold(input=sh6, clock=clock)
		sh8 = SampleAndHold(input=sh7, clock=clock)
		buf_writer.set_input("input", sh8) 
		buf_writer.set_input("delay_time", 0.05) 
		RUNGLER = (sh6/8) + (sh7/4) + (sh8/2)

		## OSCILLATOR 1
		O1 = ScaleLinLin(input=FRQ01, a=0, b=127, c=-61, d=80) + RUNGLER * (RUN01 / 2)
		O1 = MidiNoteToFrequency(a=Clip(input=O1, min=-60, max=127))
		O1_tri = TriangleOscillator(frequency=O1)
		O1_pls = SquareOscillator(frequency=O1)
		buf_pulse01_writer.set_input("input", O1_pls) 
		buf_pulse01_writer.set_input("delay_time", 0.05) 

		## OSCILLATOR 2
		O2 = ScaleLinLin(input=FRQ02, a=0, b=127, c=-59, d=80) + RUNGLER * (RUN02 / 2)
		O2 = MidiNoteToFrequency(a=Clip(input=O2, min=-60, max=127))
		O2_tri = TriangleOscillator(frequency=O2)
		O2_pls = SquareOscillator(frequency=O2)
		buf_pulse02_writer.set_input("input", O2_pls) 
		buf_pulse02_writer.set_input("delay_time", 0.05) 

		PWM = GreaterThan(a=O1_tri, b=O2_tri)
		output = (RUNGLER + PWM) * 0.5

		## FILTER
		filter_f0 = FIL_FRQ + (FIL_RUN * RUNGLER * 127) + (FIL_SWP * O2_tri)
		filter_f0 = MidiNoteToFrequency(a = Clip(input=filter_f0, min=0, max=127))
		filter_res = ScaleLinLin(input=FIL_RES, a=0, b=127, c=0, d=1)
		filter_out = SVFilter(input=output, filter_type='band_pass', cutoff=filter_f0, resonance=filter_res)
		output = filter_out * ScaleLinLin(input=FIL_RES, a=0, b=127, c=2, d=12)
		output = SVFilter(input=output, filter_type='high_pass', cutoff=10, resonance=0.0)
		output = Compressor(input=output, threshold=0.2, ratio=4, attack_time=0.01, release_time=0.1)
		output = output * gain

		self.output = output
		self.set_auto_free(False)

class FM(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0]):
        super().__init__()

        frequency = self.add_input("parameter-0", parameter_values[0])
        c0 = self.add_input("parameter-1", parameter_values[1])
        l0 = self.add_input("parameter-2", parameter_values[2])
        # c1 = self.add_input("parameter-3", parameter_values[3])
        # l1 = self.add_input("parameter-4", parameter_values[4])
        # c2 = self.add_input("parameter-5", parameter_values[5])
        # l2 = self.add_input("parameter-6", parameter_values[6])
        gain = self.add_input("parameter-7", parameter_values[3])

        default_param_values = [0.3, 0.1, 1, 0.2, 0.75, 0.91, 0.95, 0.4]
        c1 = default_param_values[3]
        l1 = default_param_values[4]
        c2 = default_param_values[5]
        l2 = default_param_values[6]
        # gain = default_param_values[7]

        frequency = ScaleLinLin(input=frequency, a=0, b=1, c=0, d=127)
        c0 = ScaleLinLin(input=c0, a=0, b=1, c=0, d=10)
        l0 = ScaleLinLin(input=l0, a=0, b=1, c=0, d=1)
        c1 = ScaleLinLin(input=c1, a=0, b=1, c=0, d=10)
        l1 = ScaleLinLin(input=l1, a=0, b=1, c=0, d=1)
        c2 = ScaleLinLin(input=c2, a=0, b=1, c=0, d=10)
        l2 = ScaleLinLin(input=l2, a=0, b=1, c=0, d=1)

        # params = [[1.00, 1.0], [2.0, 0.75], [9.1, 0.95]]
        params = [[c0,l0], [c1,l1], [c2,l2]]

        frequency = MidiNoteToFrequency(frequency)
        op2 = FMOp(frequency, *params[2])
        op1 = FMOp(frequency, *params[1], fm=op2)
        op0 = FMOp(frequency, *params[0], fm=op1+op2)
        output = (op0 + op1) * gain

        self.output = output
        self.set_auto_free(False)

class FMOp (Patch):
    def __init__(self, f0, coarse, level, fm=None):
        """
        FM operator that follows the exponential output curve of the Yamaha DX7:
        https://sound.stackexchange.com/a/42979/33063
        """
        super().__init__()
        amplitude = ScaleLinExp(level, 0, 1, 0.0005, 1)
        frequency = f0 * coarse
        if fm is not None:
            frequency = frequency + (f0 * fm * 14)
        sine = SineOscillator(frequency)
        self.output = sine * amplitude


class Granular(Patch):
	def __init__(self, parameter_values: list = [0,0,0,0]):
		super().__init__()

		rate = self.add_input("rate", parameter_values[0])
		gain = self.add_input("gain", parameter_values[1])
		duration = self.add_input("grain-duration", parameter_values[2])
		position = self.add_input("start-position", parameter_values[3])

		audio_path = os.path.join(os.path.dirname(__file__), "02_samples", "tibetan-ball.wav")
		buf = Buffer(audio_path)

		rate = ScaleLinLin(input=rate, a=0, b=1, c=0.1, d=2.1)
		# gain = ScaleLinLin(input=gain, a=0, b=1, c=-100, d=-10)
		duration = ScaleLinLin(input=duration, a=0, b=1, c=0.1, d=2)
		position = ScaleLinLin(input=position, a=0, b=1, c=0.01, d=buf.duration - 2+0.01)

		# create synth
		frequency = 100
		position_variance = 0
		duration_variance = 0
		clock = Impulse(frequency=frequency)
		granulator = Granulator(buffer=buf,
								clock=clock,
								pan=0,
								pos=RandomUniform(position-position_variance, 
									position+position_variance, clock=clock),
								rate=rate,
								duration=RandomUniform(duration-duration_variance, 
									duration+duration_variance, clock=clock))
		envelope = EnvelopeBuffer("triangle")
		granulator.set_buffer("envelope", envelope)
		# output = granulator * DecibelsToAmplitude(gain)
		output = granulator * gain

		# forward
		self.set_output(output)
		self.set_auto_free(False)



class DoubleSynth:
	def __init__(self, 
				src_synth_name: str,
				tgt_synth_name: str,
				sample_rate=44100,
				window_size = 2048,
				hop_size = 512
				):
		self.config = AudioGraphConfig()
		self.config.sample_rate = sample_rate
		self.config.output_buffer_size = hop_size
		self.src_synth_name = src_synth_name
		self.tgt_synth_name = tgt_synth_name
		self.window_size = window_size
		self.hop_size = hop_size
		self.fade_time = hop_size / sample_rate / 8
		self.connectGraph()
	def connectGraph(self):
		# intialize graph
		self.graph = AudioGraph(output_device=AudioOut_Dummy(2), start=False)
		# initialize buffers
		self.buffer = Buffer(1, self.hop_size)
		self.src_buf = Buffer(1, self.hop_size)
		self.tgt_buf = Buffer(1, self.hop_size)
		# intialize graph
		if self.src_synth_name == 'Benjolin':
			self.src_synth = Benjolin() 
		elif self.src_synth_name == 'FM':
			self.src_synth = FM() 
		elif self.src_synth_name == 'Granular':
			self.src_synth = Granular() 
		else:
			self.src_synth = Theremin() 
		if self.tgt_synth_name == 'Benjolin':
			self.tgt_synth = Benjolin() 
		elif self.tgt_synth_name == 'FM':
			self.tgt_synth = FM() 
		elif self.tgt_synth_name == 'Granular':
			self.tgt_synth = Granular() 
		else:
			self.tgt_synth = Theremin()
		self.src_synth.play()
		self.tgt_synth.play()
	def forward(self):
		# make a new recorder
		self.src_rec = BufferRecorder(self.src_buf, self.src_synth)
		self.tgt_rec = BufferRecorder(self.tgt_buf, self.tgt_synth)
		self.graph.play(self.src_rec)
		self.graph.play(self.tgt_rec)
		self.graph.render_to_buffer(self.buffer)
		return self.src_buf.data[0,:], self.tgt_buf.data[0,:]
	def resetParametersSRC(self, new_parameters: list):
		for i, param_name in enumerate(list(self.src_synth.inputs.keys())):
			self.src_synth.set_input(param_name, new_parameters[i]) 
	def resetParametersTGT(self, new_parameters: list):
		for i, param_name in enumerate(list(self.tgt_synth.inputs.keys())):
			self.tgt_synth.set_input(param_name, new_parameters[i]) 
	def fadeParametersSRC(self, new_parameters: list, fade_time: float=None):
		if fade_time is None:
			fade_time = self.fade_time
		for i, param_name in enumerate(list(self.src_synth.inputs.keys())):
			current_p = self.src_synth.inputs[param_name].value
			p_fade = Line(current_p, new_parameters[i], fade_time)
			self.src_synth.set_input(param_name, p_fade) 
	def fadeParametersTGT(self, new_parameters: list, fade_time: float=None):
		if fade_time is None:
			fade_time = self.fade_time
		for i, param_name in enumerate(list(self.tgt_synth.inputs.keys())):
			current_p = self.tgt_synth.inputs[param_name].value
			p_fade = Line(current_p, new_parameters[i], fade_time)
			self.tgt_synth.set_input(param_name, p_fade) 
	def getInputsSRC(self):
		return self.src_synth.inputs
	def getInputsTGT(self):
		return self.tgt_synth.inputs
	def resetGraph(self):
		if self.graph:
			self.graph.destroy()
		self.connectGraph()
	def close(self):
		self.graph.destroy()


if __name__ == "__main__":
	
	sr, window_size, hop_size = 44100, 4096, 2048
	synth = DoubleSynth('Theremin', 'Theremin', sr, window_size, hop_size)
	N_params_src = len(synth.getInputsSRC())
	N_params_tgt = len(synth.getInputsTGT())
	synth_parameters = np.random.rand(N_params_src).tolist()
	synth.resetParametersSRC(synth_parameters)
	synth_parameters = np.random.rand(N_params_tgt).tolist()
	synth.resetParametersTGT(synth_parameters)

	STEPS = 30
	src_array = np.zeros((hop_size*STEPS))
	tgt_array = np.zeros((hop_size*STEPS))
	total = np.zeros((hop_size*STEPS))
	rms_src = []
	rms_tgt = []
	cent_src = []
	cent_tgt = []
	for n in range(STEPS):
		src_sound, tgt_sound = synth.forward()
		synth_parameters = np.random.rand(N_params_src).tolist()
		synth.fadeParametersSRC(synth_parameters)
		synth_parameters = np.random.rand(N_params_tgt).tolist()
		synth.fadeParametersTGT(synth_parameters)
		# src_array[hop_size*n:hop_size*(n+1)], tgt_array[hop_size*n:hop_size*(n+1)] = src_sound, tgt_sound

		src_array[:-hop_size] = src_array[hop_size:] 
		src_array[-hop_size:] = src_sound[-hop_size:] 
		tgt_array[:-hop_size] = tgt_array[hop_size:]
		tgt_array[-hop_size:] = tgt_sound[-hop_size:]

		rms_src.append(librosa.feature.rms(y=src_array[-hop_size*2:], frame_length=window_size, hop_length=hop_size)[0,-2])
		rms_tgt.append(librosa.feature.rms(y=tgt_array[-hop_size*2:], frame_length=window_size, hop_length=hop_size)[0,-2])
		print(librosa.feature.rms(y=src_array[-hop_size*2:], frame_length=window_size, hop_length=hop_size))
		print(librosa.feature.rms(y=tgt_array[-hop_size*2:], frame_length=window_size, hop_length=hop_size))
		cent_src.append(librosa.feature.spectral_centroid(y=src_array, sr=sr, n_fft=window_size, hop_length=hop_size)[0,-2])
		cent_tgt.append(librosa.feature.spectral_centroid(y=tgt_array, sr=sr, n_fft=window_size, hop_length=hop_size)[0,-2])
		
	sf.write('prova_src_synth.wav', np.array(src_array), sr, subtype='PCM_24')
	sf.write('prova_tgt_synth.wav', np.array(tgt_array), sr, subtype='PCM_24')
	# sf.write('total.wav', np.array(total), sr, subtype='PCM_24')

	# print(np.array(rms_src).mean(axis=1).shape)
	# print(np.array(rms_tgt).mean(axis=1).shape)
	# print(np.array(cent_src).mean(axis=1).shape)
	# print(np.array(cent_tgt).mean(axis=1).shape)


	# fig, ax = plt.subplots(2, figsize=(20, 8))
	# librosa.display.waveshow(y=src_array, sr=sr, x_axis='time', ax=ax[0], alpha=0.5)
	# librosa.display.waveshow(y=tgt_array, sr=sr, x_axis='time', ax=ax[1], alpha=0.5)
	# ax[0].set_xlabel('time [s]')
	# ax[0].set_title(f'SRC_Theremin')
	# # ax[0].set_ylim([-1,1])
	# ax[1].set_xlabel('time [s]')
	# ax[1].set_title(f'TGT_Theremin')
	# # ax[1].set_ylim([-1,1])
	# plt.show()

	rms = librosa.feature.rms(y=src_array, frame_length=window_size, hop_length=hop_size)
	plt.plot(rms[0], label='src')
	rms = librosa.feature.rms(y=tgt_array, frame_length=window_size, hop_length=hop_size)
	plt.plot(rms[0], label='tgt')
	plt.plot(np.array(rms_src), label='src_mean')
	plt.plot(np.array(rms_tgt), label='tgt_mean')
	plt.legend()
	plt.show()

	cent = librosa.feature.spectral_centroid(y=src_array, sr=sr, n_fft=window_size, hop_length=hop_size)
	plt.plot(cent[0], label='src')
	cent = librosa.feature.spectral_centroid(y=tgt_array, sr=sr, n_fft=window_size, hop_length=hop_size)
	plt.plot(cent[0], label='tgt')
	plt.plot(np.array(cent_src), label='src_mean')
	plt.plot(np.array(cent_tgt), label='tgt_mean')
	plt.legend()
	plt.show()
