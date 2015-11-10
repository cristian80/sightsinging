
import numpy as np
import pylab as plt
import struct, wave
import aubio


try:
	import alsaaudio
	alsa_mod = True
except:
	print ("No alsaaudio")
	alsa_mod = False

try:
	import pyaudio
	pyaudio_mod = True
except:
	print ("No pyaudio")
	pyaudio_mod = False


class AudioWaveform ():

	def __init__ (self, channels = 1, rate = 44100, framesize = 1024, downsample = 2):

		self.channels 	= channels
		self.in_format 	= None
		self.rate 		= rate
		self.framesize	= framesize
		self._mod		= None

		self._waveform 	= None
		self._frames 	= None

		self.downsample = downsample
		self.samplerate = self.rate / self.downsample
		self.win_s = 1024 / self.downsample # fft size
		self.hop_s = 512  / self.downsample # hop size
		self._valid_notenames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

		self._beats = None

		self.min_freq = 50
		self.max_freq = 1500
		self._step_window_frame = 30 #step-detection widow size (in nr of frames)
		self._step_window_shift = 5
		self._std_thr = 0.2


	def _record_alsa_stream (self, nframes):
		#set up audio input
		self.in_format 	= alsaaudio.PCM_FORMAT_FLOAT_LE #32 bit samples, as float, little endian

		recorder = alsaaudio.PCM (alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK)
		recorder.setchannels (self.channels)
		recorder.setrate (self.rate)
		recorder.setformat (self.in_format)
		recorder.setperiodsize (self.framesize)

		sound = np.array([])
		for i in np.arange(nframes):
	
			[length, data] = recorder.read()
			a = np.fromstring(data, dtype='int32')
			#w.writeframes(data)
			sound = np.hstack ((sound, a))
		self._waveform = sound

	def _record_pyaudio_stream (self, nframes):
		print ("PyAudio ------")
		self.framesize = 1024*16
		self.in_format = pyaudio.paFloat32 #paInt8

		p = pyaudio.PyAudio()

		stream = p.open(format=self.in_format,
		                channels=self.channels,
		                rate=self.rate,
		                input=True,
		                frames_per_buffer=self.framesize) #buffer

		print("* recording")

		self.frames = []
		sound = []

		for i in range(0, nframes):
		    data = stream.read(self.framesize)
		    a = struct.unpack (str(self.framesize)+'f', data)
		    self.frames.append(data) # 2 bytes(16 bits) per channel

		    sound.append (a)

		print("* done recording")

		stream.stop_stream()
		stream.close()
		p.terminate()

		sound = np.array(sound).flatten()
		a = np.isnan(sound)
		if (len (a)>0):
			print ("NaN elements: ", len(a))
		#sound = sound[~np.isnan(sound)]
		self._waveform = sound
		self._t = np.arange (0, len(self._waveform))/float(self.rate)


	def plot_waveform (self, include_beats = True):

		fig = plt.figure (figsize = (10,6))
		plt.plot (self._t, self._waveform, color = 'RoyalBlue')
		plt.plot (self._t, self._waveform, '.', color = 'RoyalBlue')

		if (include_beats and (self._beats is not(None))):
			for i in range(len(self._beats)):
				plt.axvline (self._beats[i], color = 'crimson')
		plt.show()


	def record_waveform (self, module = "pyaudio", nframes=20):
		self._mod = module

		if (self._mod == "pyaudio"):
			if (pyaudio_mod):
				self._record_pyaudio_stream(nframes)
				self._total_frames = len(self._waveform)
			else:
				print ("pyAudio not working!")
		elif (self._mod == "alsa"):
			if (alsa_mod):
				self._record_alsa_stream(nframes)
				self._total_frames = len(self._waveform)
			else:
				print ("AlsaAudio not working!")
		else:
			print ("Unknwon task.")

	def output_wav (self, filename):
		wf = wave.open(filename, 'wb')
		wf.setnchannels(self.channels)
		wf.setsampwidth(p.get_sample_size(self.in_format))
		wf.setframerate(self.rate)
		wf.writeframes(b''.join(self.frames))
		wf.close()

	def load_wav (self, filename = 'good_test.wav'):
		s = aubio.source('good_test.wav', 44100, 512)
		self.samplerate = s.samplerate
		total_frames = 0
		self._waveform = []
		while True:
			samples, read = s()
			self._waveform.append (samples)
			total_frames += read
			if read < self.hop_s: 
				print ("Loaded frames: ", total_frames)
				break
		self._waveform = np.array(self._waveform).flatten()
		self._waveform = np.float32(self._waveform)
		s.close()
		self._t = np.arange (0, len(self._waveform))/float(self.rate)
		self._total_frames = len(self._waveform)

		#self._waveform = self._waveform[~np.isnan(self._waveform)]


	def calc_power_over_time (self, binsize=1000):
		i = 0
		self._power_binsize = binsize
		self._nr_bins = self._total_frames/self._power_binsize
		self._binned_power = np.zeros(self._nr_bins)

		for i in range(self._nr_bins):
			self._binned_power[i] = (np.sum ((self._waveform [i*binsize:(i+1)*binsize])**2))

		plt.plot (self._binned_power)
		plt.show()

	def remove_silence (self, threshold = 0.2, binsize = 1000):
		self.calc_power_over_time (binsize)
		thr_pwr = self._power_binsize*threshold**2
		self._binned_power[self._binned_power <= thr_pwr] = 0
		plt.plot (self._binned_power)
		plt.show()
		for i in range(self._nr_bins):
			if (self._binned_power[i] == 0):
				self._waveform [i*binsize:(i+1)*binsize] = 0*self._waveform [i*binsize:(i+1)*binsize]




	def extract_beats (self):

		samplerate, win_s, hop_s = 44100, 1024, 512
		s = aubio.source('good_test.wav', 44100, 512)

		o = aubio.tempo("specdiff", self.win_s, self.hop_s, self.rate)
		self._beats = []
		total_frames = 0
		i = 0
		print "Starting extraction ..."
		while True:
			samples = self._waveform [i*self.hop_s:(i+1)*self.hop_s]
			samples = np.float32(samples)
			is_beat = o (samples)

			if is_beat:
				this_beat = o.get_last_s()
				print this_beat
				self._beats.append(this_beat)
			i += 1
			if (i+1)*self.hop_s > len(self._waveform): break

		#bpms = 60./np.diff(self._beats)
		print "Beats:"
		print self._beats
		print "--- BMP:"
		b = 60./np.diff(self._beats)	
		self._bpm = np.median(b)
		print self._bpm

	def pitch_extraction (self, algorithm = "yin", tolerance = 0.95, unit = "midi"):

		self.tolerance = tolerance
		self.unit = unit
		win_s = 1024*4
		hop_s = 512
		pitch_o = aubio.pitch(algorithm, win_s, hop_s, self.samplerate)
		pitch_o.set_unit(self.unit)
		pitch_o.set_tolerance(self.tolerance)

		self._pitches = []
		self._confidences = []

		# total number of frames read
		total_frames = 0

		i = 0
		print "Starting extraction ..."
		while True:
			samples = self._waveform [i*hop_s:(i+1)*hop_s]
			samples = np.float32(samples)
			pitch = pitch_o(samples)[0]
			confidence = pitch_o.get_confidence()
			self._pitches += [pitch]
			self._confidences += [confidence]
			i += 1
			if (i+1)*hop_s > len(self._waveform): break

		self._pitches = np.array(self._pitches).flatten()
		self._filtered_pitches = None
		self._discrete_pitches = None

		plt.plot (self._pitches)
		#for i in range(len(self._beats)):
		#	plt.axvline (self._beats[i], color = 'crimson')
		plt.title ("Pitches")
		plt.show()
		#plt.plot (self._confidences)
		#plt.show()
		self._cleaned_pitches = self._pitches
		self._cleaned_pitches = np.ma.masked_where(self._confidences < self.tolerance, self._cleaned_pitches)
		plt.plot (self._pitches)
		plt.title ("Cleaned Pitches -- tolerance: "+str(self.tolerance))
		plt.show()


	def median_filtering (self, k):
		assert k % 2 == 1, "Median filer must be odd"

		k2 = (k-1)//2
		x = self._cleaned_pitches
		y = np.zeros ((len(x), k)) 
		y [:, k2] = x

		for i in range (k2):
			j = k2 - 1
			y[j:, i] = x[:-j]
			y[:j, i] = x[0]
			y[:-j, -(i+1)] = x[j:]
			y[-j:, -(i+1)] = x[-1]

		self._filtered_pitches = np.median (y, axis=1)
		self._k_filter = k

	def discretize (self):
		if (len(self._filtered_pitches) > 0):
			self._discrete_pitches = np.round (self._filtered_pitches)
		else:
			self._discrete_pitches = np.round (self._cleaned_pitches)

	def convert_to_note (self):
		# convert midi note number to note name, e.g. [0, 127] -> [C-1, G9] "
		if (self._discrete_pitches == None):
			self.discretize ()

		self._note_series = []

		for i in range (len(self._discrete_pitches)):
			self._note_series.append (self._valid_notenames[int(self._discrete_pitches[i]) % 12] + str(int(self._discrete_pitches[i]) / 12 - 1))

		print self._note_series

	def fourier_filter (self):

		self._fft = np.fft.fftshift(np.fft.fft (self._waveform))
		dt = np.mean(np.diff(self._t))
		df = 1./dt
		self._f = np.linspace (-.5, .5, len(self._waveform))*df

		#plt.plot (self._f, (self._fft)**2)
		#plt.show()

		#tenor filter
		ind = np.where(np.abs(self._f)<self.min_freq)
		self._fft [ind] = 0*self._fft[ind]
		ind = np.where(np.abs(self._f)>self.max_freq)
		self._fft [ind] = 0*self._fft[ind]

		#back to the time domain
		self._waveform = (np.fft.ifft (np.fft.ifftshift (self._fft)))

	def set_vocal_range (self, range):

		if (range == 'tenor'):
			self.min_freq = 130
			self.max_freq = 500
		elif (range == 'soprano'):
			self.min_freq = 130
			self.max_freq = 500
		elif (range == 'alto'):
			self.min_freq = 130
			self.max_freq = 500
		elif (range == 'bass'):
			self.min_freq = 130
			self.max_freq = 500
		elif (range == 'bariton'):
			self.min_freq = 130
			self.max_freq = 500

	def derivative (self):

		self._deriv = self._cleaned_pitches [1:] - self._cleaned_pitches [0:-1]
		#plt.plot (self._deriv)
		#plt.title ('derivative')
		#plt.show()

		t = np.arange (len (self._cleaned_pitches))
		ind = np.where (np.abs(self._deriv > 5))
		plt.plot (t, self._cleaned_pitches, 'RoyalBlue')
		plt.plot (t [ind], self._cleaned_pitches [ind], '.', color = 'Crimson')
		plt.show()

		for i in range(len(ind)):
			self._cleaned_pitches[ind[i]+1] = self._cleaned_pitches[ind[i]]

		#plt.plot (t, self._cleaned_pitches, 'RoyalBlue')
		#plt.plot (t [ind], self._cleaned_pitches [ind], '.', color = 'Crimson')
		#plt.title ("Derivative-cleaning")
		#plt.show()

	def step_detection (self):

		i = 0
		self._std_pitch = []
		shift = self._step_window_shift
		l = self._step_window_frame
		self._note_pos = []
		self._note_midi = []

		self._pitches = self._filtered_pitches 
		print "Step detection"
		print len (self._pitches)
		while True:
			curr_bin = self._pitches [i*shift:i*shift+l]
			s = np.std(curr_bin)
			self._std_pitch.append (s)
			if (s < self._std_thr):
				self._note_pos.append (int(i*shift))
				self._note_midi.append (np.mean(self._pitches [i*shift:i*shift+l]))
			i += 1
			if i*shift+l>=len(self._pitches):
				break

		#Remove tones out of vocal range +- 10%

		#find a way to remove edge if there is a sharp increase

		print "Note positions", self._note_pos
		print "Notes:", self._note_midi
		
		plt.plot (self._note_pos, self._note_midi, 'o', color='RoyalBlue')
		plt.show()
		#plt.plot (self._std_pitch, 'o', color = 'Crimson')
		#plt.show()
		#self._std_pitch = np.array (self._std_pitch).flatten()
		#plt.plot (self._std_pitch)
		#plt.show()

		#aggregate chunks with same tone
		curr_chunk = 0
		curr_bound = 1
		self._note_dict = {}
		i = 0
		while (curr_bound < len(self._note_pos)):
			curr_bin = self._pitches [self._note_pos[curr_chunk]:self._note_pos[curr_bound]]
			if (np.std(curr_bin) < self._std_thr):
				curr_bound += 1
				curr_mean = np.mean (curr_bin)
				curr_std = np.std (curr_bin)
				#print curr_chunk, curr_bound
			else:
				print "std = ", curr_std
				note_name = self._valid_notenames[int(round(curr_mean)) % 12] + str(int(round(curr_mean)) / 12 - 1)
				self._note_dict [i] = {'mean':curr_mean, 'std':curr_std, 'note': note_name}
				#'pitches':self._pitches[self._note_pos[curr_chunk]:self._note_pos[curr_bound]],
				print "new chunk! ", self._note_pos [curr_bound]
				curr_chunk = curr_bound
				curr_bound = curr_chunk + 1
				print "Note:", note_name, " --- err: ", str(abs(curr_mean - round(curr_mean))*100), "%"
				i += 1

		print "Notes dictionary", self._note_dict

wf = AudioWaveform()
#wf.load_wav ()
wf.record_waveform()

wf.plot_waveform()
wf.fourier_filter()
wf.remove_silence(threshold = 0.10, binsize = 200)
wf.plot_waveform ()


#wf.extract_beats ()
#wf.plot_waveform()

wf.pitch_extraction(tolerance =0.98)
wf.median_filtering (k=15)

wf.derivative ()
wf.derivative ()
wf.derivative ()
wf.derivative ()
wf.derivative ()
wf.derivative ()

wf.median_filtering (k=15)

plt.plot (wf._filtered_pitches)
plt.title ("Filtered pitches")
plt.show()

#wf.discretize()

#plt.plot (wf._discrete_pitches%12)
#plt.title ("Discretization")
#plt.show()

wf.step_detection ()
#wf.convert_to_note ()
