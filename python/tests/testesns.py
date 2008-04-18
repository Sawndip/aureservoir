import numpy as N
from aureservoir import *
import filtering, util
import pylab as P
from scipy import signal
from scipy.linalg import pinv, inv, norm
import scipy.lib.lapack.clapack


class DelLine():
	""" helper class for a delay line """
	
	def __init__(self, delay, initvalues):
		""" init storage and make some checks
		delay         the delay in samples
		initvalues    the initial values in the delayline
		"""
		if( len(initvalues) != delay ):
			raise ValueError, "wrong size of initvalues !"
		self.delay = delay
		self.ringbuf = initvalues.copy()
		self.readpt = 0
		
	def tic(self, insample):
		""" one step - put in insample and returns output with
		the desired delaytime.
		"""
		if( self.delay == 0 ):
			return insample
		else:
			outsample = self.ringbuf[self.readpt]
			self.ringbuf[self.readpt] = insample
			self.readpt = (self.readpt+1) % self.delay
			return outsample


class IIRESN(DoubleESN):
	""" ESN with general IIR Filter neurons.
	2007, Georg Holzmann
	"""
	
	B = N.empty((1,1))  # feedforward path
	A = N.empty((1,1))  # feedback path
	serial = 1          # serial IIR filters
	fc = N.empty((1,1)) # center frequencies (of e.g. bandpass filters)
	
	def setStdESN(self):
		""" set filter coefficients so that we get a standard
		(unfiltered) ESN
		"""
		size = self.getSize()
		self.B = N.zeros((size,2))
		self.A = N.zeros((size,2))
		self.B[:,0] = 1.
		self.A[:,0] = 1.
	
	def setGammatoneFilters(self,fs,lowFreq):
		""" Set to each neuron one gammatone filter so that the filter
		frequencies will reach from lowFreq to fs/2 (each neuron will
		have a different gammatone filter !)
		
		fs         sample rate
		lowFreq    lowest filter frequency
		"""
		size = self.getSize()
		B, A = filtering.gammatone_biquad(fs,size,lowFreq)
		
		# 4 biquads
		self.B = N.empty((size,12))
		self.A = N.empty((size,12))
		self.serial = 4
		self.B[:,0:3] = B[:,0,:]
		self.B[:,3:6] = B[:,1,:]
		self.B[:,6:9] = B[:,2,:]
		self.B[:,9:12] = B[:,3,:]
		self.A[:,0:3] = A[:,0,:]
		self.A[:,3:6] = A[:,1,:]
		self.A[:,6:9] = A[:,2,:]
		self.A[:,9:12] = A[:,3,:]
		
	
	def setERB(self,fs,lowFreq):
		""" Set to each neuron to an Equivalent Rectangular Bandwidth
		Bandpass Filter.
		
		fs         sample rate
		lowFreq    lowest filter frequency
		"""
		size = self.getSize()
		self.B, self.A = filtering.ERB(fs,size,lowFreq)
	
	def setConstBPCutoffs(self, f=[0.1,0.3], bw=0.2, fs=1.):
		""" f is a list with bandpass centerfrequencies, bw the
		bandwidth in octaves
		"""
		size = self.getSize()
		nr = len(f)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		self.fc = N.zeros(size)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,f[n%nr],'BPF',BW=bw)
			self.fc[n] = f[n%nr] / fs
	
	def setConstParametric(self, f=[0.1,0.3], bw=1., stopGain=-20, fs=1.):
		""" f is a list with bandpass centerfrequencies, bw the
		bandwidth in octaves, stopGain the gain in the stop bands
		"""
		size = self.getSize()
		nr = len(f)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		self.fc = N.zeros(size)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,f[n%nr],'peakingEQ',BW=bw, \
			                 dBgain=-stopGain)
			self.fc[n] = f[n%nr] / fs
		
		# normalize parametric EQ to 0 dB
		g = 10**(-stopGain/20.)
		self.B = self.B / g
		
	def setLogBPCutoffs(self, f_start=0.001, f_stop=0.5, bw=0.2, fs=1.):
		""" set log spaced bandpass filters from f_start to f_stop
		with a bandwidth of bw octaves
		"""
		size = self.getSize()
		self.fc = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               size, True)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		self.fc[n] /= fs
	
	def setLogParametric(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		             stopGain=-20, fs=1.):
		""" set log spaced parametric filters from f_start to f_stop
		with a bandwidth of bw octaves and stopGain in the stop bands
		"""
		size = self.getSize()
		self.fc = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               size, True)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'peakingEQ',BW=bw, \
			                 dBgain=-stopGain)
		self.fc[n] /= fs
		
		# normalize parametric EQ to 0 dB
		g = 10**(-stopGain/20.)
		self.B = self.B / g
	
	def setLinBPCutoffs(self, f_start=0.001, f_stop=0.5, bw=0.2, fs=1.):
		""" set linear spaced bandpass filters from f_start to f_stop
		with a bandwidth of bw octaves
		"""
		size = self.getSize()
		self.fc = N.linspace(f_start, f_stop, size)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		self.fc[n] /= fs
	
	def setLinParametric(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		             stopGain=-20, fs=1.):
		""" set linear spaced parametric filters from f_start to f_stop
		with a bandwidth of bw octaves and stopGain in the stop bands
		"""
		size = self.getSize()
		self.fc = N.linspace(f_start, f_stop, size)
		self.B = N.zeros((size,3))
		self.A = N.zeros((size,3))
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'BPF',BW=bw)
		for n in range(size):
			self.B[n],self.A[n] = \
			filtering.biquad(fs,self.fc[n],'peakingEQ',BW=bw, \
			                 dBgain=-stopGain)
		self.fc[n] /= fs
		
		# normalize parametric EQ to 0 dB
		g = 10**(-stopGain/20.)
		self.B = self.B / g
	
	def setGroupBP(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		       groups=10, spacing="log", fs=1.):
		""" groups neurons in groups with same filters, the spacing can
		be log or lin
		"""
		size = self.getSize()
		if spacing == "lin":
			f = N.linspace(f_start, f_stop, groups)
		elif spacing == "log":
			f = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               groups, True)
		else:
			raise ValueError, "unknown spacing"
		self.setConstBPCutoffs(f, bw, fs)
	
	def setGroupParametric(self, f_start=0.001, f_stop=0.5, bw=0.2, \
		       stopGain=-20, groups=10, spacing="log", fs=1.):
		""" groups neurons in groups with same filters, the spacing can
		be log or lin
		"""
		size = self.getSize()
		if spacing == "lin":
			f = N.linspace(f_start, f_stop, groups)
		elif spacing == "log":
			f = N.logspace(N.log10(f_start), N.log10(f_stop), \
		               groups, True)
		else:
			raise ValueError, "unknown spacing"
		self.setConstParametric(f, bw, stopGain, fs)
	
	def setRandomFIR(self, order=2):
		""" generates minimum-phase FIR filters with random zeros
		for each neuron
		order = nr of zeros in one filter
		"""
		size = self.getSize()
		zeros = (N.random.rand(size,order)*2-1) + \
		        (N.random.rand(size,order)*2-1)*1j
		k = 1.
		self.B = N.zeros((size,order+1))
		self.A = N.zeros((size,order+1))
		for n in range(size):
			self.B[n],self.A[n] = \
			signal.zpk2tf(zeros[n], [], k)
	
	def setRandomIIR(self, order=2):
		""" generates minimum-phase IIR filters with random zeros
		and poles for each neuron
		order = nr of zeros/poles in one filter
		"""
		size = self.getSize()
		zeros = (N.random.rand(size,order)*2-1) + \
		        (N.random.rand(size,order)*2-1)*1j
		poles = (N.random.rand(size,order)*2-1) + \
		        (N.random.rand(size,order)*2-1)*1j
		k = 1.
		self.B = N.zeros((size,order+1))
		self.A = N.zeros((size,order+1))
		for n in range(size):
			self.B[n],self.A[n] = \
			signal.zpk2tf(zeros[n], poles[n], k)
		
	def init(self):
		""" overrides the ESN init() method to additionaly set the
		bandpass frequencies """
		
		if( self.B.size == 1 or self.A.size == 1 ):
			print "IIRESN filter parameters not initialized !"
		
		size = self.getSize()
		
		DoubleESN.init(self)
		self.setIIRCoeff(self.B,self.A,self.serial)
	
	def plot_filters(self,every=1,fftsize=16384):
		""" plots the currently set filters
		"""
		neurons = self.getSize()
		impulse = N.zeros(fftsize)
		impulse[0] = 1
		response = N.empty((neurons,fftsize/2+1))
		
		# get frequency response of neurons
		if self.serial == 1:
			for n in range(0,neurons,every):
				tmp = signal.lfilter(self.B[n],self.A[n], impulse)
				response[n] = N.fft.rfft( tmp )
		else:
			fstep = self.B.shape[1] / self.serial
			for n in range(0,neurons,every):
				tmp = signal.lfilter(self.B[n,0:fstep], \
				                     self.A[n,0:fstep], \
				                     impulse)
				for i in range(1,self.serial):
					beg = i*fstep
					end = (i+1)*fstep
					#print i, beg, end
					tmp = signal.lfilter(self.B[n,beg:end], \
					                     self.A[n,beg:end], \
							     tmp )
				response[n] = N.fft.rfft( tmp )
		
		# color map
		col = ['b','g','r','c','m','y','k']
		
		# plot amplitude response
		le = len(abs(response[0]))
		freqscale = N.arange(le) / (float(le)/(20000))
		denormal = 1e-20
		for n in range(0,neurons,every):
			tmp = 20.*N.log10(abs(response[n])+denormal)
			#tmp = abs(response[n])
			#P.title("log spaced bandpass filters, BW = 1.5 octaves")
			P.title("parametric EQs, BW = 2 octaves, stop = -20dB")
			P.semilogx(freqscale,tmp,col[n%len(col)])
			#P.axis([20, 20000, -50, 5])
			P.axis([100, 20000, -30, 5])
			P.xlabel("frequency (Hz)")
			P.ylabel("amplitude (dB)")
		P.show()


class DSESN(IIRESN):
	""" ESN class with delay+sum readout.
	
	2007-2008, Georg Holzmann
	"""
	
	# maximum delay of the delayline
	maxdelay = 100
	
	# prefiltering of the cross correlation
	gcctype = 'unfiltered'
	
	# additional squared state updates
	squareupdate = 0
	
	# maximum delay for delays in reservoir
	maxreservoirdelay = 10
	
	# set this to 1 to also have delays in the reservoir
	have_res_delays = 0
	
	# iterations for EM based delay+weight calculation
	emiterations = 0
	
	
	def train(self, indata, outdata, washout):
		""" set to desired train method here """
		#DoubleESN.train(self,indata,outdata,washout)
		if self.emiterations==0:
			self.trainDelaySum(indata,outdata,washout)
		else:
			self.trainDelaySumEM(indata,outdata,washout)
	
	
	def simulate(self, indata, outdata):
		""" set to desired simulate method here """
		self.simulateDelaySum(indata,outdata)
		#DoubleESN.simulate(self,indata,outdata)
	
	
	def initDelay(self,Wdel):
		""" initialization of delays in the reservoir
		"""
		size = self.getSize()
		self.W = N.zeros((size,size))
		self.getW(self.W)
		
		# init and create delay lines for the reservoir
		self.Wdel = []
		for i in range(size):
			for j in range(size):
				d = int( Wdel[i,j] )
				self.Wdel.append( DelLine(d, N.zeros(d)) )
		
		# for simulation algorithm
		self.lastout = N.zeros( self.getOutputs() )
		self.Win = self.getWin()
		self.Wout = self.getWout()
		self.Wback = self.getWback()
	
	
	def trainDelaySum(self, indata, outdata, washout):
		""" Calculates the optimal delay for each readout neuron with
		the crosscorrelation.
		"""
		steps = indata.shape[1]
		size = self.getSize()
		maxdelay = min(self.maxdelay+1,steps-washout)
		insig = indata.copy()
		
		# teacher forcing
		X = self._teacherForcing(insig,outdata)
		#print "indata[0]",indata.shape,indata[0]
		#print "X[0]",X.shape,X[0]
		
		self.delays = N.zeros(size+self.getInputs())
		for n in range(size):
			xcorr = abs( util.GCC(X[n],outdata[0], \
			             self.gcctype, 0) )
			self.delays[n] = xcorr[0:maxdelay].argmax()
		for n in range(self.getInputs()):
			xcorr = abs( util.GCC(insig[n],outdata[0], \
			             self.gcctype, 0) )
			self.delays[n+size] = xcorr[0:maxdelay].argmax()
		del xcorr
		
		# setup the "delay lines" for each neuron
		self.dellines = []
		
		# now delay neuron signals as calculated:
		# add zeros to the beginning and store the rest into the delaylines
		for n in range(size):
			d = int(self.delays[n])
			rest = X[n,steps-d:steps].copy()
			X[n,:] = N.r_[ N.zeros(d), X[n,0:steps-d] ]
			self.dellines.append( DelLine(d,rest) )
		for n in range(self.getInputs()):
			d = int(self.delays[n+size])
			rest = insig[n,steps-d:steps].copy()
			insig[n,:] = N.r_[ N.zeros(d), insig[n,0:steps-d] ]
			self.dellines.append( DelLine(d,rest) )
		del rest
		
		# some debugging
		#print "delays:", self.delays
		
		# calc new washout beacuse of the delay
		# (is this good ? - should be better given from outside)
		#washout += int(self.delays.max())
		#print "effective trainsize:",steps-washout
		
		if (self.squareupdate == 0):
			# restructure data
			M = N.r_[X,insig]
			M = M[:,washout:steps].T
			T = outdata[:,washout:steps].T
			
			# calc pseudo inverse: wout = pinv(M) * T
			# or with least square (much faster):
			v,wout,s,rank,info = scipy.lib.lapack.clapack.dgelss( M, T )
			self.wout = wout[0:size+self.getInputs()].T	
		else:
			# restructure data
			M = N.r_[X,insig,X**2,insig**2]
			M = M[:,washout:steps].T
			T = outdata[:,washout:steps].T
			
			# calc pseudo inverse: wout = pinv(M) * T
			# or with least square (much faster):
			v,wout,s,rank,info = scipy.lib.lapack.clapack.dgelss( M, T )
			self.wout = wout[0:2*(size+self.getInputs())].T
		
		# set the readout to the radius
		self.setWout(self.wout.copy())


	def trainDelaySumEM(self, indata, outdata, washout):
		""" EM algorithm for delay+weight learning
		"""
		if( outdata.shape[0] > 1 ):
			raise ValueError, "ATM only for 1 output !"
		
		steps = indata.shape[1]
		size = self.getSize()
		insig = indata.copy()
		
		# teacher forcing
		X = self._teacherForcing(insig,outdata)
		
		# restructure data
		M = N.r_[X,insig]
		M = M.T
		T = outdata[0,washout:steps].flatten()
		
		self.wout,self.delays = self._delaylearning_em(T,M,washout, \
		                                          self.emiterations)
		
		# shift reservoir signals with the right delay
		K = N.zeros(( steps-washout, M.shape[1] ))
		for n in range( M.shape[1] ):
			K[:,n] = M[washout-self.delays[n]:steps-self.delays[n],n]
		
		if (self.squareupdate == 0):
			# calc least square
			v,wout,s,rank,info = scipy.lib.lapack.clapack.dgelss( K, T )
			self.wout = wout[0:size+self.getInputs()].T
		else:
			# restructure data
			M = N.c_[K,K**2]
			# calc least square
			v,wout,s,rank,info = scipy.lib.lapack.clapack.dgelss( M, T )
			self.wout = wout[0:2*(size+self.getInputs())].T
		self.wout.shape = 1,-1
		self.setWout( self.wout.copy() )
	
	
	def simulateDelaySum(self, indata, outdata):
		""" Simulation algorithm for a DelayAndSum readout.
		"""
		#print "TestESN: Simulate DelaySum ..."
		
		steps = indata.shape[1]
		outs = self.getOutputs()
		outtmp = N.empty((outs,))
		
		for n in range(steps):
			self.simulateDelaySumStep(indata[:,n].flatten(),outtmp)
			outdata[:,n] = outtmp
	
	
	def simulateDelaySumStep(self, indata, outdata):
		""" One-Step Simulation algorithm for a DelayAndSum readout.
		"""
		neurons = self.getSize()
		outtmp = N.empty((self.getOutputs(),))
		
		# simulate one step
		if self.have_res_delays==0:
			self.simulateStep(indata, outtmp)
		else:
			self._simulateStepDelay(indata, outtmp)
		x = self.getX().copy()
		
		# get delayed value for each neuron and input
		for i in range(neurons):
			x[i] = self.dellines[i].tic(x[i])
		for i in range(self.getInputs()):
			indata[i] = self.dellines[neurons+i].tic(indata[i])
		
		if self.squareupdate==0:
			state = N.r_[x,indata]
		else:
			state = N.r_[x,indata,x**2,indata**2]
		outdata[:] = N.dot( self.wout, state ).copy()
		self.setLastOutput(outdata)
		self.lastout = outdata
	
	
	def _teacherForcing(self, indata, outdata):
		""" teacher forcing and collect internal states
		"""
		#print "TestESN: teacher forcing ..."
		
		steps = indata.shape[1]
		neurons = self.getSize()
		X = N.empty((neurons,steps))
		outtmp = N.empty((self.getOutputs(),))
		ins = self.getInputs()
		outs = self.getOutputs()
		
		# step by step ESN simulation
		for n in range(steps):
			if self.have_res_delays==0:
				self.simulateStep( \
				     indata[:,n].flatten(), outtmp)
				self.setLastOutput(outdata[:,n].flatten())
			else:
				self._simulateStepDelay( \
				      indata[:,n].flatten(), outtmp)
				self.lastout = outdata[:,n].flatten().copy()
			X[:,n] = self.getX().copy()
		
		return X
	
	
	def _simulateStepDelay(self, indata, outtmp):
		""" simulate one step with delays in the reservoir
		ATTENTION: only for tanh reservoir activation function !
		"""
		x = self.getX().copy()
		Wout = self.getWout().copy()
		
		# calc new network activation
		x = self._dotDelay( self.W, x )
		x += N.dot( self.Win, indata )
		x += N.dot( self.Wback, self.lastout )
		noise = self.getNoise()
		x += (N.random.rand(len(x))*2-1)*noise
		
		# reservoir activation function
		x = N.tanh( x )
		
		# finally set new calculated state
		self.setX(x)
	
	
	def _dotDelay(self, W, x):
		""" helper function for matrix multiplication with delays
		"""
		n = W.shape[0]
		t = N.zeros(n)
		for i in range(n):
			for j in range(n):
				t[i] += self.Wdel[i*n+j].tic(W[i,j]*x[j])
		return t
	
	
	def _delaylearning_em(self,z,M,washout=100,emiterations=1):
		""" EM algorithm to learn delay and weights
		
		input:
		z              target output data
		M              matrix with all input data from the reservoir neurons
		washout        ESN washout time
		emiterations   nr of emiterations
		
		output: (w,d)
		w              estimated weights
		d              estimated delays
		"""
		# nr of steps
		steps = M.shape[0]
		# nr of components
		L = M.shape[1]
		# matrix for delayd neuron signals without washout
		K = N.zeros(( steps-washout, L ))
		
		# allocate memory for weights and delays
		w = N.zeros(L)
		d = N.zeros(L, dtype='int32')
		
		#print "EM-based delay+weight parameter estimation ..."
		converged = 0
		ccount = 0
		while not converged:
			l = 0
			wold = w.copy()
			dold = d.copy()
			
			while l<L:
				########################
				# E-step
				
				# delay all neuron signals with delay from previous step
				for n in range(L):
					K[:,n] = M[washout-d[n]:steps-d[n],n]
				
				# now remove contribution from all other neurons
				x =  (z-N.dot(K,w))/L + K[:,l]*w[l]
				#x =  ( z-N.dot(K,w) + K[:,l]*w[l] ) / L
				#x =  z-N.dot(K,w) + K[:,l]*w[l]
				
				########################
				# M-step
				
				# current neuron-signal
				r = M[washout:steps,l]
				
				# estimate time delay between target signal x
				xcorr = abs( util.GCC(r,x,self.gcctype,0) )
				d[l] = xcorr[0:self.maxdelay+1].argmax()
				
				# delay current neuron signal with new delay
				r = M[washout-d[l]:steps-d[l],l]
				
				# calculate new weight w(l) with pseudo-inverse:
				pinvr = pinv(r.reshape(-1,1))
				w[l] = N.dot( pinvr, x)
				#print "\tl:",l," - wl:",w[l],", d:",d[l]
				
				l=l+1
			ccount=ccount+1
			
			# check if we shoud continue
			print "\titeration", ccount, ", |d-dold| diff =", \
			      abs(d-dold).sum(), "- average weight =", w.mean()
			
			# check if we shoud continue
			if ccount >= emiterations:
				converged = 1
			
		# init delaylines with final delays
		self.dellines = []
		for n in range(L):
			rest = M[steps-d[n]:steps,n].copy()
			self.dellines.append( DelLine(d[n],rest) )
		
		return w, d
