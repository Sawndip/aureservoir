import sys
from numpy.testing import *
import numpy as N
import random, scipy.signal

# TODO: right module and path handling
sys.path.append("../")
from aureservoir import *


class test_correspondence(NumpyTestCase):

    def setUp(self):
	
	# parameters
	self.size = random.randint(10,15)
	self.ins = random.randint(1,5)
	self.outs = random.randint(1,5)
	self.conn = random.uniform(0.9,0.99)
	self.train_size = 25
	self.sim_size = 10
	self.dtype = 'float64'
	
	# construct network
	if self.dtype is 'float32':
		self.net = SingleESN()
	else:
		self.net = DoubleESN()
	
	# set parameters
	self.net.setReservoirAct(ACT_TANH)
	self.net.setOutputAct(ACT_TANH)
	self.net.setSize( self.size )
	self.net.setInputs( self.ins )
	self.net.setOutputs( self.outs )
	self.net.setInitParam(CONNECTIVITY, self.conn)
	self.net.setInitParam(FB_CONNECTIVITY, 0.5)
	self.net.setSimAlgorithm(SIM_STD)
	self.net.setTrainAlgorithm(TRAIN_PI)


    def testCopyConstructor(self, level=1):
	""" test if a copied net generates the same result """
        
	self.net.init()
	
	# set output weight matrix
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	
	# copy network
	# ATTENTION: operator= is shallow copy !
	if self.dtype is 'float32':
		netA = SingleESN(self.net)
	else:
		netA = DoubleESN(self.net)
	
	# test matrices
	W = N.empty((self.size,self.size),self.dtype)
	self.net.getW( W )
	WA = N.empty((self.size,self.size),self.dtype)
	netA.getW( WA )
	assert_array_almost_equal(W,WA)
	assert_array_almost_equal(self.net.getWback(),netA.getWback())
	assert_array_almost_equal(self.net.getWout(),netA.getWout())
	assert_array_almost_equal(self.net.getWin(),netA.getWin())
	assert_array_almost_equal(self.net.getX(),netA.getX())
	
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)


    def testCopyConstructorBP(self, level=1):
	""" test if a copied bandpass ESN generates the same result """
        
	# set bandpass parameters
	self.net.setSimAlgorithm(SIM_BP)
	f1 = N.linspace(0.1, 1., self.net.getSize())
	f2 = N.linspace(0.0001, 0.5, self.net.getSize())
	self.net.init()
	self.net.setBPCutoff(f1,f2)
	
	# set output weight matrix
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	
	# copy network
	# ATTENTION: operator= is shallow copy !
	if self.dtype is 'float32':
		netA = SingleESN(self.net)
	else:
		netA = DoubleESN(self.net)
	
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)



    def testSetInternalData(self, level=1):
	""" test if manually setting the weigth matrices generates the
	same result """
	
	self.net.init()
	
	# train first ESN
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	self.net.resetState()
	
	# create second ESN
	if self.dtype is 'float32':
		netA = SingleESN()
	else:
		netA = DoubleESN()
	netA.setReservoirAct(ACT_TANH)
	netA.setOutputAct(ACT_TANH)
	netA.setSize( self.size )
	netA.setInputs( self.ins )
	netA.setOutputs( self.outs )
	netA.setSimAlgorithm(SIM_STD)
	netA.setTrainAlgorithm(TRAIN_PI)
	
	# set internal data in second ESN
	netA.setWin( self.net.getWin().copy() )
	netA.setWout( self.net.getWout().copy() )
	netA.setWback( self.net.getWback().copy() )
	W = N.empty((self.size,self.size),self.dtype)
	self.net.getW( W )
	netA.setW( W )
	netA.setX( self.net.getX().copy() )
		
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)


    def testIIRFilters(self, level=1):
	""" test correspondence of SIM_FILTER and pythons lfilter """
        
	# set parameters for non-interacting neurons:
	# new state only depends on the input
	self.net.setSimAlgorithm(SIM_FILTER)
	self.ins = 1
	self.outs = 1
	self.net.setReservoirAct(ACT_LINEAR)
	self.net.setOutputAct(ACT_LINEAR)
	self.net.setInputs( self.ins )
	self.net.setOutputs( self.outs )
	self.net.setInitParam(ALPHA, 0.)
	self.net.setInitParam(FB_CONNECTIVITY, 0.)
	self.net.setInitParam(IN_CONNECTIVITY, 1.)
	self.net.setInitParam(IN_SCALE, 0.)
	self.net.setInitParam(IN_SHIFT, 1.)
	self.net.init()
	
	# set paramas for a biquad bandpass filter:
	B = N.empty((self.size,3))
	A = N.empty((self.size,3))
	B[:,0] = 0.29573818
	B[:,1] = 0.
	B[:,2] = -0.29573818
	A[:,0] = 1.29573818
	A[:,1] = -1.84775907
	A[:,2] = 0.70426182
	self.net.setIIRCoeff(B,A)
	
	# simulate network step by step:
	# the states x are now only the filtered input signal, which
	# is the same for each neuron ! (because ALPHA = 0)
	indata = N.random.rand(1,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	filterout = N.zeros((1,self.sim_size),self.dtype)
	outtmp = N.zeros((self.outs),self.dtype)
	for n in range(self.sim_size):
		intmp = indata[:,n].copy()
		self.net.simulateStep( intmp, outtmp )
		filterout[0,n] = self.net.getX()[0]
	
	# now calculate the same with scipy.signal.lfilter
	filterout2 = scipy.signal.lfilter(B[0], A[0], indata.flatten())
	
	assert_array_almost_equal(filterout.flatten(),filterout2)


    def testIIRvsSTDESN(self, level=1):
	""" test if IIR-ESN with b=1 and a=1 gives same result as
	standard ESN """
	
	self.net.init()
	
	# train first ESN
	trainin = N.random.rand(self.ins,self.train_size) * 2 - 1
	trainout = N.random.rand(self.outs,self.train_size) * 2 - 1
	trainin = N.asfarray(trainin, self.dtype)
	trainout = N.asfarray(trainout, self.dtype)
	self.net.train(trainin,trainout,1)
	self.net.resetState()
	
	# create second ESN
	if self.dtype is 'float32':
		netA = SingleESN()
	else:
		netA = DoubleESN()
	netA.setReservoirAct(ACT_TANH)
	netA.setOutputAct(ACT_TANH)
	netA.setSize( self.size )
	netA.setInputs( self.ins )
	netA.setOutputs( self.outs )
	netA.setSimAlgorithm(SIM_FILTER)
	netA.setTrainAlgorithm(TRAIN_PI)
	B = N.zeros((self.size,2))
	A = N.zeros((self.size,2))
	B[:,0] = 1.
	A[:,0] = 1.
	netA.init()
	netA.setIIRCoeff(B,A)
	
	# set internal data in second ESN
	netA.setWin( self.net.getWin().copy() )
	netA.setWout( self.net.getWout().copy() )
	netA.setWback( self.net.getWback().copy() )
	W = N.empty((self.size,self.size),self.dtype)
	self.net.getW( W )
	netA.setW( W )
	netA.setX( self.net.getX().copy() )
	
	# simulate both networks separate and test result
	indata = N.random.rand(self.ins,self.sim_size)*2-1
	indata = N.asfarray(indata, self.dtype)
	outdata = N.empty((self.outs,self.sim_size),self.dtype)
	outdataA = N.empty((self.outs,self.sim_size),self.dtype)
	self.net.simulate( indata, outdata )
	netA.simulate( indata, outdataA )
	assert_array_almost_equal(outdata,outdataA)


if __name__ == "__main__":
    NumpyTest().run()
