import sys
from numpy.testing import *
import numpy as N
import random, scipy.signal
import testesns

# TODO: right module and path handling
sys.path.append("../")
from aureservoir import *


class test_delaysum(NumpyTestCase):

    def setUp(self):
	
	# parameters
	self.size = 10
	self.ins = 1
	self.outs = 1
	self.conn = 0.8
	
	# C++ ESN parameters
	self.netA = DoubleESN() # only for double here
	self.netA.setInitParam(DS_MAXDELAY, 100)
	self.netA.setReservoirAct(ACT_TANH)
	self.netA.setOutputAct(ACT_LINEAR)
	self.netA.setSize( self.size )
	self.netA.setInputs( self.ins )
	self.netA.setOutputs( self.outs )
	self.netA.setInitParam(ALPHA, 0.8)
	self.netA.setInitParam(CONNECTIVITY, self.conn)
	self.netA.setInitParam(FB_CONNECTIVITY, 0.)
	self.netA.setSimAlgorithm(SIM_FILTER_DS)
	self.netA.setTrainAlgorithm(TRAIN_DS_PI)
	
	# python ESN parameters
	self.netB = testesns.DSESN()
	self.netB.maxdelay = 100
	self.netB.setReservoirAct(ACT_TANH)
	self.netB.setOutputAct(ACT_LINEAR)
	self.netB.setSize( self.size )
	self.netB.setInputs( self.ins )
	self.netB.setOutputs( self.outs )
	self.netA.setInitParam(ALPHA, 0.8)
	self.netB.setInitParam(CONNECTIVITY, self.conn)
	self.netB.setInitParam(FB_CONNECTIVITY, 0.)
	self.netB.setSimAlgorithm( SIM_FILTER )
	self.netB.setTrainAlgorithm( TRAIN_PI )
	self.netB.setStdESN()
	

    def _linearIIR(self,x,delay=100):
	""" a linear IIR system with an additional delay for testing
	"""
	size = len(x)
	y = N.zeros(x.shape)
	for n in range(3,size-delay):
		ny = n+delay
		y[ny] = 0.5*x[n] - 0.8*x[n-1] + 0.4*x[n-2] \
		        - 0.3*y[ny-1] - 0.1*y[ny-2] + 0.05*y[ny-3]
	return y


    def testDelaysGCC(self, level=1):
	""" test simulation, delay and Wout calculation with
	    GCC method """
        
	# init network
	self.netA.setInitParam(DS_USE_GCC)
	self.netB.gcctype = 'phat'
	self.netB.squareupdate = 0
	self.netA.init()
	self.netB.init()
	
	# set internal data of netB to the same as in netA
	self.netB.setWin( self.netA.getWin().copy() )
	W = N.zeros((self.size,self.size))
	self.netA.getW(W)
	self.netB.setW(W)
	
	# training data
	washout = 20
	iir_delay = 20
	train_size = 100
	indata = N.random.rand(train_size) * 2 - 1
	outdata = self._linearIIR(indata,iir_delay)
	indata.shape = 1,-1
	outdata.shape = 1,-1
	
	# train data with python ESN
	self.netB.train(indata, outdata, washout)
	delaysB = self.netB.delays
	#print "trained delays in python:",delaysB
	woutB = self.netB.getWout().copy()
	
	# finally train C++ network with the same data
	self.netA.train(indata, outdata, washout)
	delaysA = N.ones((self.outs,self.ins+self.size))
	self.netA.getDelays(delaysA)
	#print "trained delays:",delaysA.flatten()
	woutA = self.netA.getWout().copy()
	
	# test if delays and output weights are the same
	assert_array_almost_equal(delaysA.flatten(),delaysB,5)
	assert_array_almost_equal(woutA,woutB,5)
	
	# simulation data
	sim_size = 50
	indata = N.random.rand(sim_size) * 2 - 1
	indata.shape = 1,-1
	outA = N.zeros( indata.shape )
	outB = N.zeros( indata.shape )
	
	# simulate both networks
	self.netB.simulate(indata, outB)
	self.netA.simulate(indata, outA)
	
	# test if simulation result is the same
	assert_array_almost_equal(outA,outB,5)


    def testDelaysCrosscorrFeedback(self, level=1):
	""" test simulation, delay and Wout calculation with
	    crosscorrelation method and feedback """
        
	# init network
	self.netA.setReservoirAct(ACT_LINEAR)
	self.netB.setReservoirAct(ACT_LINEAR)
	self.netA.setInitParam(FB_CONNECTIVITY, 0.3)
	self.netB.setInitParam(FB_CONNECTIVITY, 0.3)
	self.netA.setInitParam(DS_USE_CROSSCORR)
	self.netB.gcctype = 'unfiltered'
	self.netB.squareupdate = 0
	self.netA.init()
	self.netB.init()
	
	# set internal data of netB to the same as in netA
	self.netB.setWin( self.netA.getWin().copy() )
	self.netB.setWback( self.netA.getWback().copy() )
	W = N.zeros((self.size,self.size))
	self.netA.getW(W)
	self.netB.setW(W)
	
	# training data
	washout = 20
	iir_delay = 20
	train_size = 100
	indata = N.random.rand(train_size) * 2 - 1
	outdata = self._linearIIR(indata,iir_delay)
	indata.shape = 1,-1
	outdata.shape = 1,-1
	
	# train data with python ESN
	self.netB.train(indata, outdata, washout)
	delaysB = self.netB.delays
	#print "trained delays in python:",delaysB
	woutB = self.netB.getWout().copy()
	
	# finally train C++ network with the same data
	self.netA.train(indata, outdata, washout)
	delaysA = N.ones((self.outs,self.ins+self.size))
	self.netA.getDelays(delaysA)
	#print "trained delays:",delaysA.flatten()
	woutA = self.netA.getWout().copy()
	
	# test if delays and output weights are the same
	assert_array_almost_equal(delaysA.flatten(),delaysB,5)
	assert_array_almost_equal(woutA,woutB,5)
	
	# simulation data
	sim_size = 50
	indata = N.random.rand(sim_size) * 2 - 1
	indata.shape = 1,-1
	outA = N.zeros( indata.shape )
	outB = N.zeros( indata.shape )
	
	# simulate both networks
	self.netB.simulate(indata, outB)
	self.netA.simulate(indata, outA)
	
	# test if simulation result is the same
	assert_array_almost_equal(outA,outB,5)


# single neuron sin-osc machen

# two-neuron multiple sin-osc ? (einfach mit starken bp filter ?)

# mehrere outputs halt auch irgendwie testen
# -> wie ?
# -> z.B. auf beiden outputs das gleiche und schaun ob dann 2 mal das gleich
#    rauskommt !


if __name__ == "__main__":
    NumpyTest().run()
