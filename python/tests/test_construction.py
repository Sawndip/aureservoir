import sys
from numpy.testing import *
import numpy as N
import random

# TODO: right module and path handling
sys.path.append("../")
from aureservoir import *


class test_construction(NumpyTestCase):

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
	self.net.init()


    def testCopyConstructor(self, level=1):
	""" test if a copied net generates the same result """
        
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


if __name__ == "__main__":
    NumpyTest().run()
