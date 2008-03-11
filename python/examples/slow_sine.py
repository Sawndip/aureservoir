###########################################################
# slow sine genration task with standard ESN
# with standard and bandpass ESNs
#
# 2007, Georg Holzmann
###########################################################

import numpy as N
import pylab as P
from aureservoir import *
import sys
sys.path.append("../")
from aureservoir import *


###########################################################
# FUNCTIONS

def setup_STD_ESN():
	""" configuration of a standard ESN
	"""
	net = DoubleESN()
	net.setSize(20)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 0.45 )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setSimAlgorithm( SIM_STD )
	net.setTrainAlgorithm( TRAIN_PI )
	#net.setTrainAlgorithm( TRAIN_RIDGEREG )
	#net.setInitParam( TIKHONOV_FACTOR, 1e-4 )
	trainnoise = 1e-6
	testnoise = 0.
	return net, trainnoise, testnoise

def setup_ESN_LI():
	""" configuration of a leaky integrating ESN """
	net = DoubleESN()
	net.setSize(20)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 0.45*0.8 )
	net.setInitParam( LEAKING_RATE, 0.8 )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setSimAlgorithm( SIM_LI )
	net.setTrainAlgorithm( TRAIN_PI )
	#net.setTrainAlgorithm( TRAIN_RIDGEREG )
	#net.setInitParam( TIKHONOV_FACTOR, 1e-4 )
	trainnoise = 1e-6
	#trainnoise = 0.
	testnoise = 0.
	return net, trainnoise, testnoise

def generate_slow_sine(size,ampl=1):
	""" generates a slow sinewave:
	    y[n] = ampl * sin(n/100)
	    omega = 2*pi*f -> f = 0.0015915494309189536
	    -> Periode = 628.318 """
	x = N.arange( float(size) )
	y = ampl*N.sin( x/100 )
	return y

def get_esn_data(signal,trainsize,testsize):
	""" returns trainin, trainout, testin, testout """
	
	trainout = signal[0:trainsize]
	trainout.shape = 1,-1
	trainin = N.zeros(trainout.shape)
	
	testout = signal[trainsize:trainsize+testsize]
	testout.shape = 1,-1
	testin = N.zeros(testout.shape)
	
	return trainin, trainout, testin, testout

def nrmse( testsig, origsig, discard=0 ):
	""" calculates the NRMSE (normalized root mean square error) """
	# TODO: make for matrix in and target
	
	# reshape values
	testsig.shape = -1,
	origsig.shape = -1,
	
	error = (origsig - testsig)**2
	nrmse = N.sqrt( error.mean() / (origsig.var()**2) )
	
	return nrmse

def plot(esnout,testout):
	""" plotting """
	P.title('Original=blue, ESNout=red')
	P.plot(testout,'b',esnout,'r')
	P.show()


###########################################################
# MAIN

trainsize = 4000
washout = 2000
testsize = 8000

# choose ESN
#net, trainnoise, testnoise = setup_STD_ESN()
net, trainnoise, testnoise = setup_ESN_LI()
net.init()

# generate signals
slsine = generate_slow_sine(trainsize+testsize)
trainin, trainout, testin, testout = get_esn_data(slsine,trainsize,testsize)

# ESN training
net.setNoise(trainnoise)
net.train(trainin,trainout,washout)
print "output weights:"
print "\tmean: ", net.getWout().mean(), "\tmax: ", abs(net.getWout()).max()

# ESN simulation
esnout = N.empty(testout.shape)
net.setNoise(testnoise)
net.simulate(testin,esnout)
print "\nNRMSE: ", nrmse( esnout, testout, 50 )
plot(esnout,testout)
