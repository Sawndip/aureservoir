#!/usr/bin/env python
###########################################################
# ESN generator of multiple sines
# possible with bandpass style neurons
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

def setup_ESN_STD():
	""" configuration of a standard ESN -
	does not work ! """
	net = DoubleESN()
	net.setSize(60)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 0.5 )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setSimAlgorithm( SIM_STD )
	net.setTrainAlgorithm( TRAIN_PI )
	trainnoise = 1e-4
	testnoise = 0.
	net.init()
	return net, trainnoise, testnoise

def setup_ESN_BP():
	""" configuration of a bandpass ESN """
	net = DoubleESN()
	net.setSize(100)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 0.6 )
	net.setInitParam( BP_F1, 0.01 )
	net.setInitParam( BP_F2, 0.01 )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setInitAlgorithm( INIT_BP_CONST )
	net.setSimAlgorithm( SIM_BP )
	net.setTrainAlgorithm( TRAIN_PI )
	#net.setTrainAlgorithm( TRAIN_RIDGEREG )
	#net.setInitParam( TIKHONOV_FACTOR, 1e-4 )
	trainnoise = 0.
	testnoise = 0.
	net.init()
	
	size = net.getSize()
	f1 = N.ones(size)
	f2 = N.ones(size)
	
	#set bandpass cutoffs:
	f1[0:size/2] = 0.01
	f2[0:size/2] = 0.01
	f1[size/2:1] = 0.25
	f2[size/2:1] = 0.25
	net.setBPCutoff(f1,f2)
	
	return net, trainnoise, testnoise

def setup_ESN_BP_octave():
	""" bandpass ESN with octave style cutoffs """
	net = DoubleESN()
	net.setSize(100)
	net.setInputs(1)
	net.setOutputs(1)
	net.setInitParam( CONNECTIVITY, 0.2 )
	net.setInitParam( ALPHA, 0.6 )
	net.setInitParam( IN_CONNECTIVITY, 0. )
	net.setInitParam( IN_SCALE, 0. )
	net.setInitParam( FB_CONNECTIVITY, 1. )
	net.setInitParam( FB_SCALE, 1. )
	net.setReservoirAct( ACT_TANH )
	net.setOutputAct( ACT_LINEAR )
	net.setInitAlgorithm( INIT_STD )
	net.setSimAlgorithm( SIM_BP )
	net.setTrainAlgorithm( TRAIN_PI )
	trainnoise = 0.
	testnoise = 0.
	net.init()
	
	# set octave style bandpass cutoffs
	size = net.getSize()
	f1 = N.empty(size)
	f2 = N.empty(size)
	subsize = size / 10
	for n in range(10):
		f1[n*subsize:(n+1)*subsize] = 0.2 * 2.**(-n)
		f2[n*subsize:(n+1)*subsize] = 0.1 * 2.**(-n)
	net.setBPCutoff(f1,f2)
	
	return net, trainnoise, testnoise

def generate_two_sines(size,ampl1=0.5,ampl2=0.5):
	""" generates a slow and a fast sinewave:
	    y1[n] = ampl1 * sin(n/100)
	    y2[n] = ampl2 * sin(n/4)
	    -> Periode1 = 628.318 = 2*pi*100
	       Periode2 = 25.1327 = 2*pi*4
	       """
	x = N.arange( float(size) )
	y1 = ampl1*N.sin( x/100 )
	y2 = ampl2*N.sin( x/4 )
	
	return y1+y2

def generate_three_sines(size):
	""" generates three sinewave:
	    y1[n] = 0.33 * sin(n/100)
	    y2[n] = 0.33 * sin(n/55)
	    y3[n] = 0.33 * sin(n/4)
	    """
	x = N.arange( float(size) )
	y1 = 0.33*N.sin( x/100 )
	y2 = 0.33*N.sin( x/55 )
	y3 = 0.33*N.sin( x/4 )
	return y1+y2+y3

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
# PROGRAM

trainsize = 4000
washout = 2000
testsize = 4000

# generate signals
#slsine = generate_two_sines(trainsize+testsize)
slsine = generate_three_sines(trainsize+testsize)
trainin, trainout, testin, testout = get_esn_data(slsine,trainsize,testsize)

# ESN training
#net, trainnoise, testnoise = setup_ESN_STD()
#net, trainnoise, testnoise = setup_ESN_BP()
net, trainnoise, testnoise = setup_ESN_BP_octave()
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
