
// File: classaureservoir_1_1ESN.xml
%feature("docstring") ESN "

class for a basic Echo State Network

This class implements a basic Echo State Network as described in
articles by Herbert Jaeger on the following page: See:
http://www.scholarpedia.org/article/Echo_State_Network  The template
argument T can be float or double. Single Precision (float) saves
quite some computation time.

The \"echo state\" approach looks at RNNs from a new angle. Large RNNs
are interpreted as \"reservoirs\" of complex, excitable dynamics.
Output units \"tap\" from this reservoir by linearly combining the
desired output signal from the rich variety of excited reservoir
signals. This idea leads to training algorithms where only the
network-to-output connection weights have to be trained. This can be
done with known, highly efficient linear regression algorithms. from
See:  http://www.faculty.iu-bremen.de/hjaeger/esn_research.html

C++ includes: esn.h ";

/*  algorithms are friends  */

/*  Algorithm interface  */

%feature("docstring")  ESN::init "throw ( AUExcept)
Initialization Algorithm for an Echo State Network See:  class
InitBase ";

%feature("docstring")  ESN::train "throw ( AUExcept)
Training Algorithm Interface See:  class TrainBase

Parameters:
-----------

in:  matrix of input values (inputs x timesteps)

out:  matrix of desired output values (outputs x timesteps) for
teacher forcing

washout:  washout time in samples, used to get rid of the transient
dynamics of the network starting state ";

%feature("docstring")  ESN::simulate "

Simulation Algorithm Interface See:  class SimBase

Parameters:
-----------

in:  matrix of input values (inputs x timesteps)

out:  matrix for output values (outputs x timesteps) ";

%feature("docstring")  ESN::resetState "

resets the internal state vector x of the reservoir to zero ";

/*  C-style Algorithm interface  */

%feature("docstring")  ESN::train "throw ( AUExcept)
C-style Training Algorithm Interface (data will be copied into a FLENS
matrix) See:  class TrainBase

Parameters:
-----------

inmtx:  input matrix in row major storage (usual C array) (inputs x
timesteps)

outmtx:  output matrix in row major storage (outputs x timesteps) for
teacher forcing

washout:  washout time in samples, used to get rid of the transient
dynamics of the network starting state

Todo check how we can do that without copying ";

%feature("docstring")  ESN::simulate "throw ( AUExcept)
C-style Simulation Algorithm Interface with some additional error
checking. (data will be copied into a FLENS matrix) See:  class
SimBase

Parameters:
-----------

inmtx:  input matrix in row major storage (usual C array) (inputs x
timesteps)

outmtx:  output matrix in row major storage (outputs x timesteps),

Data must be already allocated!

Todo make these checks here ? ";

%feature("docstring")  ESN::simulateStep "throw (
AUExcept) C-style Simulation Algorithm Interface, for single step
simulation See:  class SimBase Todo see if we can do this in python
without this additional method

Parameters:
-----------

inmtx:  input vector, size = inputs

outmtx:  output vector, size = outputs

Data must be already allocated!

Todo make these checks here ? ";

/*  Additional Interface for Bandpass Neurons  */

/* Todo rethink if this is consistent

*/

%feature("docstring")  ESN::setBPCutoff "throw (
AUExcept) Set lowpass/highpass cutoff frequencies for bandpass style
neurons. \" See:  class SimBP

Parameters:
-----------

f1:  vector with lowpass cutoff for all neurons (size = neurons)

f2:  vector with highpass cutoffs (size = neurons) ";

%feature("docstring")  ESN::setBPCutoff "throw (
AUExcept) Set lowpass/highpass cutoff frequencies for bandpass style
neurons \" (C-style Interface).

Parameters:
-----------

f1:  vector with lowpass cutoff for all neurons (size = neurons)

f2:  vector with highpass cutoffs (size = neurons) ";

/*  GET parameters  */

%feature("docstring")  ESN::post "

posts current parameters to stdoutTodo maybe return a outputstream (if
stdout is not useful) or just use the << operator ? ";

%feature("docstring")  ESN::getSize "

reservoir size (nr of neurons) ";

%feature("docstring")  ESN::getInputs "

nr of inputs to the reservoir ";

%feature("docstring")  ESN::getOutputs "

nr of outputs from the reservoir ";

%feature("docstring")  ESN::getNoise "

current noise level ";

%feature("docstring")  ESN::getInitParam "

returns an initialization parametern from the parameter map

Parameters:
-----------

key:  the requested parameter

the value of the parameter ";

%feature("docstring")  ESN::getInitAlgorithm "

initialization algorithm ";

%feature("docstring")  ESN::getTrainAlgorithm "

training algorithm ";

%feature("docstring")  ESN::getSimAlgorithm "

simulation algorithm ";

%feature("docstring")  ESN::getReservoirAct "

reservoir activation function ";

%feature("docstring")  ESN::getOutputAct "

output activation function ";

/*  GET internal data  */

%feature("docstring")  ESN::getWin "

input weight matrix ";

%feature("docstring")  ESN::getW "

reservoir weight matrix ";

%feature("docstring")  ESN::getWback "

feedback (output to reservoir) weight matrix ";

%feature("docstring")  ESN::getWout "

output weight matrix ";

%feature("docstring")  ESN::getX "

internal state vector x ";

/*  GET internal data C-style interface  */

%feature("docstring")  ESN::getWin "

get pointer to input weight matrix data and dimensions WARNING:  This
data is in fortran style column major storage ! ";

%feature("docstring")  ESN::getWback "

get pointer to feedback weight matrix data and dimensions WARNING:
This data is in fortran style column major storage ! ";

%feature("docstring")  ESN::getWout "

get pointer to output weight matrix data and dimensions WARNING:  This
data is in fortran style column major storage ! ";

%feature("docstring")  ESN::getX "

get pointer to internal state vector x data and length ";

%feature("docstring")  ESN::getW "throw ( AUExcept)
Copies data of the sparse reservoir weight matrix into a dense C-style
matrix. Memory of the C array must be allocated before!

Parameters:
-----------

wmtx:  pointer to matrix of size (neurons_ x neurons_)

Todo check if this can be avoided ";

/*  SET methods  */

%feature("docstring")  ESN::setInitAlgorithm "throw (
AUExcept) set initialization algorithm ";

%feature("docstring")  ESN::setTrainAlgorithm "throw (
AUExcept) set training algorithm ";

%feature("docstring")  ESN::setSimAlgorithm "throw (
AUExcept) set simulation algorithm ";

%feature("docstring")  ESN::setSize "throw ( AUExcept)
set reservoir size (nr of neurons) ";

%feature("docstring")  ESN::setInputs "throw (
AUExcept) set nr of inputs to the reservoir ";

%feature("docstring")  ESN::setOutputs "throw (
AUExcept) set nr of outputs from the reservoir ";

%feature("docstring")  ESN::setNoise "throw ( AUExcept)
set noise level for training/simulation algorithm

Parameters:
-----------

noise:  with uniform distribution within [-noise|+noise] ";

%feature("docstring")  ESN::setInitParam "

set initialization parameter ";

%feature("docstring")  ESN::setReservoirAct "throw (
AUExcept) set reservoir activation function ";

%feature("docstring")  ESN::setOutputAct "throw (
AUExcept) set output activation function ";

%feature("docstring")  ESN::setWout "throw ( AUExcept)
set output weight matrix ";

%feature("docstring")  ESN::setWout "throw ( AUExcept)
set output weight matrix C-style interface (data will be copied into a
FLENS matrix)

Parameters:
-----------

wout:  pointer to wout matrix in row major storage (usual C array)

rows:  number of rows

cols:  number of columns

Todo check how we can do that without copying ";

%feature("docstring")  ESN::ESN "

Constructor. ";

%feature("docstring")  ESN::ESN "

Copy Constructor.

Todo check if maps operator= performs a deep copy !

Todo bei SIM_BP die restlichen buffer variablen auch kopieren ";

%feature("docstring")  ESN::~ESN "

Destructor. ";

