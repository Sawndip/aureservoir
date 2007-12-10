/***************************************************************************/
/*!
 *  \file   simulate.h
 *
 *  \brief  simulation algorithms for Echo State Networks
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Sept 2007
 *
 *   ::::_aureservoir_::::
 *   C++ library for analog reservoir computing neural networks
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU Lesser General Public
 *   License as published by the Free Software Foundation; either
 *   version 2.1 of the License, or (at your option) any later version.
 *
 ***************************************************************************/

#ifndef AURESERVOIR_SIMULATE_H__
#define AURESERVOIR_SIMULATE_H__

#include "utilities.h"
#include "filter.h"
#include <vector>

namespace aureservoir
{

/*!
 * \enum SimAlgorithm
 *
 * all possible simulation algorithms
 * \sa class SimStd
 */
enum SimAlgorithm
{
  SIM_STD,    //!< standard simulation \sa class SimStd
  SIM_SQUARE, //!< additional squared state updates \sa class SimSquare
  SIM_LI,     //!< simulation with leaky integrator neurons \sa class SimLI
  SIM_BP,     //!< simulation with bandpass neurons \sa class SimBP
  SIM_FILTER, //!< simulation with IIR-Filter neurons \sa class SimFilter
  SIM_FILTER2 //!< IIR-Filter before nonlinearity \sa class SimFilter2
};

template <typename T> class ESN;

/*!
 * \class SimBase
 *
 * \brief abstract base class for simulation algorithms
 *
 * This class is an abstract base class for all different kinds of
 * simulation algorithms.
 * The idea behind this system is that the algorithms can be exchanged
 * at runtime (strategy pattern).
 * \note I did quite some research and benchmarks, if a strategy design
 *       pattern should be used here (due to the virtual function overhead).
 *       In the end the virtual function call really did not matter,
 *       because there is quite much computation inside these methods.
 *
 * Simply derive from this class if you want to add a new algorithm.
 */
template <typename T>
class SimBase
{
 public:

  /// Constructor
  SimBase(ESN<T> *esn);

  /// Destructor
  virtual ~SimBase() {}

  /// Clones the simulation algorithm with all its internal data
  /// (virtual constructor idiom)
  virtual SimBase<T> *clone(ESN<T> *esn) const = 0;

  /*!
   * simulation algorithm
   *
   * @param in matrix of input values (inputs x timesteps)
   * @param out matrix for output values (outputs x timesteps)
   */
  virtual void simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out) = 0;

  /// reallocates data buffers
  virtual void reallocate();

  //! @name additional interface for bandpass and filter neurons
  //@{
  virtual void setBPCutoffConst(T f1, T f2) throw(AUExcept);
  virtual void setBPCutoff(const typename ESN<T>::DEVector &f1,
                           const typename ESN<T>::DEVector &f2)
                           throw(AUExcept);
  virtual void setIIRCoeff(const typename DEMatrix<T>::Type &B,
                           const typename DEMatrix<T>::Type &A)
                           throw(AUExcept);
  //@}

  /// output from last simulation
  typename ESN<T>::DEMatrix last_out_;

  /// temporary object needed for algorithm calculation
  typename ESN<T>::DEVector t_;

 protected:

  /// reference to the data of the network
  ESN<T> *esn_;
};

/*!
 * \class SimStd
 *
 * \brief standard simulation algorithm as in Jaeger's initial paper
 *
 * simulates an ESN with on activation function in the reservoir
 * and one activation function for the readout neurons, as described in
 * Jaeger's "Tutorial on training recurrent neural networks"
 * \sa http://www.faculty.iu-bremen.de/hjaeger/pubs/ESNTutorial.pdf
 */
template <typename T>
class SimStd : public SimBase<T>
{
  using SimBase<T>::esn_;
  using SimBase<T>::last_out_;
  using SimBase<T>::t_;

 public:
  SimStd(ESN<T> *esn) : SimBase<T>(esn) {}
  virtual ~SimStd() {}

  /// virtual constructor idiom
  virtual SimStd<T> *clone(ESN<T> *esn) const
  {
    SimStd<T> *new_obj = new SimStd<T>(esn);
    new_obj->t_ = t_; new_obj->last_out_ = last_out_;
    return new_obj;
  }

  /// implementation of the algorithm
  /// \sa class SimBase::simulate
  virtual void simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out);
};

/*!
 * \class SimSquare
 *
 * \brief algorithm with additional squared state updates
 *
 * Same as SimStd but with additional squared state updates, which
 * has the sense to get more nonlinearities in the reservoir without
 * a need of a very big reservoir size.
 * Describtion in following paper:
 * \sa http://www.faculty.iu-bremen.de/hjaeger/pubs/esn_NIPS02.pdf
 */
template <typename T>
class SimSquare : public SimBase<T>
{
  using SimBase<T>::esn_;
  using SimBase<T>::last_out_;
  using SimBase<T>::t_;

 public:
  SimSquare(ESN<T> *esn) : SimBase<T>(esn) {}
  virtual ~SimSquare() {}

  /// virtual constructor idiom
  virtual SimSquare<T> *clone(ESN<T> *esn) const
  {
    SimSquare<T> *new_obj = new SimSquare<T>(esn);
    new_obj->t_ = t_; new_obj->last_out_ = last_out_;
    return new_obj;
  }

  /// implementation of the algorithm
  /// \sa class SimBase::simulate
  virtual void simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out);
};

/*!
 * \class SimLI
 *
 * \brief algorithm with leaky integrator neurons
 *
 * ESN with leaky integrator reservoir units, which are usefull for
 * learning slow dynamical systems (amongst others).
 * \attention For stability reasons the spectral radius should be
 *            not bigger than the leaking rate!
 *            If leaking rate = spectral radius the resulting system
 *            will have unit spectral radius.
 *
 * This implementation is done according to:
 * \sa Optimization and applications of echo state networks with leaky
 *     integrator neurons. Neural Networks, 20(3)
 */
template <typename T>
class SimLI : public SimBase<T>
{
  using SimBase<T>::esn_;
  using SimBase<T>::last_out_;
  using SimBase<T>::t_;

 public:
  SimLI(ESN<T> *esn) : SimBase<T>(esn) {}
  virtual ~SimLI() {}

  /// virtual constructor idiom
  virtual SimLI<T> *clone(ESN<T> *esn) const
  {
    SimLI<T> *new_obj = new SimLI<T>(esn);
    new_obj->t_ = t_; new_obj->last_out_ = last_out_;
    return new_obj;
  }

  /// implementation of the algorithm
  /// \sa class SimBase::simulate
  virtual void simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out);
};

/*!
 * \class SimBP
 *
 * \brief algorithm with bandpass style neurons
 *
 * It can be shown that leaky integrator neurons perform somehow a lowpass
 * filtering on the reservoir states.
 * This is an extension of this interpration, using a lowpass and highpass
 * filter to get a bandpass style neuron. One reservoir can have neurons
 * with different cutoff frequencies to get richer activations on
 * different timescales.
 * \sa class SimLI
 *
 * The new activations can be calculated like this:
 * ema1  = ema1 + f1 * (activation - ema1);
 * ema2  = ema2 + f2 * (ema1 - ema2);
 * new_activation = ema1 - ema2;
 *
 * ema = exponential moving average filter, corresponds to a LOP
 * f1 = lowpass cutoff frequency
 * f2 = highpass cutoff frequency
 * 0 \< f2 \< f1 \< 1
 * f1=1 -> no LOP filtering, f2=0 -> no highpass filtering
 * f1=1 and f2=0 -> standard ESN
 * f2=0 -> leaky integrator ESN
 *
 * Finally the output needs to be scale with
 * new_activation = new_activation * (1 + f2/f1)
 * to keep the current spectral radius.
 */
template <typename T>
class SimBP : public SimBase<T>
{
  using SimBase<T>::esn_;
  using SimBase<T>::last_out_;
  using SimBase<T>::t_;

 public:
  SimBP(ESN<T> *esn) : SimBase<T>(esn) {}
  virtual ~SimBP() {}

  /// virtual constructor idiom
  virtual SimBP<T> *clone(ESN<T> *esn) const
  {
    SimBP<T> *new_obj = new SimBP<T>(esn);
    new_obj->t_ = t_; new_obj->last_out_ = last_out_;
    new_obj->filter_ = filter_;
    return new_obj;
  }

  /// needed for bandpass style neurons
  virtual void setBPCutoffConst(T f1, T f2) throw(AUExcept);

  /// set all LOP and HIP cutoff frequencies
  virtual void setBPCutoff(const typename ESN<T>::DEVector &f1,
                           const typename ESN<T>::DEVector &f2)
                           throw(AUExcept);

  /// implementation of the algorithm
  /// \sa class SimBase::simulate
  virtual void simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out);

  /// the filter object
  BPFilter<T> filter_;
};

/*!
 * \class SimFilter
 *
 * \brief algorithm with general IIR-Filter neurons
 *
 * This is an extension of the bandpass style neurons to general n-order
 * IIR-filter neurons.
 * \sa class SimBP
 *
 * The IIR Filter is implemented in Transposed Direct Form 2,
 * which has good numeric stability properties.
 * \sa http://ccrma.stanford.edu/~jos/filters/Transposed_Direct_Forms.html
 *
 * The filter calculates the following difference equation (same usage as
 * Matlab's filter object):
 * a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
 *                       - a[1]*y[n-1] - ... - a[na]*y[n-na]
 * where y is the new calculated activation after the filter and x is the
 * input to the filter.
 * The filter is always calculated _after_ the nonlinearity.
 * \sa class IIRFilter
 */
template <typename T>
class SimFilter : public SimBase<T>
{
  using SimBase<T>::esn_;
  using SimBase<T>::last_out_;
  using SimBase<T>::t_;

 public:
  SimFilter(ESN<T> *esn) : SimBase<T>(esn) {}
  virtual ~SimFilter() {}

  /// virtual constructor idiom
  virtual SimFilter<T> *clone(ESN<T> *esn) const
  {
    SimFilter<T> *new_obj = new SimFilter<T>(esn);
    new_obj->t_ = t_; new_obj->last_out_ = last_out_;
    new_obj->filter_ = filter_;
    return new_obj;
  }

  /**
   * sets the filter coefficients
   * @param B matrix with numerator coefficient vectors (m x nb)
   *          m  ... nr of parallel filters (neurons)
   *          nb ... nr of filter coefficients
   * @param A matrix with denominator coefficient vectors (m x na)
   *          m  ... nr of parallel filters (neurons)
   *          na ... nr of filter coefficients
   */
  virtual void setIIRCoeff(const typename DEMatrix<T>::Type &B,
                           const typename DEMatrix<T>::Type &A)
                           throw(AUExcept);

  /// implementation of the algorithm
  /// \sa class SimBase::simulate
  virtual void simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out);

  /// the filter object
  IIRFilter<T> filter_;
};

/*!
 * \class SimFilter2
 *
 * \brief algorithm with IIR-Filter before neuron nonlinearity
 *
 * This version is similar to SimFilter, but calculates the filtering
 * _before_ the nonlinearity of the reservoir neurons on input, feedback
 * and the network states.
 * \sa class SimFilter
 * \sa class IIRFilter
 */
template <typename T>
class SimFilter2 : public SimBase<T>
{
  using SimBase<T>::esn_;
  using SimBase<T>::last_out_;
  using SimBase<T>::t_;

 public:
  SimFilter2(ESN<T> *esn) : SimBase<T>(esn) {}
  virtual ~SimFilter2() {}

  /// virtual constructor idiom
  virtual SimFilter2<T> *clone(ESN<T> *esn) const
  {
    SimFilter2<T> *new_obj = new SimFilter2<T>(esn);
    new_obj->t_ = t_; new_obj->last_out_ = last_out_;
    new_obj->x_filter_ = x_filter_;
    new_obj->in_filter_ = in_filter_;
    new_obj->out_filter_ = out_filter_;
    return new_obj;
  }

  /**
   * sets the filter coefficients
   * @param B matrix with numerator coefficient vectors (m x nb)
   *          m  ... nr of parallel filters (neurons)
   *          nb ... nr of filter coefficients
   * @param A matrix with denominator coefficient vectors (m x na)
   *          m  ... nr of parallel filters (neurons)
   *          na ... nr of filter coefficients
   */
  virtual void setIIRCoeff(const typename DEMatrix<T>::Type &B,
                           const typename DEMatrix<T>::Type &A)
                           throw(AUExcept);

  /// implementation of the algorithm
  /// \sa class SimBase::simulate
  virtual void simulate(const typename ESN<T>::DEMatrix &in,
                        typename ESN<T>::DEMatrix &out);

  /// filter object for network states
  IIRFilter<T> x_filter_;

  /// filter object for inputs
  IIRFilter<T> in_filter_;

  /// filter object for outputs
  IIRFilter<T> out_filter_;

  /// temporary object needed for algorithm calculation
  typename ESN<T>::DEVector tin_, tfb_;
};

} // end of namespace aureservoir

#endif // AURESERVOIR_SIMULATE_H__
