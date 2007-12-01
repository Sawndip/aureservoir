/***************************************************************************/
/*!
 *  \file   filter.h
 *
 *  \brief  implementation of some general filters used in the reservoir
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Dec 2007
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

#ifndef AURESERVOIR_FILTER_H__
#define AURESERVOIR_FILTER_H__

#include "utilities.h"

namespace aureservoir
{

/*!
 * \class BPFilter
 *
 * \brief simple bandpass filter based on an exponential moving average
 *
 * This is an implementation of a band pass filter with two exponential
 * moving average filters:
 * ema1  = ema1 + f1 * (input - ema1);
 * ema2  = ema2 + f2 * (ema1 - ema2);
 * output = ema1 - ema2;
 *
 * ema = exponential moving average filter, corresponds to a LOP
 * f1 = lowpass cutoff frequency
 * f2 = highpass cutoff frequency
 * 0 \< f2 \< f1 \< 1
 * f1=1 -> no LOP filtering, f2=0 -> no highpass filtering
 * f1=1 and f2=0 -> no filtering
 * f2=0 -> only lowpass filtering
 *
 * Finally the output get rescaled with
 * output = output * (1 + f2/f1)
 * (to keep the spectral radius in ESNs)
 */
template <typename T>
class BPFilter
{
 public:

  /// Constructor
  BPFilter() {}

  /// Destructor
  virtual ~BPFilter() {}

  /// set LOP and HIP cutoff frequencies
  void setBPCutoff(const typename DEVector<T>::Type &f1,
                   const typename DEVector<T>::Type &f2)
                   throw(AUExcept);

  /// calculates one filter step on each element of x and writes
  /// the result back to x
  void calc(typename DEVector<T>::Type &x);

 protected:

  /// last output of ema1 (exponential moving average filter 1)
  typename DEVector<T>::Type ema1_;
  /// last output of ema2 (exponential moving average filter 2)
  typename DEVector<T>::Type ema2_;
  /// low pass cutoff frequencies
  typename DEVector<T>::Type f1_;
  /// high pass cutoff frequencies
  typename DEVector<T>::Type f2_;
  /// scale factor for output, to not shrink spectral radius
  typename DEVector<T>::Type scale_;
};

/*!
 * \class IIRFilter
 *
 * \brief general IIR filter implemented in transposed direct form 2
 *
 * General IIR Filter Implementation in Transposed Direct Form 2,
 * which has good numeric stability properties.
 * \sa http://ccrma.stanford.edu/~jos/filters/Transposed_Direct_Forms.html
 *
 * The filter calculates the following difference equation:
 * a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
 *                       - a[1]*y[n-1] - ... - a[na]*y[n-na]
 */
template <typename T>
class IIRFilter
{
 public:

  /// Constructor
  IIRFilter() {}

  /// Destructor
  virtual ~IIRFilter() {}

  /**
   * sets the filter coefficients
   * @param B matrix with numerator coefficient vectors (m x nb)
   *          m  ... nr of parallel filters (neurons)
   *          nb ... nr of filter coefficients
   * @param A matrix with denominator coefficient vectors (m x na)
   *          \note a[0] must be 1 !
   *          m  ... nr of parallel filters (neurons)
   *          na ... nr of filter coefficients
   */
  void setIIRCoeff(const typename DEMatrix<T>::Type &B,
                   const typename DEMatrix<T>::Type &A)
                   throw(AUExcept);

  /// calculates one filter step on each element of x and writes
  /// the result back to x
  void calc(typename DEVector<T>::Type &x);

 protected:

  /// filter numerator coefficients
  typename DEMatrix<T>::Type B_;
  /// filter denominator coefficients
  typename DEMatrix<T>::Type A_;
  /// internal data for calculation
  typename DEMatrix<T>::Type S_;
  /// temporal object to store output
  typename DEVector<T>::Type y_;

};

} // end of namespace aureservoir

#include <aureservoir/filter.hpp>

#endif // AURESERVOIR_FILTER_H__
