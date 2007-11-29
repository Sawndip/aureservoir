/***************************************************************************/
/*!
 *  \file   simulate.hpp
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

#include <assert.h>

namespace aureservoir
{

//! @name class BPFilter Implementation
//@{

template <typename T>
void BPFilter<T>::setBPCutoff(const typename DEVector<T>::Type &f1,
                              const typename DEVector<T>::Type &f2)
  throw(AUExcept)
{
  if( f1.length() != f2.length() )
    throw AUExcept("BPFilter: f1 must be same size as f2!");

  int size = f1.length();

  // allocate data
  ema1_.resizeOrClear(size);
  ema2_.resizeOrClear(size);
  f1_.resize(size);
  f2_.resize(size);
  scale_.resize(size);

  f1_ = f1;
  f2_ = f2;

  // calculate scaler values:
  // scale = 1 + f2/f1;
  for(int i=1; i<=size; ++i)
    scale_(i) = 1 + f2_(i)/f1_(i);
}

template <typename T>
void BPFilter<T>::calc(typename DEVector<T>::Type &x)
{
  // Bandpass Filtering: new activation = ema1(act) - ema2(ema1(act))
  // ema1 += f1 * (activation - ema1)
  // ema2  += f2 * (ema1 - ema2)
  // activation = (ema1 - ema2) * scale
  for(int i=1; i<=ema1_.length(); ++i)
  {
    ema1_(i) += f1_(i) * ( x(i) - ema1_(i) );
    ema2_(i) += f2_(i) * ( ema1_(i) - ema2_(i) );
    x(i) = (ema1_(i) - ema2_(i)) * scale_(i);
  }
}

//@}

} // end of namespace aureservoir
