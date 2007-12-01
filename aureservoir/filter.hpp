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
//! @name class BPFilter Implementation
//@{

template <typename T>
void IIRFilter<T>::setIIRCoeff(const typename DEMatrix<T>::Type &B,
                   const typename DEMatrix<T>::Type &A)
  throw(AUExcept)
{
  if( B.numRows() != A.numRows() )
    throw AUExcept("BPFilter: B and A must have same rows!");
  if( B.numCols() != A.numCols() )
    throw AUExcept("BPFilter: B and A must have same cols!");

  // check if a[0] is one !
  bool check = false;
  for(int i=1; i<=A.numRows(); ++i)
    check = (A(i,1) == 1);
  if( check )
    throw AUExcept("BPFilter: a[0] must be 1 for all components!");

  B_ = B;
  A_ = A;
  S_.resizeOrClear(A.numRows(), A.numCols()-1);
  y_.resizeOrClear(A.numRows());

  /// \todo calc scale matrix
}

template <typename T>
void IIRFilter<T>::calc(typename DEVector<T>::Type &x)
{
  assert( x.length() == S_.numRows() );

  int N = S_.numCols();
  int size = S_.numRows();

  for(int i=1; x<=size; ++i)
  {
    // calc new output
    y_(i) = B_(1,i) * x(i) + S_(1,i);

    // update internal storage
    for(int j=1; j<=(N-1); ++j)
      S_(j,i) = B_(j+1,i) * x(i) - A_(j+1,i) * y_(i) + S_(j+1,i);
    S_(N,i) = B_(N+1,i) * x(i) - A_(N+1,i) * y_(i);
  }

  /// \todo rescale output to keep spectral radius

  x = y_;
}

//@}

} // end of namespace aureservoir
