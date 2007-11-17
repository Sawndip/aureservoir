/***************************************************************************/
/*!
 *  \file   activations.h
 *
 *  \brief  file for all kinds of different neuron activation functions
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

#ifndef AURESERVOIR_ACTIVATIONS_H__
#define AURESERVOIR_ACTIVATIONS_H__

#include "utilities.h"
#include <math.h>

namespace aureservoir
{

/*!
 * \enum ActivationFunction
 * all possible activation functions for reservoir and output neurons
 */
enum ActivationFunction
{
  ACT_LINEAR,      //!< linear activation function
  ACT_TANH         //!< tanh activation function
};

//! @name linear activation functions
//@{

/*!
 * linear activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_linear(T *data, int size)
{ }

/*!
 * inverse linear activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_invlinear(T *data, int size)
{ }

//@}
//! @name tanh activation functions
/// \todo see if there are faster tanh interpolations or SSE2 instructions
//@{

/*!
 * tanh activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_tanh(T *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] = tanh( data[i] );
}

/*!
 * inverse tanh activation function, performed on each element
 * @param data pointer to the data
 * @param size of the data
 */
template <typename T>
inline void act_invtanh(T *data, int size)
{
  for(int i=0; i<size; ++i)
    data[i] = atanh( data[i] );
}

//@}

} // end of namespace aureservoir

#endif // AURESERVOIR_ACTIVATIONS_H__
