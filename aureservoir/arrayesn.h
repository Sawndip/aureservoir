/***************************************************************************/
/*!
 *  \file   arrayesn.h
 *
 *  \brief  an array of ESNs where their outputs are averaged
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   April 2008
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

#ifndef AURESERVOIR_ARRAY_ESN_H__
#define AURESERVOIR_ARRAY_ESN_H__

#include "esn.h"
#include <vector>
#include <algorithm>

namespace aureservoir
{

/*!
 * \class ArrayESN
 *
 * \brief an array of ESNs where their outputs are averaged
 *
 * This class uses an array of the same ESNs and all are trained
 * independently. In simulation the overal output is the average of all
 * the individual network outputs and if there are feedback connections
 * the overal output will be fed back to the individual networks.
 *
 * This is a standard trick to boost performance, as described in
 * \sa "Harnessing nonlinearity: predicting chaotic systems and saving energy
 *      in wireless telecommunication" by Jäger and Haas
 */
template <typename T = float>
class ArrayESN
{
 public:

  /*!
   * Constructor
   *
   * @param model a model-ESN which will be cloned to create all the other
   *              ESNs in the array
   * @param array_size  number of parallel ESNs in the array
   */
  ArrayESN(const ESN<T> &model, int array_size)
    throw(AUExcept)
  {
    if( array_size <= 0 )
      throw AUExcept("ArrayESN: array_size must be at least 1 !");
    array_size_ = array_size;

    // create array of ESNs
    for(int i=0; i<array_size_; ++i)
      esns_.push_back(model);
  }

  /// Destructor
  ~ArrayESN() {}

  /*!
   * Initializes all networks
   */
  void init()
    throw(AUExcept)
  {
    for(int i=0; i<array_size_; ++i)
      esns_[i].init();
  }

  /*!
   * Trains all networks independently
   *
   * @param in matrix of input values (inputs x timesteps)
   * @param out matrix of desired output values (outputs x timesteps)
   *            for teacher forcing
   * @param washout washout time in samples, used to get rid of the
   *                transient dynamics of the network starting state
   */
  void train(const typename ESN<T>::DEMatrix &in,
             const typename ESN<T>::DEMatrix &out, int washout)
    throw(AUExcept)
  {
    for(int i=0; i<array_size_; ++i)
      esns_[i].train(in,out,washout);
  }

  /*!
   * C-style Training Algorithm Interface
   * (data will be copied into a FLENS matrix)
   * Trains all networks independently
   *
   * @param inmtx input matrix in row major storage (usual C array)
   *              (inputs x timesteps)
   * @param outmtx output matrix in row major storage (outputs x timesteps)
   *               for teacher forcing
   * @param washout washout time in samples, used to get rid of the
   *                transient dynamics of the network starting state
   */
  void train(T *inmtx, int inrows, int incols,
             T *outmtx, int outrows, int outcols, int washout)
    throw(AUExcept)
  {
    typename ESN<T>::DEMatrix flin(inrows,incols);
    typename ESN<T>::DEMatrix flout(outrows,outcols);

    // copy data to FLENS matrix (column major storage)
    for(int i=0; i<inrows; ++i) {
    for(int j=0; j<incols; ++j) {
      flin(i+1,j+1) = inmtx[i*incols+j];
    } }
    for(int i=0; i<outrows; ++i) {
    for(int j=0; j<outcols; ++j) {
      flout(i+1,j+1) = outmtx[i*outcols+j];
    } }

    train(flin, flout, washout);
  }

  /*!
   * Simulation: the overal output is the average of all
   * individual network outputs.
   * For ESNs with feedback connections the overal output will be fed back to
   * the individual networks.
   *
   * @param in matrix of input values (inputs x timesteps)
   * @param out matrix for output values (outputs x timesteps)
   */
  void simulate(const typename ESN<T>::DEMatrix &in, typename ESN<T>::DEMatrix &out)
  {
    if( out.numCols() != in.numCols() )
      throw AUExcept("ArrayESN: output and input must have same nr of columns!");
    if( in.numRows() != esns_[0].getInputs() )
      throw AUExcept("ArrayESN: wrong input row size!");
    if( out.numRows() != esns_[0].getOutputs() )
      throw AUExcept("ArrayESN: wrong output row size!");

    int steps = in.numCols();
    typename ESN<T>::DEMatrix sim_in(esns_[0].getInputs() ,1),
                              sim_out(esns_[0].getOutputs() ,1);

    // do single step simulation with all networks
    for(int n=1; n<=steps; ++n)
    {
      sim_in(_,1) = in(_,n);
      // clear output vektor
      for(int i=1; i<=esns_[0].getOutputs(); ++i)
        out(i,n) = 0.;

      // simulate all networks
      for(int i=0; i<array_size_; ++i)
      {
        esns_[i].simulate(sim_in, sim_out);
        out(_,n) = out(_,n) + sim_out(_,1);
      }

      // calculate average output
      out(_,n) = out(_,n) / array_size_;

      // feed back average output to all networks
      for(int i=0; i<array_size_; ++i)
        esns_[i].setLastOutput( out(_,n) );
    }
  }

  /*!
   * C-style Simulation Algorithm Interface
   * (data will be copied into a FLENS matrix)
   * Simulation: the overal output is the average of all
   * individual network outputs.
   * For ESNs with feedback connections the overal output will be fed back to
   * the individual networks.
   *
   * @param inmtx input matrix in row major storage (usual C array)
   *              (inputs x timesteps)
   * @param outmtx output matrix in row major storage (outputs x timesteps),
   *               \attention Data must be already allocated!
   */
  void simulate(T *inmtx, int inrows, int incols,
                T *outmtx, int outrows, int outcols)
    throw(AUExcept)
  {
    typename ESN<T>::DEMatrix flin(inrows,incols);
    typename ESN<T>::DEMatrix flout(outrows,outcols);

    // copy data to FLENS matrix
    for(int i=0; i<inrows; ++i) {
    for(int j=0; j<incols; ++j) {
      flin(i+1,j+1) = inmtx[i*incols+j];
    } }

    simulate(flin, flout);

    // copy data to output
    for(int i=0; i<outrows; ++i) {
    for(int j=0; j<outcols; ++j) {
      outmtx[i*outcols+j] = flout(i+1,j+1);
    } }
  }

  /*!
   * Teacher-Forcing all networks independently
   *
   * @param in matrix of input values (inputs x timesteps)
   * @param out matrix for output values (outputs x timesteps)
   */
  void teacherForce(const typename ESN<T>::DEMatrix &in, typename ESN<T>::DEMatrix &out)
    throw(AUExcept)
  {
    for(int i=0; i<array_size_; ++i)
      esns_[i].teacherForce(in,out);
  }

 /*!
   * Teacher Forcing a input and target signal without learning output weights.
   * This is useful for ESNs in generator mode to initialize the internal state.
   * Teacher-Forcing all networks independently.
   *
   * @param inmtx input matrix in row major storage (usual C array)
   *              (inputs x timesteps)
   * @param outmtx output matrix in row major storage (outputs x timesteps)
   *               for teacher forcing
   */
  void teacherForce(T *inmtx, int inrows, int incols,
                    T *outmtx, int outrows, int outcols)
    throw(AUExcept)
  {
    typename ESN<T>::DEMatrix flin(inrows,incols);
    typename ESN<T>::DEMatrix flout(outrows,outcols);

    // copy data to FLENS matrix (column major storage)
    for(int i=0; i<inrows; ++i) {
    for(int j=0; j<incols; ++j) {
      flin(i+1,j+1) = inmtx[i*incols+j];
    } }
    for(int i=0; i<outrows; ++i) {
    for(int j=0; j<outcols; ++j) {
      flout(i+1,j+1) = outmtx[i*outcols+j];
    } }

    teacherForce(flin, flout);
  }


  /// returns the number of ESNs in the ArrayESN
  int getArraySize() const
  { return array_size_; }

  /// @param index returns ESN with that index, index starts with 0
  ESN<T> getNetwork(int index) const
    throw(AUExcept)
  {
    if( index < 0 || index >= array_size_ )
      throw AUExcept("ArrayESN: wrong index !");

    return esns_[index];
  }

  /// prints out all output weight mean values
  void printWoutMean()
  {
    std::cout << "Wout Mean: ";
    typename ESN<T>::DEMatrix wout;

    for(int i=0; i<array_size_; ++i)
    {
      T meanval = 0;
      wout = esns_[i].getWout();
      int le = wout.numRows()*wout.numCols();

      // calculate mean value
      for(int j=0; j<le; ++j)
        meanval += wout.data()[j];

      std::cout << " " << meanval/le;
    }

    std::cout << "\n";
  }

  /// prints out all output weight max values
  void printWoutMax()
  {
    std::cout << "Wout Max: ";
    typename ESN<T>::DEMatrix wout;

    for(int i=0; i<array_size_; ++i)
    {
      wout = esns_[i].getWout();
      int le = wout.numRows()*wout.numCols();

      // calculate absolute value
      for(int j=0; j<le; ++j)
        wout.data()[j] = std::abs( wout.data()[j] );

      T maxval = *std::max_element( wout.data(), wout.data()+le );
      std::cout << " " << maxval;
    }

    std::cout << "\n";
  }

  /// set noise level for all networks
  /// @param noise with uniform distribution within [-noise|+noise]
  void setNoise(double noise)
    throw(AUExcept)
  {
    for(int i=0; i<array_size_; ++i)
      esns_[i].setNoise(noise);
  }

 protected:

  /// nr of parallel ESNs in the array
  int array_size_;

  /// the vector with ESNs
  std::vector< ESN<T> > esns_;
};

} // end of namespace aureservoir

#endif // AURESERVOIR_ARRAY_ESN_H__
