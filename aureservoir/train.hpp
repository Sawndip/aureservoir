/***************************************************************************/
/*!
 *  \file   train.hpp
 *
 *  \brief  training algorithms for Echo State Networks
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

namespace aureservoir
{

//! @name class TrainBase Implementation
//@{

template <typename T>
void TrainBase<T>::checkParams(const typename ESN<T>::DEMatrix &in,
                               const typename ESN<T>::DEMatrix &out,
                               int washout)
  throw(AUExcept)
{
  // check data size
  if( in.numCols() != out.numCols() )
    throw AUExcept("TrainBase::train: input and output must be same column size!");
  if( in.numRows() != esn_->inputs_ )
    throw AUExcept("TrainBase::train: wrong input row size!");
  if( out.numRows() != esn_->outputs_ )
    throw AUExcept("TrainBase::train: wrong output row size!");

  // check if we have enough training data
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
  {
    if( (in.numCols()-washout) < esn_->neurons_+esn_->inputs_ )
    throw AUExcept("TrainBase::train: too few training data!");
  }
  else
  {
    if( (in.numCols()-washout) < 2*(esn_->neurons_+esn_->inputs_) )
    throw AUExcept("TrainBase::train: too few training data!");
  }

  // check if we have an Wout matrix
  if( esn_->Wout_.numRows() == 0 || esn_->Wout_.numCols() == 0 )
    throw AUExcept("TrainBase::train: you need to have a Wout matrix, so init the net or set Wout manually!");

  /// \todo check also for the right algorithm combination
  ///       -> or better do that in init()

  // reallocate data buffer for simulation algorithm
  esn_->sim_->reallocate();
}

template <typename T>
void TrainBase<T>::collectStates(const typename ESN<T>::DEMatrix &in,
                                 const typename ESN<T>::DEMatrix &out,
                                 int washout)
{
  int steps = in.numCols();

  // collects output of all timesteps in O
  O.resize(steps-washout, esn_->outputs_);

  // collects reservoir activations and inputs of all timesteps in M
  // (for squared algorithm we need a bigger matrix)
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
    M.resize(steps-washout, esn_->neurons_+esn_->inputs_);
  else
    M.resize(steps-washout, 2*(esn_->neurons_+esn_->inputs_));


  typename ESN<T>::DEMatrix sim_in(esn_->inputs_ ,1),
                            sim_out(esn_->outputs_ ,1);
  for(int n=1; n<=steps; ++n)
  {
    sim_in(_,1) = in(_,n);
    esn_->simulate(sim_in, sim_out);

    // for teacherforcing with feedback in single step simulation
    // we need to set the correct last output
    esn_->sim_->last_out_(_,1) = out(_,n);

//     std::cout << esn_->x_ << std::endl;

    // store internal states, inputs and outputs after washout
    if( n > washout )
    {
      M(n-washout,_(1,esn_->neurons_)) = esn_->x_;
      M(n-washout,_(esn_->neurons_+1,esn_->neurons_+esn_->inputs_)) =
      sim_in(_,1);
    }
  }

  // collect desired outputs
  O = flens::transpose( out( _,_(washout+1,steps) ) );
}

template <typename T>
void TrainBase<T>::squareStates()
{
  // add additional squared states and inputs
  /// \todo vectorize that
  int Msize = esn_->neurons_+esn_->inputs_;
  int Mrows = M.numRows();
  for(int i=1; i<=Mrows; ++i) {
  for(int j=1; j<=Msize; ++j) {
    M(i,j+Msize) = pow( M(i,j), 2 );
  } }
}

//@}
//! @name class TrainPI Implementation
//@{

template <typename T>
void TrainPI<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);

  // 1. teacher forcing, collect states
  this->collectStates(in,out,washout);

  // add additional squared states when using SIM_SQUARE
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
    this->squareStates();


  // 2. offline weight computation

  // undo output activation function
  esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );

  // calc weights with pseudo inv: Wout_ = (M^-1) * O
  flens::lss( M, O );
  esn_->Wout_ = flens::transpose( O(_( 1, M.numCols() ),_) );

  this->clearData();
}

//@}
//! @name class TrainLS Implementation
//@{

template <typename T>
void TrainLS<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);

  // 1. teacher forcing, collect states
  this->collectStates(in,out,washout);

  // add additional squared states when using SIM_SQUARE
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
    this->squareStates();


  // 2. offline weight computation

  // undo output activation function
  esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );

  // calc weights with least square solver: Wout_ = (M^-1) * O
  flens::ls( flens::NoTrans, M, O );
  esn_->Wout_ = flens::transpose( O(_( 1, M.numCols() ),_) );

  this->clearData();
}

//@}
//! @name class TrainRidgeReg Implementation
//@{

template <typename T>
void TrainRidgeReg<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);

  // 1. teacher forcing, collect states
  this->collectStates(in,out,washout);

  // add additional squared states when using SIM_SQUARE
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
    this->squareStates();


  // 2. offline weight computation

  // undo output activation function
  esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );


  // calc weights with ridge regression (.T = transpose):
  // Wout = ( (M.T*M + alpha^2*I)^-1 *M.T * O )

  // get regularization factor and square it
  T alpha = pow(esn_->init_params_[TIKHONOV_FACTOR],2);

  // temporal objects
  typename ESN<T>::DEMatrix T1(esn_->neurons_+esn_->inputs_,
                               esn_->neurons_+esn_->inputs_);
  flens::DenseVector<flens::Array<int> > t2( M.numCols() );

  // M.T * M
  T1 = flens::transpose(M)*M;

  // ans + alpha^2*I
  for(int i=1; i<=T1.numRows(); ++i)
    T1(i,i) += alpha;

  // calc inverse: (ans)^-1
  flens::trf(T1, t2);
  flens::tri(T1, t2);

  // ans * M.T
  esn_->Wout_ = T1 * flens::transpose(M);

  // ans * O
  T1 = esn_->Wout_ * O;

  // result = ans.T
  esn_->Wout_ = flens::transpose(T1);


  this->clearData();
}

//@}
//! @name class TrainDSPI Implementation
//@{

template <typename T>
void TrainDSPI<T>::train(const typename ESN<T>::DEMatrix &in,
                       const typename ESN<T>::DEMatrix &out,
                       int washout)
  throw(AUExcept)
{
  this->checkParams(in,out,washout);

  // check for right simulation algorithm
  if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_FILTER_DS &&
      esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
    throw AUExcept("TrainDSPI::train: you need to use SIM_FILTER_DS or SIM_SQUARE for this training algorithm!");


  // 1. teacher forcing, collect states

  int steps = in.numCols();

  // collects reservoir activations and inputs of all timesteps in M
  M.resize(steps, esn_->neurons_+esn_->inputs_);

  // collect desired outputs
  O.resize(steps-washout, 1);

  typename ESN<T>::DEMatrix sim_in(esn_->inputs_ ,1),
                            sim_out(esn_->outputs_ ,1);
  for(int n=1; n<=steps; ++n)
  {
    sim_in(_,1) = in(_,n);
    esn_->simulate(sim_in, sim_out);

    // for teacherforcing with feedback in single step simulation
    // we need to set the correct last output
    esn_->sim_->last_out_(_,1) = out(_,n);

    // store internal states, inputs and outputs
    M(n,_(1,esn_->neurons_)) = esn_->x_;
    M(n,_(esn_->neurons_+1,esn_->neurons_+esn_->inputs_)) =
    sim_in(_,1);
  }


  // 2. delay calculation for delay&sum readout

  // get maxdelay or set it to 0 if not given
  int maxdelay;
  if( esn_->init_params_.find(DS_MAXDELAY) == esn_->init_params_.end() )
    maxdelay = 0;
  else
    maxdelay = (int) esn_->init_params_[DS_MAXDELAY];

  // maximum of maxdelay is the number of steps
  if( esn_->init_params_.find(DS_FORCE_MAXDELAY) != 
      esn_->init_params_.end() )
    maxdelay = maxdelay+1;
  else
    maxdelay = (steps-washout < maxdelay+1) ?
                steps-washout : maxdelay+1;

  // see if we use GCC or simple crosscorr, standard is GCC
  int filter;
  if( esn_->init_params_.find(DS_USE_CROSSCORR) == esn_->init_params_.end() )
    filter = 1;
  else
    filter = 0;

  // get the nr of iterations for EM algorithm
  int emiters = 0; /// \todo change this ?
  if( esn_->init_params_.find(DS_EM_ITERATIONS) != esn_->init_params_.end() )
  {
    emiters = (int) esn_->init_params_[DS_EM_ITERATIONS];
    emiters = (emiters < 0) ? 0 : emiters;
  }


  // 3. finally perform the delay learning algorithm
  if( emiters > 0 )
    delayLearningEM(in,out,washout,steps,maxdelay, filter, emiters);
  else
    delayLearningSimple(in,out,washout,steps,maxdelay, filter, emiters);
  
  this->clearData();
}


template <typename T>
void TrainDSPI<T>::delayLearningSimple(const typename ESN<T>::DEMatrix &in,
                                       const typename ESN<T>::DEMatrix &out,
				       int washout, int steps, int maxdelay, int filter,
				       int emiters)
  throw(AUExcept)
{
  int delay = 0;
  int fftsize = (int) pow( 2, ceil(log(steps)/log(2)) ); // next power of 2
  typename CDEVector<T>::Type X,Y;
  typename DEVector<T>::Type x,y,rest;
  typename DEMatrix<T>::Type T1(1,esn_->neurons_+esn_->inputs_);
  typename DEMatrix<T>::Type Mtmp(M.numRows(),M.numCols()); /// \todo memory !!!

  for(int i=1; i<=esn_->outputs_; ++i)
  {
    // calc FFT of target vector
    y = out(i,_);
    esn_->outputInvAct_( y.data(), y.length() );
    rfft( y, Y, fftsize );

    // calc delays to reservoir neurons and inputs
    for(int j=1; j<=esn_->neurons_+esn_->inputs_; ++j)
    {
      // calc FFT of neuron/input vector
      x = M(_,j);
      rfft( x, X, fftsize );

      // calc delay with GCC
      delay = CalcDelay<T>::gcc(X,Y,maxdelay,filter);

      if( delay != 0 )
      {
        // shift signal the right amount
        rest = M( _(M.numRows()-delay+1,M.numRows()), j );
        Mtmp( _(1,delay), j ) = 0.;
        Mtmp( _(delay+1,M.numRows()), j ) = M( _(1,M.numRows()-delay), j );

        // init delay lines with the rest of the buffer
        esn_->sim_->initDelayLine((i-1)*(esn_->neurons_+esn_->inputs_)+j-1, rest);
      }
      else
        Mtmp(_,j) = M(_,j);
    }


    // offline weight computation for each output extra

    // collect desired outputs
    O(_,1) = out( i ,_(washout+1,steps) );

    // undo output activation function
    esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );

    // square and double state if we have additional squared state updates
    if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
    {
      M = Mtmp( _(washout+1,steps), _);
    }
    else
    {
      M.resize( steps-washout, Mtmp.numCols()*2 );
      M( _, _(1,Mtmp.numCols()) ) = Mtmp( _(washout+1,steps), _);
      this->squareStates();
    }

    // calc weights with pseudo inv: Wout_ = (M^-1) * O
    flens::lss( M, O );
    T1 = flens::transpose( O(_( 1, M.numCols() ),_) );
    esn_->Wout_(i,_) = T1(1,_);


    // 4. restore simulation matrix M
    if( i < esn_->outputs_ )
    {
      M.resize( Mtmp.numRows(), Mtmp.numCols() );

      // undo the delays and store it again into M
      for(int j=1; j<=esn_->neurons_+esn_->inputs_; ++j)
      {
        rest = esn_->sim_->getDelayBuffer(i-1,j-1);
        delay = rest.length();

        if( delay != 0 )
        {
          M( _(1,steps-delay), j ) = Mtmp( _(delay+1,steps), j );
          M( _(steps-delay+1,steps), j ) = rest;
        }
        else
          M(_,j) = Mtmp(_,j);
      }
    }
  }
}


template <typename T>
void TrainDSPI<T>::delayLearningEM(const typename ESN<T>::DEMatrix &in,
                                   const typename ESN<T>::DEMatrix &out,
				   int washout, int steps, int maxdelay, int filter,
				   int emiters)
  throw(AUExcept)
{
  // check if maxdelay is bigger than the washout
  if( maxdelay > washout+1 )
    throw AUExcept("TrainDSPI::train: max delay must be <= as the washout (or use the simple delay calculation algorithm without EM) !");
  
  int fftsize = (int) pow( 2, ceil(log(steps-washout+1)/log(2)) ); // next power of 2
  typename DEMatrix<T>::Type Mtmp(steps-washout,M.numCols());
  int L = Mtmp.numCols();
  typename DEVector<T>::Type t(steps-washout),targ(steps-washout);
  typename DEVector<T>::Type r(steps-washout);
  typename CDEVector<T>::Type rF,tF;
  typename DEVector<T>::Type rest;
  typename DEMatrix<T>::Type t2(steps-washout,1);
  typename DEMatrix<T>::Type r2(steps-washout,1);
  typename DEVector<T>::Type w(L), wold(L);

  int em_version = 1;
  if( esn_->init_params_.find(EM_VERSION) != esn_->init_params_.end() )
  {
    em_version = (int) esn_->init_params_[EM_VERSION];
    if ( em_version<1 || em_version>3 ) em_version = 1;
    std::cout << "EM_VERSION: " << em_version << "\n";
  }


  // iterate over all outputs

  for(int n=1; n<=esn_->outputs_; ++n)
  {
    // collect desired outputs
    O(_,1) = out( n ,_(washout+1,steps) );
    // undo output activation function
    esn_->outputInvAct_( O.data(), O.numRows()*O.numCols() );
    
    // restore size of t2
    t2.resize(steps-washout,1);

    int delays[L];
    int delold[L];

    // set initial weights and delays to 0
    std::fill_n( w.data(), w.length(), 0. );
    std::fill_n( delays, L, 0 );

    bool converged = false;
    int itercount = 0;

    // delay neuron signals without delay (just copy)
    for(int i=0; i<Mtmp.numCols(); ++i)
      Mtmp(_, i+1) = M( _(washout+1,M.numRows()), i+1 );

    // init target vector
    targ = O(_,1);

    bool first_time = true;

    while( !converged )
    {
      // store previous delays and weights
      wold = w;
      for(int i=0; i<esn_->neurons_+esn_->inputs_; ++i)
        delold[i] = delays[i];

      for(int i=0; i<L; ++i)
      {
        //////////////////
        // E-step

        // delay neuronsignal from last step with new delay and remove it from target
        int ii = L-1;
        if( i>0 )
          ii = (i-1) % L; // to not get negative numbers out of %
        Mtmp(_, ii+1) = M( _(washout+1-delays[ii],M.numRows()-delays[ii]), ii+1 );
        targ = targ-Mtmp(_,ii+1)*w(ii+1);

        // remove contribution from all other neuron signals from target output O
        // (recursive implementation)
        if( em_version == 1 )
        {
          t2(_,1) = Mtmp(_,i+1)*w(i+1);
          t = targ + t2(_,1);
          targ = targ + t2(_,1);
        }
        if( em_version == 2 )
        {
          t2(_,1) = Mtmp(_,i+1)*w(i+1);
          t = targ/L + t2(_,1);
          targ = targ + t2(_,1);
        }
        if( em_version == 3 )
        {
          T beta = 0.;
          if( first_time )
          {
            beta = 1.;
            first_time = false;
          }
          else
          {
            // beta = 1 - ( abs(w(i+1)) / abs(w).sum() )
            T sum = 0.;
            for(int n=1; n<=L; ++n)
              sum += std::abs( w(n) );
            beta = 1 - ( std::abs( w(i+1) ) / sum );
          }
          t2(_,1) = Mtmp(_,i+1)*w(i+1);
          t = targ*beta + t2(_,1);
          targ = targ + t2(_,1);
        }


        //////////////////
        // M-step

        // current neuron signal
        r = M( _(washout+1,M.numRows()), i+1 );

        // calc FFT of target and neuron signal
        rfft( r, rF, fftsize );
        rfft( t, tF, fftsize );
        // estimate time delay between target signal x
        delays[i] = CalcDelay<T>::gcc(rF,tF,maxdelay,filter);

        // delay current neuron signal with new delay
        r = M( _(washout+1-delays[i],M.numRows()-delays[i]), i+1 );

        r2(_,1) = r;
        t2(_,1) = t;
        flens::ls( flens::NoTrans, r2, t2 );
//         flens::lss( r2, t2 );  // pinv
        w(i+1) = t2(1,1);
      }
      itercount++;

      // TODO: make a variable to switch on/off these diagnostics

      // calc delay difference:
      int deldiff = 0;
      for(int i=0; i<esn_->neurons_+esn_->inputs_; ++i)
        deldiff += abs( delold[i] - delays[i] );

      // calc weight difference:
      T wdiff = 0;
      for(int i=0; i<esn_->neurons_+esn_->inputs_; ++i)
        wdiff += std::abs( wold(i+1) - w(i+1) );

      // calc average weight
      T wmean = 0;
      for(int i=0; i<w.length(); ++i)
        wmean += w(i+1);
      wmean /= w.length();

      std::cout << "\titeration " << itercount << " - |d-dold| diff = "
                << deldiff << " - |w-wold| diff = " << wdiff
                << " - average weight = " << wmean << "\n";

      if( itercount >= emiters )
        converged = true;
    }

    // init delay lines with calculated delays
    for(int i=0; i<L; ++i)
    {
      if( delays[i] != 0 )
      {
        // shift reservoir signals the right amount
        Mtmp(_, i+1) = M( _(washout+1-delays[i],M.numRows()-delays[i]), i+1 );

        // init delay lines with the rest of the buffer
        rest = M( _(M.numRows()-delays[i]+1,M.numRows()), i+1 );
        esn_->sim_->initDelayLine((n-1)*(esn_->neurons_+esn_->inputs_)+i, rest);
      }
      else
        Mtmp(_, i+1) = M( _(washout+1,M.numRows()), i+1 );
    }

    // check if we should take the weight from EM algorithm
    if( esn_->init_params_.find(DS_WEIGHTS_EM) != esn_->init_params_.end() )
    {
      /// \todo squared state update !
      if( esn_->net_info_[ESN<T>::SIMULATE_ALG] == SIM_SQUARE )
        throw AUExcept("TrainDSPI::train: SQUARE not yet implemented for DS_WEIGHTS_EM!");

      std::cout << "\tusing weights from EM algorithm !\n";
      esn_->Wout_(n,_) = w;
      this->clearData();
      return;
    }

    // otherwise calculate output weights with pseudo inverse

    if( esn_->net_info_[ESN<T>::SIMULATE_ALG] != SIM_SQUARE )
    {
       // calc weights with pseudo inv: Wout_ = (M^-1) * O
       flens::lss( Mtmp, O );
       t2 = flens::transpose( O(_( 1, M.numCols() ),_) );
       esn_->Wout_(n,_) = t2(1,_);
    }
    else
    {
      // square and double state if we have additional squared state updates
      typename DEMatrix<T>::Type Mtmp2(M.numRows(),M.numCols());
      Mtmp2 = M; /// \todo find a better way to reconstruct this matrix ...
      
      M.resize( steps-washout, Mtmp.numCols()*2 );
      M( _, _(1,Mtmp.numCols()) ) = Mtmp;
      this->squareStates();

      // calc weights with pseudo inv: Wout_ = (M^-1) * O
      flens::lss( M, O );
      t2 = flens::transpose( O(_( 1, M.numCols() ),_) );
      esn_->Wout_(n,_) = t2(1,_);
      
      // resconstruct old M
      M = Mtmp2;
    }
  }
}

//@}

} // end of namespace aureservoir
