/***************************************************************************/
/*!
 *  \file   esn_test.cpp
 *
 *  \brief  unit tests for the ESN class
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

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include "aureservoir/aureservoir.h"
using namespace aureservoir;

#include "test_utils.h"

template <typename T>
class ESNTest : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( ESNTest );
  CPPUNIT_TEST( exceptionTest );
  CPPUNIT_TEST( resetTest );
  CPPUNIT_TEST_SUITE_END();

  public:
    void setUp (void)
    { net_ = new ESN<T>; }

    void tearDown (void)
    { delete net_; }

  protected:
    void exceptionTest(void);
    void resetTest(void);

  private:
    ESN<T> *net_;
};

template <typename T>
void ESNTest<T>::exceptionTest(void)
{
  CPPUNIT_ASSERT_THROW( net_->setInputs(0), AUExcept );
  CPPUNIT_ASSERT_THROW( net_->setInputs(-4), AUExcept );

  CPPUNIT_ASSERT_THROW( net_->setReservoirSize(0), AUExcept );
  CPPUNIT_ASSERT_THROW( net_->setReservoirSize(-3), AUExcept );

  CPPUNIT_ASSERT_THROW( net_->setOutputs(0), AUExcept );
  CPPUNIT_ASSERT_THROW( net_->setOutputs(-4), AUExcept );

  typename ESN<T>::DEMatrix A(5,3);
  CPPUNIT_ASSERT_THROW( net_->setWout(A), AUExcept );
}

template <typename T>
void ESNTest<T>::resetTest(void)
{
  int n = 10;

  net_->setReservoirSize(n);
  net_->setInitParam(CONNECTIVITY, 0.9);
  net_->init();

  // set Wout
  typename ESN<T>::DEMatrix Wout(1, n+1);
  data_set_random( Wout.data(), (1+n) );
  net_->setWout( Wout );

  // simulate net
  typename ESN<T>::DEMatrix inp(1,10), outp(1,10);
  data_set_random( inp.data(), 10 );
  net_->simulate( inp, outp );

  net_->resetState();

  for(int i=1; i<=n; ++i)
    CPPUNIT_ASSERT_DOUBLES_EQUAL( net_->getX()(i), 0, 0.001 );
}

// register float and double version in test suite
CPPUNIT_TEST_SUITE_REGISTRATION ( ESNTest<float> );
CPPUNIT_TEST_SUITE_REGISTRATION ( ESNTest<double> );
