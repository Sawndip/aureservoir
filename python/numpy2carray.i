/***************************************************************************/
/*!
 *  \file   numpy2carray.i
 *
 *  \brief  SWIG numpy to C/C++ array conversions
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Oct 2007
 *
 *  Interface between numpy arrays and C/C++ style arrays.
 *  Build after looking at the umfpack.i interface from scipy SVN.
 *
 *  This should be seen as a demonstration and should be customized to
 *  your application (e.g. maybe more error checking etc.).
 *
 *  More detailed information:
 *  - swig and python: \sa http://www.swig.org/Doc1.3/Python.html
 *  - numpy c api: \sa http://numpy.scipy.org/numpydoc/numpy-13.html
 *  - swig and numpy: \sa http://www.scipy.org/Cookbook/SWIG_and_NumPy
 *
 ***************************************************************************/

%{
// include numpy C API
#define PY_ARRAY_UNIQUE_SYMBOL __example
#include "numpy/arrayobject.h"
%}

// init numpy
%init %{
  import_array();
%}

/*--------------------------- HELPER FUNCTIONS ----------------------------*/

%{
/*!
 * Appends @a what to @a where. On input, @a where need not to be
 * a tuple, but on return it always is.
 */
PyObject *helper_appendToTuple( PyObject *where, PyObject *what ) {
  PyObject *o2, *o3;

  if ((!where) || (where == Py_None)) {
    where = what;
  } else {
    if (!PyTuple_Check( where )) {
      o2 = where;
      where = PyTuple_New( 1 );
      PyTuple_SetItem( where, 0, o2 );
    }
    o3 = PyTuple_New( 1 );
    PyTuple_SetItem( o3, 0, what );
    o2 = where;
    where = PySequence_Concat( o2, o3 );
    Py_DECREF( o2 );
    Py_DECREF( o3 );
  }
  return where;
}

/*!
 * Helper to get a PyArrayObject from a PyObject.
 */
PyArrayObject *helper_getCArrayObject( PyObject *input, int type,
				       int minDim, int maxDim ) {
  PyArrayObject *obj;

  if (PyArray_Check( input )) {
    obj = (PyArrayObject *) input;
    if (!PyArray_ISCARRAY( obj )) {
      PyErr_SetString( PyExc_TypeError, "not a C array" );
      return NULL;
    }
    obj = (PyArrayObject *)
      PyArray_ContiguousFromAny( input, type, minDim, maxDim );
    if (!obj) return NULL;
  } else {
    PyErr_SetString( PyExc_TypeError, "not an array" );
    return NULL;
  }
  return obj;
}
%}

/*------------------------------- ARRAY INPUT -----------------------------*/

/*!
 * One dimensional array input.
 * Data can also be manipulated directly in C.
 * @a rtype ... C return data type
 * @a ctype ... C data type of the C function
 * @a atype ... PyArray_* suffix
 */
#define ARRAY1_IN( rtype, ctype, atype ) \
%typemap( in ) (ctype *array, int size) { \
  PyArrayObject *obj; \
  obj = helper_getCArrayObject( $input, PyArray_##atype, 1, 1 ); \
  if (!obj) return NULL; \
  $1 = (rtype *) obj->data; \
  $2 = obj->dimensions[0]; \
  Py_DECREF( obj ); \
};

/*!
 * Two dimensional C-style array input.
 * Data can also be manipulated directly in C.
 * @a rtype ... C return data type
 * @a ctype ... C data type of the C function
 * @a atype ... PyArray_* suffix
 */
#define ARRAY2_IN( rtype, ctype, atype ) \
%typemap( in ) (ctype *array, int rows, int cols) { \
  PyArrayObject *obj; \
  obj = helper_getCArrayObject( $input, PyArray_##atype, 2, 2 ); \
  if (!obj) return NULL; \
  $1 = (rtype *) obj->data; \
  $2 = obj->dimensions[0]; \
  $3 = obj->dimensions[1]; \
  Py_DECREF( obj ); \
};

/*------------------------------- ARRAY OUTPUT ----------------------------*/

/*!
 * One dimensional array output without copying data.
 * ATTENTION: you are responsible that this data stays alive
 *            basically for your whole python session !
 * @a ttype ... data type of the C function
 * @a atype ... PyArray_* suffix
 */
#define ARRAY1_OUT( ttype, atype ) \
%typemap( in, numinputs=0 ) (ttype *array, int *size) \
                            (ttype t1, int t2 ) { \
  $1 = &t1; \
  $2 = &t2; \
}; \
%typemap( argout ) (ttype *array, int *size) { \
  PyObject *obj; \
  int dim0[1]; dim0[0] = (*$2); \
  obj = PyArray_FromDimsAndData(1, dim0, PyArray_##atype, (char*)(*$1)); \
  $result = helper_appendToTuple( $result, obj ); \
};

/*!
 * Two dimensional C-style array output without copying data.
 * ATTENTION: you are responsible that this data stays alive
 *            basically for your whole python session !
 * @a ttype ... data type of the C function
 * @a atype ... PyArray_* suffix
 */
#define ARRAY2_OUT( ttype, atype ) \
%typemap( in, numinputs=0 ) (ttype *array, int *rows, int *cols) \
                            (ttype t1, int t2, int t3 ) { \
  $1 = &t1; \
  $2 = &t2; \
  $3 = &t3; \
}; \
%typemap( argout ) (ttype *array, int *rows, int *cols) { \
  PyObject *obj; \
  int dim0[2]; \
  dim0[0] = (*$2); dim0[1] = (*$3); \
  obj = PyArray_FromDimsAndData(2, dim0, PyArray_##atype, (char*)(*$1)); \
  PyArrayObject *tmp = (PyArrayObject*)obj; \
  $result = helper_appendToTuple( $result, obj ); \
};

/*!
 * Two dimensional Fortran-style array output without copying data.
 * ATTENTION: you are responsible that this data stays alive
 *            basically for your whole python session !
 * @a ttype ... data type of the C function
 * @a atype ... PyArray_* suffix
 */
#define FARRAY2_OUT( ttype, atype ) \
%typemap( in, numinputs=0 ) (ttype *array, int *rows, int *cols) \
                            (ttype t1, int t2, int t3 ) { \
  $1 = &t1; \
  $2 = &t2; \
  $3 = &t3; \
}; \
%typemap( argout ) (ttype *array, int *rows, int *cols) { \
  PyObject *obj; \
  int dim0[2]; \
  dim0[0] = (*$2); dim0[1] = (*$3); \
  obj = PyArray_FromDimsAndData(2, dim0, PyArray_##atype, (char*)(*$1)); \
  PyArrayObject *tmp = (PyArrayObject*)obj; \
  tmp->flags = NPY_FARRAY; \
  int s = tmp->strides[1]; \
  tmp->strides[0] = s; \
  tmp->strides[1] = s * dim0[0]; \
  $result = helper_appendToTuple( $result, obj ); \
};

/*--------------------------- ARRAY OUTPUT COPY ---------------------------*/

/*!
 * One dimensional array output, copying the data.
 * @a ttype ... data type of the C function
 * @a atype ... PyArray_* suffix
 */
#define ARRAY1_OUT_COPY( ttype, atype ) \
%typemap( in, numinputs=0 ) (ttype *cparray, int *cpsize) \
                            (ttype t1, int t2 ) { \
  $1 = &t1; \
  $2 = &t2; \
}; \
%typemap( argout ) (ttype *cparray, int *cpsize) { \
  PyObject *obj; \
  int dim0[1]; dim0[0] = (*$2); \
  obj = PyArray_FromDims(1, dim0, PyArray_##atype); \
  char *data = ((PyArrayObject *)obj)->data; \
  int *strides = ((PyArrayObject *)obj)->strides; \
  ttype ptr; \
  for(int i=0; i<dim0[0]; ++i) { \
    ptr = (ttype) (data+i*strides[0]); \
    *ptr = (*$1)[i]; \
  } \
  $result = helper_appendToTuple( $result, obj ); \
};

/*!
 * Two dimensional C-style array output, copying the data.
 * @a ttype ... data type of the C function
 * @a atype ... PyArray_* suffix
 */
#define ARRAY2_OUT_COPY( ttype, atype ) \
%typemap( in, numinputs=0 ) (ttype *cparray, int *cprows, int *cpcols) \
                            (ttype t1, int t2, int t3 ) { \
  $1 = &t1; \
  $2 = &t2; \
  $3 = &t3; \
}; \
%typemap( argout ) (ttype *cparray, int *cprows, int *cpcols) { \
  PyObject *obj; \
  int dim0[2]; \
  dim0[0] = (*$2); dim0[1] = (*$3); \
  obj = PyArray_FromDims(2, dim0, PyArray_##atype); \
  char *data = ((PyArrayObject *)obj)->data; \
  int *strides = ((PyArrayObject *)obj)->strides; \
  ttype ptr; \
  for(int i=0; i<dim0[0]; ++i) { \
  for(int j=0; j<dim0[1]; ++j) { \
    ptr = (ttype) (data + i*strides[0] + j*strides[1]); \
    *ptr = (*$1)[ i*dim0[1] + j ]; \
  } } \
  $result = helper_appendToTuple( $result, obj ); \
};
