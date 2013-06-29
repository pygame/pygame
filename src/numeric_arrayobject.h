/*
 * Imported Numeric-24.2 numeric/arrayobject.h header file to make the
 * Numeric dependency obsolete.
 */

/*
 * Legal Notice
 *
 * *** Legal Notice for all LLNL-contributed files ***
 *
 * Copyright (c) 1996. The Regents of the University of California. All
 * rights reserved.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire
 * notice is included in all copies of any software which is or includes
 * a copy or modification of this software and in all copies of the
 * supporting documentation for such software.
 *
 * This work was produced at the University of California, Lawrence
 * Livermore National Laboratory under contract no. W-7405-ENG-48
 * between the U.S. Department of Energy and The Regents of the
 * University of California for the operation of UC LLNL.
 *
 * DISCLAIMER
 *
 * This software was prepared as an account of work sponsored by an
 * agency of the United States Government. Neither the United States
 * Government nor the University of California nor any of their
 * employees, makes any warranty, express or implied, or assumes any
 * liability or responsibility for the accuracy, completeness, or
 * usefulness of any information, apparatus, product, or process
 * disclosed, or represents that its use would not infringe
 * privately-owned rights. Reference herein to any specific commercial
 * products, process, or service by trade name, trademark, manufacturer,
 * or otherwise, does not necessarily constitute or imply its
 * endorsement, recommendation, or favoring by the United States
 * Government or the University of California. The views and opinions of
 * authors expressed herein do not necessarily state or reflect those of
 * the United States Government or the University of California, and
 * shall not be used for advertising or product endorsement purposes.
 */

#ifndef Py_ARRAYOBJECT_H
#define Py_ARRAYOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#define REFCOUNT(obj) (((PyObject *)(obj))->ob_refcnt)
#define MAX_ELSIZE 16

#define PyArray_UNSIGNED_TYPES

enum PyArray_TYPES {PyArray_CHAR, PyArray_UBYTE, PyArray_SBYTE,
                    PyArray_SHORT, PyArray_USHORT,
                    PyArray_INT, PyArray_UINT,
                    PyArray_LONG,
                    PyArray_FLOAT, PyArray_DOUBLE,
                    PyArray_CFLOAT, PyArray_CDOUBLE,
                    PyArray_OBJECT,
                    PyArray_NTYPES, PyArray_NOTYPE};

typedef void (PyArray_VectorUnaryFunc) (char *, int, char *, int, int);

typedef PyObject * (PyArray_GetItemFunc) (char *);
typedef int (PyArray_SetItemFunc) (PyObject *, char *);

typedef struct {
  PyArray_VectorUnaryFunc *cast[PyArray_NTYPES]; /* Functions to cast to */
                                                 /* all other types */
  PyArray_GetItemFunc *getitem;
  PyArray_SetItemFunc *setitem;

  int type_num, elsize;
  char *one, *zero;
  char type;

} PyArray_Descr;

/* Array flags */
#define CONTIGUOUS 1
#define OWN_DIMENSIONS 2
#define OWN_STRIDES 4
#define OWN_DATA 8
#define SAVESPACE 16

/* type bit */
#define SAVESPACEBIT 128

typedef struct {
  PyObject_HEAD
  char *data;
  int nd;
  int *dimensions, *strides;
  PyObject *base;
  PyArray_Descr *descr;
  int flags;
  PyObject *weakreflist;
} PyArrayObject;

/* Array Interface flags */
#define FORTRAN       0x002
#define ALIGNED       0x100
#define NOTSWAPPED    0x200
#define WRITEABLE     0x400

typedef struct {
  int version;
  int nd;
  char typekind;
  int itemsize;
  int flags;
  Py_intptr_t *shape;
  Py_intptr_t *strides;
  void *data;
} PyArrayInterface;


/*
 * C API
 */

/* Type definitions */

#define PyArray_Type_NUM 0

/* Function definitions */

/* The following are not intended for use in user code, only needed by umath. */
/* If you write your own match library, you might want this function. */
#define PyArray_SetNumericOps_RET int
#define PyArray_SetNumericOps_PROTO (PyObject *)
#define PyArray_SetNumericOps_NUM 1

#define PyArray_INCREF_RET int
#define PyArray_INCREF_PROTO (PyArrayObject *ap)
#define PyArray_INCREF_NUM 2

#define PyArray_XDECREF_RET int
#define PyArray_XDECREF_PROTO (PyArrayObject *ap)
#define PyArray_XDECREF_NUM 3

/* Export the array error object.  Is this a good idea?  */
#define PyArrayError_RET PyObject *
#define PyArrayError_PROTO (void)
#define PyArrayError_NUM 4

/* Set the array print function to be a python function */
#define PyArray_SetStringFunction_RET void
#define PyArray_SetStringFunction_PROTO (PyObject *op, int repr)
#define PyArray_SetStringFunction_NUM 5

/* Get the PyArray_Descr structure for a typecode */
#define PyArray_DescrFromType_RET PyArray_Descr *
#define PyArray_DescrFromType_PROTO (int)
#define PyArray_DescrFromType_NUM 6

/* Cast an array to a different type */
#define PyArray_Cast_RET PyObject *
#define PyArray_Cast_PROTO (PyArrayObject *, int)
#define PyArray_Cast_NUM 7

/* Check the type coercion rules */
#define PyArray_CanCastSafely_RET int
#define PyArray_CanCastSafely_PROTO (int fromtype, int totype)
#define PyArray_CanCastSafely_NUM 8

/* Return the typecode to use for an object if it was an array */
#define PyArray_ObjectType_RET int
#define PyArray_ObjectType_PROTO (PyObject *, int)
#define PyArray_ObjectType_NUM 9

#define _PyArray_multiply_list_RET int
#define _PyArray_multiply_list_PROTO (int *lp, int n)
#define _PyArray_multiply_list_NUM 10


/* The following defines the C API for the array object for most users */

#define PyArray_SIZE(mp) (_PyArray_multiply_list((mp)->dimensions, (mp)->nd))
#define PyArray_NBYTES(mp) ((mp)->descr->elsize * PyArray_SIZE(mp))
/* Obviously this needs some work. */
#define PyArray_ISCONTIGUOUS(m) ((m)->flags & CONTIGUOUS)
#define PyArray_ISSPACESAVER(m) (((PyArrayObject *)m)->flags & SAVESPACE)
#define PyScalarArray_Check(m) (PyArray_Check((m)) && (((PyArrayObject *)(m))->nd == 0))

/* Return the size in number of items of an array */
#define PyArray_Size_RET int
#define PyArray_Size_PROTO (PyObject *)
#define PyArray_Size_NUM 11


/* Array creation functions */
/* new_array = PyArray_FromDims(n_dimensions, dimensions[n_dimensions], item_type); */
#define PyArray_FromDims_RET PyObject *
#define PyArray_FromDims_PROTO (int, int *, int)
#define PyArray_FromDims_NUM 12

/* array_from_existing_data = PyArray_FromDimsAndData(n_dims, dims[n_dims], item_type, old_data); */
/* WARNING: using PyArray_FromDimsAndData is not reccommended!  It should only be used to refer to */
/* global arrays that will never be freed (like FORTRAN common blocks). */
#define PyArray_FromDimsAndData_RET PyObject *
#define PyArray_FromDimsAndData_PROTO (int, int *, int, char *)
#define PyArray_FromDimsAndData_NUM 13

/* Initialize from a python object. */

/* PyArray_ContiguousFromObject(object, typecode, min_dimensions, max_dimensions) */
/* if max_dimensions = 0, then any number of dimensions are allowed. */
/* If you want an exact number of dimensions, you should use max_dimensions */
/* = min_dimensions. */

#define PyArray_ContiguousFromObject_RET PyObject *
#define PyArray_ContiguousFromObject_PROTO (PyObject *, int, int, int)
#define PyArray_ContiguousFromObject_NUM 14

/* Same as contiguous, except guarantees a copy of the original data */
#define PyArray_CopyFromObject_RET PyObject *
#define PyArray_CopyFromObject_PROTO (PyObject *, int, int, int)
#define PyArray_CopyFromObject_NUM 15

/* Shouldn't be used unless you know what you're doing and are not scared by discontiguous arrays */
#define PyArray_FromObject_RET PyObject *
#define PyArray_FromObject_PROTO (PyObject *, int, int, int)
#define PyArray_FromObject_NUM 16

/* Return either an array, or if passed a 0d array return the appropriate python scalar */
#define PyArray_Return_RET PyObject *
#define PyArray_Return_PROTO (PyArrayObject *)
#define PyArray_Return_NUM 17

#define PyArray_Reshape_RET PyObject *
#define PyArray_Reshape_PROTO (PyArrayObject *ap, PyObject *shape)
#define PyArray_Reshape_NUM 18

#define PyArray_Copy_RET PyObject *
#define PyArray_Copy_PROTO (PyArrayObject *ap)
#define PyArray_Copy_NUM 19

#define PyArray_Take_RET PyObject *
#define PyArray_Take_PROTO (PyObject *ap, PyObject *items, int axis)
#define PyArray_Take_NUM 20

/*Getting arrays in various useful forms. */
#define PyArray_As1D_RET int
#define PyArray_As1D_PROTO (PyObject **op, char **ptr, int *d1, int typecode)
#define PyArray_As1D_NUM 21

#define PyArray_As2D_RET int
#define PyArray_As2D_PROTO (PyObject **op, char ***ptr, int *d1, int *d2, int typecode)
#define PyArray_As2D_NUM 22

#define PyArray_Free_RET int
#define PyArray_Free_PROTO (PyObject *op, char *ptr)
#define PyArray_Free_NUM 23

/* array_from_existing_data = PyArray_FromDimsAndDataAndDescr(n_dims, dims[n_dims], descr, old_data); */
/* WARNING: using PyArray_FromDimsAndDataAndDescr is not reccommended!  It should only be used to refer to */
/* global arrays that will never be freed (like FORTRAN common blocks). */
#define PyArray_FromDimsAndDataAndDescr_RET PyObject *
#define PyArray_FromDimsAndDataAndDescr_PROTO (int, int *, PyArray_Descr *, char *)
#define PyArray_FromDimsAndDataAndDescr_NUM 24

#define PyArray_Converter_RET int
#define PyArray_Converter_PROTO (PyObject *, PyObject **)
#define PyArray_Converter_NUM 25

#define PyArray_Put_RET PyObject *
#define PyArray_Put_PROTO (PyObject *ap, PyObject *items, PyObject* values)
#define PyArray_Put_NUM 26

#define PyArray_PutMask_RET PyObject *
#define PyArray_PutMask_PROTO (PyObject *ap, PyObject *mask, PyObject* values)
#define PyArray_PutMask_NUM 27

#define PyArray_CopyArray_RET int
#define PyArray_CopyArray_PROTO (PyArrayObject *dest, PyArrayObject *src)
#define PyArray_CopyArray_NUM 28

#define PyArray_ValidType_RET int
#define PyArray_ValidType_PROTO (int type)
#define PyArray_ValidType_NUM 29

/* Convert a Python object to a C int, if possible. Checks for
   potential overflow, which is important on machines where
   sizeof(int) != sizeof(long) (note that a Python int is a C long).
   Handles Python ints, Python longs, and any ArrayObject that
   works in int(). */
#define PyArray_IntegerAsInt_RET int
#define PyArray_IntegerAsInt_PROTO (PyObject *o)
#define PyArray_IntegerAsInt_NUM 30

/* Total number of C API pointers */
#define PyArray_API_pointers 31


#ifdef _ARRAY_MODULE

extern PyTypeObject PyArray_Type;
#define PyArray_Check(op) ((op)->ob_type == &PyArray_Type)

extern PyArray_SetNumericOps_RET PyArray_SetNumericOps \
       PyArray_SetNumericOps_PROTO;
extern PyArray_INCREF_RET PyArray_INCREF PyArray_INCREF_PROTO;
extern PyArray_XDECREF_RET PyArray_XDECREF PyArray_XDECREF_PROTO;
extern PyArrayError_RET PyArrayError PyArrayError_PROTO;
extern PyArray_SetStringFunction_RET PyArray_SetStringFunction \
       PyArray_SetStringFunction_PROTO;
extern PyArray_DescrFromType_RET PyArray_DescrFromType \
       PyArray_DescrFromType_PROTO;
extern PyArray_Cast_RET PyArray_Cast PyArray_Cast_PROTO;
extern PyArray_CanCastSafely_RET PyArray_CanCastSafely \
       PyArray_CanCastSafely_PROTO;
extern PyArray_ObjectType_RET PyArray_ObjectType PyArray_ObjectType_PROTO;
extern _PyArray_multiply_list_RET _PyArray_multiply_list \
       _PyArray_multiply_list_PROTO;
extern PyArray_Size_RET PyArray_Size PyArray_Size_PROTO;
extern PyArray_FromDims_RET PyArray_FromDims PyArray_FromDims_PROTO;
extern PyArray_FromDimsAndData_RET PyArray_FromDimsAndData \
       PyArray_FromDimsAndData_PROTO;
extern PyArray_FromDimsAndDataAndDescr_RET PyArray_FromDimsAndDataAndDescr \
       PyArray_FromDimsAndDataAndDescr_PROTO;
extern PyArray_ContiguousFromObject_RET PyArray_ContiguousFromObject \
       PyArray_ContiguousFromObject_PROTO;
extern PyArray_CopyFromObject_RET PyArray_CopyFromObject \
       PyArray_CopyFromObject_PROTO;
extern PyArray_FromObject_RET PyArray_FromObject PyArray_FromObject_PROTO;
extern PyArray_Return_RET PyArray_Return PyArray_Return_PROTO;
extern PyArray_Reshape_RET PyArray_Reshape PyArray_Reshape_PROTO;
extern PyArray_Copy_RET PyArray_Copy PyArray_Copy_PROTO;
extern PyArray_Take_RET PyArray_Take PyArray_Take_PROTO;
extern PyArray_As1D_RET PyArray_As1D PyArray_As1D_PROTO;
extern PyArray_As2D_RET PyArray_As2D PyArray_As2D_PROTO;
extern PyArray_Free_RET PyArray_Free PyArray_Free_PROTO;
extern PyArray_Converter_RET PyArray_Converter PyArray_Converter_PROTO;
extern PyArray_Put_RET PyArray_Put PyArray_Put_PROTO;
extern PyArray_PutMask_RET PyArray_PutMask PyArray_PutMask_PROTO;
extern PyArray_CopyArray_RET PyArray_CopyArray PyArray_CopyArray_PROTO;
extern PyArray_ValidType_RET PyArray_ValidType PyArray_ValidType_PROTO;
extern PyArray_IntegerAsInt_RET PyArray_IntegerAsInt PyArray_IntegerAsInt_PROTO;

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
#endif

/* C API address pointer */
#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **PyArray_API;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
void **PyArray_API;
#else
static void **PyArray_API;
#endif
#endif

#define PyArray_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyArray_API[PyArray_Type_NUM])
#define PyArray_Type *(PyTypeObject *)PyArray_API[PyArray_Type_NUM]

#define PyArray_SetNumericOps \
  (*(PyArray_SetNumericOps_RET (*)PyArray_SetNumericOps_PROTO) \
   PyArray_API[PyArray_SetNumericOps_NUM])
#define PyArray_INCREF \
  (*(PyArray_INCREF_RET (*)PyArray_INCREF_PROTO) \
   PyArray_API[PyArray_INCREF_NUM])
#define PyArray_XDECREF \
  (*(PyArray_XDECREF_RET (*)PyArray_XDECREF_PROTO) \
   PyArray_API[PyArray_XDECREF_NUM])
#define PyArrayError \
  (*(PyArrayError_RET (*)PyArrayError_PROTO) \
   PyArray_API[PyArrayError_NUM])
#define PyArray_SetStringFunction \
  (*(PyArray_SetStringFunction_RET (*)PyArray_SetStringFunction_PROTO) \
   PyArray_API[PyArray_SetStringFunction_NUM])
#define PyArray_DescrFromType \
  (*(PyArray_DescrFromType_RET (*)PyArray_DescrFromType_PROTO) \
   PyArray_API[PyArray_DescrFromType_NUM])
#define PyArray_Cast \
  (*(PyArray_Cast_RET (*)PyArray_Cast_PROTO) \
   PyArray_API[PyArray_Cast_NUM])
#define PyArray_CanCastSafely \
  (*(PyArray_CanCastSafely_RET (*)PyArray_CanCastSafely_PROTO) \
   PyArray_API[PyArray_CanCastSafely_NUM])
#define PyArray_ObjectType \
  (*(PyArray_ObjectType_RET (*)PyArray_ObjectType_PROTO) \
   PyArray_API[PyArray_ObjectType_NUM])
#define _PyArray_multiply_list \
  (*(_PyArray_multiply_list_RET (*)_PyArray_multiply_list_PROTO) \
   PyArray_API[_PyArray_multiply_list_NUM])
#define PyArray_Size \
  (*(PyArray_Size_RET (*)PyArray_Size_PROTO) \
   PyArray_API[PyArray_Size_NUM])
#define PyArray_FromDims \
  (*(PyArray_FromDims_RET (*)PyArray_FromDims_PROTO) \
   PyArray_API[PyArray_FromDims_NUM])
#define PyArray_FromDimsAndData \
  (*(PyArray_FromDimsAndData_RET (*)PyArray_FromDimsAndData_PROTO) \
   PyArray_API[PyArray_FromDimsAndData_NUM])
#define PyArray_FromDimsAndDataAndDescr \
  (*(PyArray_FromDimsAndDataAndDescr_RET (*)PyArray_FromDimsAndDataAndDescr_PROTO) \
   PyArray_API[PyArray_FromDimsAndDataAndDescr_NUM])
#define PyArray_ContiguousFromObject \
  (*(PyArray_ContiguousFromObject_RET (*)PyArray_ContiguousFromObject_PROTO) \
   PyArray_API[PyArray_ContiguousFromObject_NUM])
#define PyArray_CopyFromObject \
  (*(PyArray_CopyFromObject_RET (*)PyArray_CopyFromObject_PROTO) \
   PyArray_API[PyArray_CopyFromObject_NUM])
#define PyArray_FromObject \
  (*(PyArray_FromObject_RET (*)PyArray_FromObject_PROTO) \
   PyArray_API[PyArray_FromObject_NUM])
#define PyArray_Return \
  (*(PyArray_Return_RET (*)PyArray_Return_PROTO) \
   PyArray_API[PyArray_Return_NUM])
#define PyArray_Reshape \
  (*(PyArray_Reshape_RET (*)PyArray_Reshape_PROTO) \
   PyArray_API[PyArray_Reshape_NUM])
#define PyArray_Copy \
  (*(PyArray_Copy_RET (*)PyArray_Copy_PROTO) \
   PyArray_API[PyArray_Copy_NUM])
#define PyArray_Take \
  (*(PyArray_Take_RET (*)PyArray_Take_PROTO) \
   PyArray_API[PyArray_Take_NUM])
#define PyArray_As1D \
  (*(PyArray_As1D_RET (*)PyArray_As1D_PROTO) \
   PyArray_API[PyArray_As1D_NUM])
#define PyArray_As2D \
  (*(PyArray_As2D_RET (*)PyArray_As2D_PROTO) \
   PyArray_API[PyArray_As2D_NUM])
#define PyArray_Free \
  (*(PyArray_Free_RET (*)PyArray_Free_PROTO) \
   PyArray_API[PyArray_Free_NUM])
#define PyArray_Converter \
  (*(PyArray_Converter_RET (*)PyArray_Converter_PROTO) \
   PyArray_API[PyArray_Converter_NUM])
#define PyArray_Put \
  (*(PyArray_Put_RET (*)PyArray_Put_PROTO) \
   PyArray_API[PyArray_Put_NUM])
#define PyArray_PutMask \
  (*(PyArray_PutMask_RET (*)PyArray_PutMask_PROTO) \
   PyArray_API[PyArray_PutMask_NUM])
#define PyArray_CopyArray \
  (*(PyArray_CopyArray_RET (*)PyArray_CopyArray_PROTO) \
   PyArray_API[PyArray_CopyArray_NUM])
#define PyArray_ValidType \
  (*(PyArray_ValidType_RET (*)PyArray_ValidType_PROTO) \
   PyArray_API[PyArray_ValidType_NUM])
#define PyArray_IntegerAsInt \
  (*(PyArray_IntegerAsInt_RET (*)PyArray_IntegerAsInt_PROTO) \
   PyArray_API[PyArray_IntegerAsInt_NUM])

#define import_array() \
{ \
  PyObject *numpy = PyImport_ImportModule("_numpy"); \
  if (numpy != NULL) { \
    PyObject *module_dict = PyModule_GetDict(numpy); \
    PyObject *c_api_object = PyDict_GetItemString(module_dict, "_ARRAY_API"); \
    if (PyCObject_Check(c_api_object)) { \
      PyArray_API = (void **)PyCObject_AsVoidPtr(c_api_object); \
    } \
  } \
}

#endif


#ifdef __cplusplus
}
#endif
#endif /* !Py_ARRAYOBJECT_H */
