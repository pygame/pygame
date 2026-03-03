#ifndef BASE_H
#define BASE_H

#define NO_PYGAME_C_API
#define PYGAMEAPI_BASE_INTERNAL

#include "pygame.h"

#include <signal.h>
#include "doc/pygame_doc.h"
#include "pgarrinter.h"
#include "pgcompat.h"

/* This file controls all the initialization of
 * the module and the various SDL subsystems
 */

/*platform specific init stuff*/

#ifdef MS_WIN32 /*python gives us MS_WIN32*/
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
extern int
SDL_RegisterApp(const char *, Uint32, void *);
#endif

#if defined(macintosh)
#if (!defined(__MWERKS__) && !TARGET_API_MAC_CARBON)
QDGlobals pg_qd;
#endif
#endif

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define PAI_MY_ENDIAN '<'
#define PAI_OTHER_ENDIAN '>'
#define BUF_OTHER_ENDIAN '>'
#else
#define PAI_MY_ENDIAN '>'
#define PAI_OTHER_ENDIAN '<'
#define BUF_OTHER_ENDIAN '<'
#endif
#define BUF_MY_ENDIAN '='

/* Extended array struct */
typedef struct pg_capsule_interface_s {
    PyArrayInterface inter;
    Py_intptr_t imem[1];
} pgCapsuleInterface;

/* Py_buffer internal data for an array interface/struct */
typedef struct pg_view_internals_s {
    char format[4]; /* make 4 byte word sized */
    Py_ssize_t imem[1];
} pgViewInternals;

extern PG_PixelFormatEnum pg_default_convert_format;
extern PyObject *pgExc_BufferError;
extern PyObject *pgExc_SDLError;
extern SDL_Window *pg_default_window;
extern pgSurfaceObject *pg_default_screen;

void
pg_install_parachute(void);
void
pg_uninstall_parachute(void);
void
pg_atexit_quit(void);
int
pgGetArrayStruct(PyObject *, PyObject **, PyArrayInterface **);
PyObject *
pgArrayStruct_AsDict(PyArrayInterface *);
PyObject *
pgBuffer_AsArrayInterface(Py_buffer *);
PyObject *
pgBuffer_AsArrayStruct(Py_buffer *);
int
_pg_buffer_is_byteswapped(Py_buffer *);
void
pgBuffer_Release(pg_buffer *);
int
pgObject_GetBuffer(PyObject *, pg_buffer *, int);
static inline PyObject *
pgObject_getRectHelper(PyObject *, PyObject *const *, Py_ssize_t, PyObject *,
                       char *);
int
pgGetArrayInterface(PyObject **, PyObject *);
int
pgArrayStruct_AsBuffer(pg_buffer *, PyObject *, PyArrayInterface *, int);
int
_pg_arraystruct_as_buffer(Py_buffer *, PyObject *, PyArrayInterface *, int);
int
_pg_arraystruct_to_format(char *, PyArrayInterface *, int);
int
pgDict_AsBuffer(pg_buffer *, PyObject *, int);
int
_pg_shape_check(PyObject *);
int
_pg_typestr_check(PyObject *);
int
_pg_strides_check(PyObject *);
int
_pg_data_check(PyObject *);
int
_pg_is_int_tuple(PyObject *);
int
_pg_values_as_buffer(Py_buffer *, int, PyObject *, PyObject *, PyObject *,
                     PyObject *);
int
_pg_int_tuple_as_ssize_arr(PyObject *, Py_ssize_t *);
int
_pg_typestr_as_format(PyObject *, char *, Py_ssize_t *);
PyObject *
pg_view_get_typestr_obj(Py_buffer *);
PyObject *
pg_view_get_shape_obj(Py_buffer *);
PyObject *
pg_view_get_strides_obj(Py_buffer *);
PyObject *
pg_view_get_data_obj(Py_buffer *);
char
_pg_as_arrayinter_typekind(Py_buffer *);
char
_pg_as_arrayinter_byteorder(Py_buffer *);
int
_pg_as_arrayinter_flags(Py_buffer *);
pgCapsuleInterface *
_pg_new_capsuleinterface(Py_buffer *);
void
_pg_capsule_PyMem_Free(PyObject *);
PyObject *
_pg_shape_as_tuple(PyArrayInterface *);
PyObject *
_pg_typekind_as_str(PyArrayInterface *);
PyObject *
_pg_strides_as_tuple(PyArrayInterface *);
PyObject *
_pg_data_as_tuple(PyArrayInterface *);
PyObject *
pg_get_array_interface(PyObject *, PyObject *);
void
_pg_release_buffer_array(Py_buffer *);
void
_pg_release_buffer_generic(Py_buffer *);
SDL_Window *
pg_GetDefaultWindow(void);
void
pg_SetDefaultWindow(SDL_Window *);
pgSurfaceObject *
pg_GetDefaultWindowSurface(void);
void
pg_SetDefaultWindowSurface(pgSurfaceObject *);
int
pg_EnvShouldBlendAlphaSDL2(void);
int
pg_CheckSDLVersions(void);
void
pg_RegisterQuit(void (*func)(void));
PyObject *
pg_register_quit(PyObject *self, PyObject *value);
int
pg_mod_autoinit(const char *modname);
void
pg_mod_autoquit(const char *modname);
PyObject *
pg_init(PyObject *self, PyObject *_null);
PyObject *
pg_get_sdl_version(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *
pg_get_sdl_byteorder(PyObject *self, PyObject *_null);
void
_pg_quit(void);
PyObject *
pg_quit(PyObject *self, PyObject *_null);
PyObject *
pg_base_get_init(PyObject *self, PyObject *_null);
/* internal C API utility functions */
int
pg_IntFromObj(PyObject *obj, int *val);
int
pg_IntFromObjIndex(PyObject *obj, int _index, int *val);
int
pg_TwoIntsFromObj(PyObject *obj, int *val1, int *val2);
int
pg_FloatFromObj(PyObject *obj, float *val);
int
pg_FloatFromObjIndex(PyObject *obj, int _index, float *val);
int
pg_TwoFloatsFromObj(PyObject *obj, float *val1, float *val2);
static inline int
pg_DoubleFromObj(PyObject *obj, double *val);
/*Assumes obj is a Sequence, internal or conscious use only*/
static inline int
_pg_DoubleFromObjIndex(PyObject *obj, int index, double *val);
static inline int
pg_TwoDoublesFromObj(PyObject *obj, double *val1, double *val2);
static inline int
pg_TwoDoublesFromFastcallArgs(PyObject *const *args, Py_ssize_t nargs,
                              double *val1, double *val2);
int
pg_UintFromObj(PyObject *obj, Uint32 *val);
int
pg_UintFromObjIndex(PyObject *obj, int _index, Uint32 *val);
/* You probably want to use the pg_RGBAFromObjEx function instead of this. */
int
pg_RGBAFromObj(PyObject *obj, Uint8 *RGBA);
PyObject *
pg_get_error(PyObject *self, PyObject *_null);
PyObject *
pg_set_error(PyObject *s, PyObject *args);

/*error signal handlers(replacing SDL parachute)*/
void
pygame_parachute(int sig);

void
pg_SetDefaultConvertFormat(PG_PixelFormatEnum format);

PG_PixelFormatEnum
pg_GetDefaultConvertFormat(void);

#if defined(BUILD_STATIC) && defined(NO_PYGAME_C_API)
// in case of wasm+dynamic loading it could be a trampoline in the globals
// generated at runtime.
// when building static make global accessible symbol directly.
extern PyObject *pgExc_SDLError;
#endif

MODINIT_DEFINE(base);

/*=======static inline function definitions=======*/
static inline PyObject *
pgObject_getRectHelper(PyObject *rect, PyObject *const *args, Py_ssize_t nargs,
                       PyObject *kwnames, char *type)
{
    if (nargs > 0) {
        Py_DECREF(rect);
        return PyErr_Format(PyExc_TypeError,
                            "get_%s only accepts keyword arguments", type);
    }

    if (rect && kwnames) {
        Py_ssize_t i, sequence_len;
        PyObject **sequence_items;
        sequence_items = PySequence_Fast_ITEMS(kwnames);
        sequence_len = PyTuple_GET_SIZE(kwnames);

        for (i = 0; i < sequence_len; ++i) {
            if ((PyObject_SetAttr(rect, sequence_items[i], args[i]) == -1)) {
                Py_DECREF(rect);
                return NULL;
            }
        }
    }
    return rect;
}

static inline int
pg_DoubleFromObj(PyObject *obj, double *val)
{
    if (PyFloat_Check(obj)) {
        *val = PyFloat_AS_DOUBLE(obj);
        return 1;
    }

    *val = (double)PyLong_AsLong(obj);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    return 1;
}

/*Assumes obj is a Sequence, internal or conscious use only*/
static inline int
_pg_DoubleFromObjIndex(PyObject *obj, int index, double *val)
{
    int result = 0;

    PyObject *item = PySequence_ITEM(obj, index);
    if (!item) {
        PyErr_Clear();
        return 0;
    }
    result = pg_DoubleFromObj(item, val);
    Py_DECREF(item);

    return result;
}

static inline int
pg_TwoDoublesFromObj(PyObject *obj, double *val1, double *val2)
{
    Py_ssize_t length;
    /*Faster path for tuples and lists*/
    if (pgSequenceFast_Check(obj)) {
        length = PySequence_Fast_GET_SIZE(obj);
        PyObject **f_arr = PySequence_Fast_ITEMS(obj);
        if (length == 2) {
            if (!pg_DoubleFromObj(f_arr[0], val1) ||
                !pg_DoubleFromObj(f_arr[1], val2)) {
                return 0;
            }
        }
        else if (length == 1) {
            /* Handle case of ((x, y), ) 'nested sequence' */
            return pg_TwoDoublesFromObj(f_arr[0], val1, val2);
        }
        else {
            return 0;
        }
    }
    else if (PySequence_Check(obj)) {
        length = PySequence_Length(obj);
        if (length == 2) {
            if (!_pg_DoubleFromObjIndex(obj, 0, val1)) {
                return 0;
            }
            if (!_pg_DoubleFromObjIndex(obj, 1, val2)) {
                return 0;
            }
        }
        else if (length == 1 && !PyUnicode_Check(obj)) {
            /* Handle case of ((x, y), ) 'nested sequence' */
            PyObject *tmp = PySequence_ITEM(obj, 0);
            int ret = pg_TwoDoublesFromObj(tmp, val1, val2);
            Py_DECREF(tmp);
            return ret;
        }
        else {
            PyErr_Clear();
            return 0;
        }
    }
    else {
        return 0;
    }

    return 1;
}

static inline int
pg_TwoDoublesFromFastcallArgs(PyObject *const *args, Py_ssize_t nargs,
                              double *val1, double *val2)
{
    if (nargs == 1 && pg_TwoDoublesFromObj(args[0], val1, val2)) {
        return 1;
    }
    else if (nargs == 2 && pg_DoubleFromObj(args[0], val1) &&
             pg_DoubleFromObj(args[1], val2)) {
        return 1;
    }
    return 0;
}

#endif  // #ifndef BASE_H
