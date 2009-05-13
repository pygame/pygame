/* 0.9.7 on Fri Feb  6 11:12:52 2009 */

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"
#ifndef PY_LONG_LONG
  #define PY_LONG_LONG LONG_LONG
#endif
#if PY_VERSION_HEX < 0x02050000
  typedef int Py_ssize_t;
  #define PY_SSIZE_T_MAX INT_MAX
  #define PY_SSIZE_T_MIN INT_MIN
  #define PyInt_FromSsize_t(z) PyInt_FromLong(z)
  #define PyInt_AsSsize_t(o)	PyInt_AsLong(o)
#endif
#ifndef WIN32
  #ifndef __stdcall
    #define __stdcall
  #endif
  #ifndef __cdecl
    #define __cdecl
  #endif
#endif
#ifdef __cplusplus
#define __PYX_EXTERN_C extern "C"
#else
#define __PYX_EXTERN_C extern
#endif
#include <math.h>
#include "portmidi.h"
#include "porttime.h"


typedef struct {PyObject **p; char *s;} __Pyx_InternTabEntry; /*proto*/
typedef struct {PyObject **p; char *s; long n;} __Pyx_StringTabEntry; /*proto*/

static PyObject *__pyx_m;
static PyObject *__pyx_b;
static int __pyx_lineno;
static char *__pyx_filename;
static char **__pyx_f;

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list); /*proto*/

static int __Pyx_PrintItem(PyObject *); /*proto*/
static int __Pyx_PrintNewline(void); /*proto*/

static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb); /*proto*/

static PyObject *__Pyx_GetName(PyObject *dict, PyObject *name); /*proto*/

static int __Pyx_InternStrings(__Pyx_InternTabEntry *t); /*proto*/

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t); /*proto*/

static void __Pyx_AddTraceback(char *funcname); /*proto*/

/* Declarations from pypm */

struct __pyx_obj_4pypm_Output {
  PyObject_HEAD
  int i;
  PmStream *midi;
  int debug;
};

struct __pyx_obj_4pypm_Input {
  PyObject_HEAD
  PmStream *midi;
  int debug;
  int i;
};



static PyTypeObject *__pyx_ptype_4pypm_Output = 0;
static PyTypeObject *__pyx_ptype_4pypm_Input = 0;
static PyObject *__pyx_k3;
static PyObject *__pyx_k4;
static PyObject *__pyx_k5;
static PyObject *__pyx_k6;


/* Implementation of pypm */

static char __pyx_k1[] = "0.03";

static PyObject *__pyx_n___version__;
static PyObject *__pyx_n_array;
static PyObject *__pyx_n_FILT_ACTIVE;
static PyObject *__pyx_n_FILT_SYSEX;
static PyObject *__pyx_n_FILT_CLOCK;
static PyObject *__pyx_n_FILT_PLAY;
static PyObject *__pyx_n_FILT_F9;
static PyObject *__pyx_n_FILT_TICK;
static PyObject *__pyx_n_FILT_FD;
static PyObject *__pyx_n_FILT_UNDEFINED;
static PyObject *__pyx_n_FILT_RESET;
static PyObject *__pyx_n_FILT_REALTIME;
static PyObject *__pyx_n_FILT_NOTE;
static PyObject *__pyx_n_FILT_CHANNEL_AFTERTOUCH;
static PyObject *__pyx_n_FILT_POLY_AFTERTOUCH;
static PyObject *__pyx_n_FILT_AFTERTOUCH;
static PyObject *__pyx_n_FILT_PROGRAM;
static PyObject *__pyx_n_FILT_CONTROL;
static PyObject *__pyx_n_FILT_PITCHBEND;
static PyObject *__pyx_n_FILT_MTC;
static PyObject *__pyx_n_FILT_SONG_POSITION;
static PyObject *__pyx_n_FILT_SONG_SELECT;
static PyObject *__pyx_n_FILT_TUNE;
static PyObject *__pyx_n_FALSE;
static PyObject *__pyx_n_TRUE;

static PyObject *__pyx_k1p;

static PyObject *__pyx_f_4pypm_Initialize(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_Initialize[] = "\nInitialize: call this first\n    ";
static PyObject *__pyx_f_4pypm_Initialize(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":128 */
  Pm_Initialize();

  /* "/home/rsd/dev/pygame/src/pypm.pyx":129 */
  Pt_Start(1,NULL,NULL);

  __pyx_r = Py_None; Py_INCREF(Py_None);
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_Terminate(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_Terminate[] = "\nTerminate: call this to clean up Midi streams when done.\nIf you do not call this on Windows machines when you are\ndone with MIDI, your system may crash.\n    ";
static PyObject *__pyx_f_4pypm_Terminate(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  Pm_Terminate();

  __pyx_r = Py_None; Py_INCREF(Py_None);
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_GetDefaultInputDeviceID(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_f_4pypm_GetDefaultInputDeviceID(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  __pyx_1 = PyInt_FromLong(Pm_GetDefaultInputDeviceID()); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 140; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("pypm.GetDefaultInputDeviceID");
  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_GetDefaultOutputDeviceID(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_f_4pypm_GetDefaultOutputDeviceID(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  __pyx_1 = PyInt_FromLong(Pm_GetDefaultOutputDeviceID()); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 143; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("pypm.GetDefaultOutputDeviceID");
  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_CountDevices(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static PyObject *__pyx_f_4pypm_CountDevices(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  __pyx_1 = PyInt_FromLong(Pm_CountDevices()); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 146; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("pypm.CountDevices");
  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_GetDeviceInfo(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_GetDeviceInfo[] = "\nGetDeviceInfo(<device number>): returns 5 parameters\n  - underlying MIDI API\n  - device name\n  - TRUE iff input is available\n  - TRUE iff output is available\n  - TRUE iff device stream is already open\n    ";
static PyObject *__pyx_f_4pypm_GetDeviceInfo(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_i = 0;
  PmDeviceInfo *__pyx_v_info;
  PyObject *__pyx_r;
  PmDeviceID __pyx_1;
  int __pyx_2;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  PyObject *__pyx_6 = 0;
  PyObject *__pyx_7 = 0;
  PyObject *__pyx_8 = 0;
  static char *__pyx_argnames[] = {"i",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_i)) return 0;
  Py_INCREF(__pyx_v_i);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":160 */
  __pyx_1 = PyInt_AsLong(__pyx_v_i); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 160; goto __pyx_L1;}
  __pyx_v_info = ((PmDeviceInfo *)Pm_GetDeviceInfo(__pyx_1));

  /* "/home/rsd/dev/pygame/src/pypm.pyx":162 */
  __pyx_2 = (__pyx_v_info != NULL);
  if (__pyx_2) {
    __pyx_3 = PyString_FromString(__pyx_v_info->interf); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 162; goto __pyx_L1;}
    __pyx_4 = PyString_FromString(__pyx_v_info->name); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 162; goto __pyx_L1;}
    __pyx_5 = PyInt_FromLong(__pyx_v_info->input); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 162; goto __pyx_L1;}
    __pyx_6 = PyInt_FromLong(__pyx_v_info->output); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 162; goto __pyx_L1;}
    __pyx_7 = PyInt_FromLong(__pyx_v_info->opened); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 162; goto __pyx_L1;}
    __pyx_8 = PyTuple_New(5); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 162; goto __pyx_L1;}
    PyTuple_SET_ITEM(__pyx_8, 0, __pyx_3);
    PyTuple_SET_ITEM(__pyx_8, 1, __pyx_4);
    PyTuple_SET_ITEM(__pyx_8, 2, __pyx_5);
    PyTuple_SET_ITEM(__pyx_8, 3, __pyx_6);
    PyTuple_SET_ITEM(__pyx_8, 4, __pyx_7);
    __pyx_3 = 0;
    __pyx_4 = 0;
    __pyx_5 = 0;
    __pyx_6 = 0;
    __pyx_7 = 0;
    __pyx_r = __pyx_8;
    __pyx_8 = 0;
    goto __pyx_L0;
    goto __pyx_L2;
  }
  /*else*/ {
    __pyx_r = Py_None; Py_INCREF(Py_None);
    goto __pyx_L0;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  Py_XDECREF(__pyx_6);
  Py_XDECREF(__pyx_7);
  Py_XDECREF(__pyx_8);
  __Pyx_AddTraceback("pypm.GetDeviceInfo");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_i);
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_Time(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_Time[] = "\nTime() returns the current time in ms\nof the PortMidi timer\n    ";
static PyObject *__pyx_f_4pypm_Time(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  __pyx_1 = PyInt_FromLong(Pt_Time()); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 170; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  __Pyx_AddTraceback("pypm.Time");
  __pyx_r = 0;
  __pyx_L0:;
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_GetErrorText(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_GetErrorText[] = "\nGetErrorText(<err num>) returns human-readable error\nmessages translated from error numbers\n    ";
static PyObject *__pyx_f_4pypm_GetErrorText(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_err = 0;
  PyObject *__pyx_r;
  PmError __pyx_1;
  PyObject *__pyx_2 = 0;
  static char *__pyx_argnames[] = {"err",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_err)) return 0;
  Py_INCREF(__pyx_v_err);
  __pyx_1 = ((PmError)PyInt_AsLong(__pyx_v_err)); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 177; goto __pyx_L1;}
  __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_1)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 177; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("pypm.GetErrorText");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_err);
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_Channel(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_Channel[] = "\nChannel(<chan>) is used with ChannelMask on input MIDI streams.\nExample: to receive input on channels 1 and 10 on a MIDI\n         stream called MidiIn:\nMidiIn.SetChannelMask(pypm.Channel(1) | pypm.Channel(10))\n\nnote: PyPortMidi Channel function has been altered from\n      the original PortMidi c call to correct for what\n      seems to be a bug --- i.e. channel filters were\n      all numbered from 0 to 15 instead of 1 to 16.\n    ";
static PyObject *__pyx_f_4pypm_Channel(PyObject *__pyx_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_chan = 0;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  int __pyx_3;
  static char *__pyx_argnames[] = {"chan",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_chan)) return 0;
  Py_INCREF(__pyx_v_chan);
  __pyx_1 = PyInt_FromLong(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 191; goto __pyx_L1;}
  __pyx_2 = PyNumber_Subtract(__pyx_v_chan, __pyx_1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 191; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_3 = PyInt_AsLong(__pyx_2); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 191; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyInt_FromLong(Pm_Channel(__pyx_3)); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 191; goto __pyx_L1;}
  __pyx_r = __pyx_1;
  __pyx_1 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("pypm.Channel");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_chan);
  return __pyx_r;
}

static PyObject *__pyx_k7p;
static PyObject *__pyx_k8p;
static PyObject *__pyx_k9p;

static char __pyx_k7[] = "Opening Midi Output";
static char __pyx_k8[] = "Unable to open Midi OutputDevice=";
static char __pyx_k9[] = " err=";

static int __pyx_f_4pypm_6Output___init__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static int __pyx_f_4pypm_6Output___init__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_OutputDevice = 0;
  PyObject *__pyx_v_latency = 0;
  PmError __pyx_v_err;
  PmTimeProcPtr __pyx_v_PmPtr;
  PyObject *__pyx_v_s;
  int __pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  long __pyx_3;
  static char *__pyx_argnames[] = {"OutputDevice","latency",0};
  __pyx_v_latency = __pyx_k3;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_OutputDevice, &__pyx_v_latency)) return -1;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_OutputDevice);
  Py_INCREF(__pyx_v_latency);
  __pyx_v_s = Py_None; Py_INCREF(Py_None);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":211 */
  __pyx_1 = PyInt_AsLong(__pyx_v_OutputDevice); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 211; goto __pyx_L1;}
  ((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->i = __pyx_1;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":212 */
  ((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->debug = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":214 */
  __pyx_2 = PyInt_FromLong(0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 214; goto __pyx_L1;}
  if (PyObject_Cmp(__pyx_v_latency, __pyx_2, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 214; goto __pyx_L1;}
  __pyx_1 = __pyx_1 == 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_1) {
    __pyx_v_PmPtr = NULL;
    goto __pyx_L2;
  }
  /*else*/ {
    __pyx_v_PmPtr = ((PmTimeProcPtr)(&Pt_Time));
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":218 */
  __pyx_1 = ((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->debug;
  if (__pyx_1) {
    if (__Pyx_PrintItem(__pyx_k7p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; goto __pyx_L1;}
    if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 218; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":220 */
  __pyx_3 = PyInt_AsLong(__pyx_v_latency); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 220; goto __pyx_L1;}
  __pyx_v_err = Pm_OpenOutput((&((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->midi),((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->i,NULL,0,__pyx_v_PmPtr,NULL,__pyx_3);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":221 */
  __pyx_1 = (__pyx_v_err < 0);
  if (__pyx_1) {

    /* "/home/rsd/dev/pygame/src/pypm.pyx":222 */
    __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 222; goto __pyx_L1;}
    Py_DECREF(__pyx_v_s);
    __pyx_v_s = __pyx_2;
    __pyx_2 = 0;

    /* "/home/rsd/dev/pygame/src/pypm.pyx":225 */
    __pyx_1 = (!(__pyx_v_err == (-10000)));
    if (__pyx_1) {
      __Pyx_Raise(PyExc_Exception, __pyx_v_s, 0);
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 226; goto __pyx_L1;}
      goto __pyx_L5;
    }
    /*else*/ {
      if (__Pyx_PrintItem(__pyx_k8p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 228; goto __pyx_L1;}
      if (__Pyx_PrintItem(__pyx_v_OutputDevice) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 228; goto __pyx_L1;}
      if (__Pyx_PrintItem(__pyx_k9p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 228; goto __pyx_L1;}
      if (__Pyx_PrintItem(__pyx_v_s) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 228; goto __pyx_L1;}
      if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 228; goto __pyx_L1;}
    }
    __pyx_L5:;
    goto __pyx_L4;
  }
  __pyx_L4:;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("pypm.Output.__init__");
  __pyx_r = -1;
  __pyx_L0:;
  Py_DECREF(__pyx_v_s);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_OutputDevice);
  Py_DECREF(__pyx_v_latency);
  return __pyx_r;
}

static PyObject *__pyx_k10p;

static char __pyx_k10[] = "Closing MIDI output stream and destroying instance";

static void __pyx_f_4pypm_6Output___dealloc__(PyObject *__pyx_v_self); /*proto*/
static void __pyx_f_4pypm_6Output___dealloc__(PyObject *__pyx_v_self) {
  PyObject *__pyx_v_err;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  PmError __pyx_3;
  Py_INCREF(__pyx_v_self);
  __pyx_v_err = Py_None; Py_INCREF(Py_None);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":231 */
  __pyx_1 = ((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->debug;
  if (__pyx_1) {
    if (__Pyx_PrintItem(__pyx_k10p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 231; goto __pyx_L1;}
    if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 231; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":232 */
  __pyx_2 = PyInt_FromLong(Pm_Abort(((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->midi)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 232; goto __pyx_L1;}
  Py_DECREF(__pyx_v_err);
  __pyx_v_err = __pyx_2;
  __pyx_2 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":233 */
  __pyx_2 = PyInt_FromLong(0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 233; goto __pyx_L1;}
  if (PyObject_Cmp(__pyx_v_err, __pyx_2, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 233; goto __pyx_L1;}
  __pyx_1 = __pyx_1 < 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_1) {
    __pyx_3 = ((PmError)PyInt_AsLong(__pyx_v_err)); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 233; goto __pyx_L1;}
    __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_3)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 233; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_2, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 233; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":234 */
  __pyx_2 = PyInt_FromLong(Pm_Close(((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->midi)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 234; goto __pyx_L1;}
  Py_DECREF(__pyx_v_err);
  __pyx_v_err = __pyx_2;
  __pyx_2 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":235 */
  __pyx_2 = PyInt_FromLong(0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 235; goto __pyx_L1;}
  if (PyObject_Cmp(__pyx_v_err, __pyx_2, &__pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 235; goto __pyx_L1;}
  __pyx_1 = __pyx_1 < 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_1) {
    __pyx_3 = ((PmError)PyInt_AsLong(__pyx_v_err)); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 235; goto __pyx_L1;}
    __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_3)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 235; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_2, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 235; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("pypm.Output.__dealloc__");
  __pyx_L0:;
  Py_DECREF(__pyx_v_err);
  Py_DECREF(__pyx_v_self);
}

static PyObject *__pyx_n_range;

static PyObject *__pyx_k11p;
static PyObject *__pyx_k12p;
static PyObject *__pyx_k13p;
static PyObject *__pyx_k14p;
static PyObject *__pyx_k15p;

static char __pyx_k11[] = "maximum list length is 1024";
static char __pyx_k12[] = " arguments in event list";
static char __pyx_k13[] = " : ";
static char __pyx_k14[] = " : ";
static char __pyx_k15[] = "writing to midi buffer";

static PyObject *__pyx_f_4pypm_6Output_Write(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_6Output_Write[] = "\nWrite(data)\n    output a series of MIDI information in the form of a list:\n         Write([[[status <,data1><,data2><,data3>],timestamp],\n                [[status <,data1><,data2><,data3>],timestamp],...])\n    <data> fields are optional\n    example: choose program change 1 at time 20000 and\n    send note 65 with velocity 100 500 ms later.\n         Write([[[0xc0,0,0],20000],[[0x90,60,100],20500]])\n    notes:\n      1. timestamps will be ignored if latency = 0.\n      2. To get a note to play immediately, send MIDI info with\n         timestamp read from function Time.\n      3. understanding optional data fields:\n           Write([[[0xc0,0,0],20000]]) is equivalent to\n           Write([[[0xc0],20000]])\n        ";
static PyObject *__pyx_f_4pypm_6Output_Write(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_data = 0;
  PmEvent __pyx_v_buffer[1024];
  PmError __pyx_v_err;
  int __pyx_v_i;
  PyObject *__pyx_v_loop1;
  PyObject *__pyx_r;
  Py_ssize_t __pyx_1;
  int __pyx_2;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PyObject *__pyx_5 = 0;
  Py_ssize_t __pyx_6;
  PyObject *__pyx_7 = 0;
  PyObject *__pyx_8 = 0;
  PyObject *__pyx_9 = 0;
  PmMessage __pyx_10;
  PmTimestamp __pyx_11;
  static char *__pyx_argnames[] = {"data",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_data)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_data);
  __pyx_v_loop1 = Py_None; Py_INCREF(Py_None);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":259 */
  __pyx_1 = PyObject_Length(__pyx_v_data); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 259; goto __pyx_L1;}
  __pyx_2 = (__pyx_1 > 1024);
  if (__pyx_2) {
    __Pyx_Raise(PyExc_IndexError, __pyx_k11p, 0);
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 259; goto __pyx_L1;}
    goto __pyx_L2;
  }
  /*else*/ {
    __pyx_3 = __Pyx_GetName(__pyx_b, __pyx_n_range); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 261; goto __pyx_L1;}
    __pyx_1 = PyObject_Length(__pyx_v_data); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 261; goto __pyx_L1;}
    __pyx_4 = PyInt_FromSsize_t(__pyx_1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 261; goto __pyx_L1;}
    __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 261; goto __pyx_L1;}
    PyTuple_SET_ITEM(__pyx_5, 0, __pyx_4);
    __pyx_4 = 0;
    __pyx_4 = PyObject_CallObject(__pyx_3, __pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 261; goto __pyx_L1;}
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    __pyx_3 = PyObject_GetIter(__pyx_4); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 261; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    for (;;) {
      __pyx_5 = PyIter_Next(__pyx_3);
      if (!__pyx_5) {
        if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 261; goto __pyx_L1;}
        break;
      }
      Py_DECREF(__pyx_v_loop1);
      __pyx_v_loop1 = __pyx_5;
      __pyx_5 = 0;

      /* "/home/rsd/dev/pygame/src/pypm.pyx":262 */
      __pyx_4 = PyObject_GetItem(__pyx_v_data, __pyx_v_loop1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; goto __pyx_L1;}
      __pyx_5 = PySequence_GetItem(__pyx_4, 0); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; goto __pyx_L1;}
      Py_DECREF(__pyx_4); __pyx_4 = 0;
      __pyx_1 = PyObject_Length(__pyx_5); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 262; goto __pyx_L1;}
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      __pyx_4 = PyObject_GetItem(__pyx_v_data, __pyx_v_loop1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 263; goto __pyx_L1;}
      __pyx_5 = PySequence_GetItem(__pyx_4, 0); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 263; goto __pyx_L1;}
      Py_DECREF(__pyx_4); __pyx_4 = 0;
      __pyx_6 = PyObject_Length(__pyx_5); if (__pyx_6 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 263; goto __pyx_L1;}
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      __pyx_2 = ((__pyx_1 > 4) | (__pyx_6 < 1));
      if (__pyx_2) {
        __pyx_4 = PyObject_GetItem(__pyx_v_data, __pyx_v_loop1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        __pyx_5 = PySequence_GetItem(__pyx_4, 0); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        Py_DECREF(__pyx_4); __pyx_4 = 0;
        __pyx_1 = PyObject_Length(__pyx_5); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        Py_DECREF(__pyx_5); __pyx_5 = 0;
        __pyx_4 = PyInt_FromSsize_t(__pyx_1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        PyTuple_SET_ITEM(__pyx_5, 0, __pyx_4);
        __pyx_4 = 0;
        __pyx_4 = PyObject_CallObject(((PyObject *)(&PyString_Type)), __pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        Py_DECREF(__pyx_5); __pyx_5 = 0;
        __pyx_5 = PyNumber_Add(__pyx_4, __pyx_k12p); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        Py_DECREF(__pyx_4); __pyx_4 = 0;
        __Pyx_Raise(PyExc_IndexError, __pyx_5, 0);
        Py_DECREF(__pyx_5); __pyx_5 = 0;
        {__pyx_filename = __pyx_f[0]; __pyx_lineno = 264; goto __pyx_L1;}
        goto __pyx_L5;
      }
      __pyx_L5:;

      /* "/home/rsd/dev/pygame/src/pypm.pyx":265 */
      __pyx_6 = PyInt_AsSsize_t(__pyx_v_loop1); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 265; goto __pyx_L1;}
      (__pyx_v_buffer[__pyx_6]).message = 0;

      /* "/home/rsd/dev/pygame/src/pypm.pyx":266 */
      __pyx_4 = __Pyx_GetName(__pyx_b, __pyx_n_range); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      __pyx_5 = PyObject_GetItem(__pyx_v_data, __pyx_v_loop1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      __pyx_7 = PySequence_GetItem(__pyx_5, 0); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      __pyx_1 = PyObject_Length(__pyx_7); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      Py_DECREF(__pyx_7); __pyx_7 = 0;
      __pyx_5 = PyInt_FromSsize_t(__pyx_1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      __pyx_7 = PyTuple_New(1); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      PyTuple_SET_ITEM(__pyx_7, 0, __pyx_5);
      __pyx_5 = 0;
      __pyx_5 = PyObject_CallObject(__pyx_4, __pyx_7); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      Py_DECREF(__pyx_4); __pyx_4 = 0;
      Py_DECREF(__pyx_7); __pyx_7 = 0;
      __pyx_4 = PyObject_GetIter(__pyx_5); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      for (;;) {
        __pyx_7 = PyIter_Next(__pyx_4);
        if (!__pyx_7) {
          if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
          break;
        }
        __pyx_2 = PyInt_AsLong(__pyx_7); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 266; goto __pyx_L1;}
        Py_DECREF(__pyx_7); __pyx_7 = 0;
        __pyx_v_i = __pyx_2;
        __pyx_6 = PyInt_AsSsize_t(__pyx_v_loop1); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        __pyx_5 = PyInt_FromLong((__pyx_v_buffer[__pyx_6]).message); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        __pyx_7 = PyObject_GetItem(__pyx_v_data, __pyx_v_loop1); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        __pyx_8 = PySequence_GetItem(__pyx_7, 0); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        Py_DECREF(__pyx_7); __pyx_7 = 0;
        __pyx_7 = PySequence_GetItem(__pyx_8, __pyx_v_i); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        Py_DECREF(__pyx_8); __pyx_8 = 0;
        __pyx_8 = PyInt_FromLong(0xFF); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        __pyx_9 = PyNumber_And(__pyx_7, __pyx_8); if (!__pyx_9) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        Py_DECREF(__pyx_7); __pyx_7 = 0;
        Py_DECREF(__pyx_8); __pyx_8 = 0;
        __pyx_7 = PyInt_FromLong((8 * __pyx_v_i)); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        __pyx_8 = PyNumber_Lshift(__pyx_9, __pyx_7); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        Py_DECREF(__pyx_9); __pyx_9 = 0;
        Py_DECREF(__pyx_7); __pyx_7 = 0;
        __pyx_9 = PyNumber_Add(__pyx_5, __pyx_8); if (!__pyx_9) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        Py_DECREF(__pyx_5); __pyx_5 = 0;
        Py_DECREF(__pyx_8); __pyx_8 = 0;
        __pyx_10 = PyInt_AsLong(__pyx_9); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        Py_DECREF(__pyx_9); __pyx_9 = 0;
        __pyx_1 = PyInt_AsSsize_t(__pyx_v_loop1); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 267; goto __pyx_L1;}
        (__pyx_v_buffer[__pyx_1]).message = __pyx_10;
      }
      Py_DECREF(__pyx_4); __pyx_4 = 0;

      /* "/home/rsd/dev/pygame/src/pypm.pyx":268 */
      __pyx_7 = PyObject_GetItem(__pyx_v_data, __pyx_v_loop1); if (!__pyx_7) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 268; goto __pyx_L1;}
      __pyx_5 = PySequence_GetItem(__pyx_7, 1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 268; goto __pyx_L1;}
      Py_DECREF(__pyx_7); __pyx_7 = 0;
      __pyx_11 = PyInt_AsLong(__pyx_5); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 268; goto __pyx_L1;}
      Py_DECREF(__pyx_5); __pyx_5 = 0;
      __pyx_6 = PyInt_AsSsize_t(__pyx_v_loop1); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 268; goto __pyx_L1;}
      (__pyx_v_buffer[__pyx_6]).timestamp = __pyx_11;

      /* "/home/rsd/dev/pygame/src/pypm.pyx":269 */
      __pyx_2 = ((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->debug;
      if (__pyx_2) {
        if (__Pyx_PrintItem(__pyx_v_loop1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        if (__Pyx_PrintItem(__pyx_k13p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        __pyx_1 = PyInt_AsSsize_t(__pyx_v_loop1); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        __pyx_8 = PyInt_FromLong((__pyx_v_buffer[__pyx_1]).message); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        if (__Pyx_PrintItem(__pyx_8) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        Py_DECREF(__pyx_8); __pyx_8 = 0;
        if (__Pyx_PrintItem(__pyx_k14p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        __pyx_6 = PyInt_AsSsize_t(__pyx_v_loop1); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        __pyx_9 = PyInt_FromLong((__pyx_v_buffer[__pyx_6]).timestamp); if (!__pyx_9) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        if (__Pyx_PrintItem(__pyx_9) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        Py_DECREF(__pyx_9); __pyx_9 = 0;
        if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 269; goto __pyx_L1;}
        goto __pyx_L8;
      }
      __pyx_L8:;
    }
    Py_DECREF(__pyx_3); __pyx_3 = 0;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":270 */
  __pyx_2 = ((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->debug;
  if (__pyx_2) {
    if (__Pyx_PrintItem(__pyx_k15p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 270; goto __pyx_L1;}
    if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 270; goto __pyx_L1;}
    goto __pyx_L9;
  }
  __pyx_L9:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":271 */
  __pyx_1 = PyObject_Length(__pyx_v_data); if (__pyx_1 == -1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 271; goto __pyx_L1;}
  __pyx_v_err = Pm_Write(((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->midi,__pyx_v_buffer,__pyx_1);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":272 */
  __pyx_2 = (__pyx_v_err < 0);
  if (__pyx_2) {
    __pyx_4 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 272; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_4, 0);
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 272; goto __pyx_L1;}
    goto __pyx_L10;
  }
  __pyx_L10:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  Py_XDECREF(__pyx_5);
  Py_XDECREF(__pyx_7);
  Py_XDECREF(__pyx_8);
  Py_XDECREF(__pyx_9);
  __Pyx_AddTraceback("pypm.Output.Write");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_loop1);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_data);
  return __pyx_r;
}

static PyObject *__pyx_k16p;

static char __pyx_k16[] = "Writing to MIDI buffer";

static PyObject *__pyx_f_4pypm_6Output_WriteShort(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_6Output_WriteShort[] = "\nWriteShort(status <, data1><, data2>)\n     output MIDI information of 3 bytes or less.\n     data fields are optional\n     status byte could be:\n          0xc0 = program change\n          0x90 = note on\n          etc.\n          data bytes are optional and assumed 0 if omitted\n     example: note 65 on with velocity 100\n          WriteShort(0x90,65,100)\n        ";
static PyObject *__pyx_f_4pypm_6Output_WriteShort(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_status = 0;
  PyObject *__pyx_v_data1 = 0;
  PyObject *__pyx_v_data2 = 0;
  PmEvent __pyx_v_buffer[1];
  PmError __pyx_v_err;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  PmMessage __pyx_5;
  int __pyx_6;
  static char *__pyx_argnames[] = {"status","data1","data2",0};
  __pyx_v_data1 = __pyx_k4;
  __pyx_v_data2 = __pyx_k5;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|OO", __pyx_argnames, &__pyx_v_status, &__pyx_v_data1, &__pyx_v_data2)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_status);
  Py_INCREF(__pyx_v_data1);
  Py_INCREF(__pyx_v_data2);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":290 */
  (__pyx_v_buffer[0]).timestamp = Pt_Time();

  /* "/home/rsd/dev/pygame/src/pypm.pyx":291 */
  __pyx_1 = PyInt_FromLong(16); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  __pyx_2 = PyNumber_Lshift(__pyx_v_data2, __pyx_1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_1 = PyInt_FromLong(0xFF0000); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  __pyx_3 = PyNumber_And(__pyx_2, __pyx_1); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_2 = PyInt_FromLong(8); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  __pyx_1 = PyNumber_Lshift(__pyx_v_data1, __pyx_2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_2 = PyInt_FromLong(0xFF00); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  __pyx_4 = PyNumber_And(__pyx_1, __pyx_2); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_1 = PyNumber_Or(__pyx_3, __pyx_4); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  __pyx_2 = PyInt_FromLong(0xFF); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  __pyx_3 = PyNumber_And(__pyx_v_status, __pyx_2); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  __pyx_4 = PyNumber_Or(__pyx_1, __pyx_3); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  Py_DECREF(__pyx_3); __pyx_3 = 0;
  __pyx_5 = PyInt_AsLong(__pyx_4); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 291; goto __pyx_L1;}
  Py_DECREF(__pyx_4); __pyx_4 = 0;
  (__pyx_v_buffer[0]).message = __pyx_5;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":292 */
  __pyx_6 = ((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->debug;
  if (__pyx_6) {
    if (__Pyx_PrintItem(__pyx_k16p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 292; goto __pyx_L1;}
    if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 292; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":293 */
  __pyx_v_err = Pm_Write(((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->midi,__pyx_v_buffer,1);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":294 */
  __pyx_6 = (__pyx_v_err < 0);
  if (__pyx_6) {
    __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 294; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_2, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 294; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("pypm.Output.WriteShort");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_status);
  Py_DECREF(__pyx_v_data1);
  Py_DECREF(__pyx_v_data2);
  return __pyx_r;
}

static PyObject *__pyx_n_B;
static PyObject *__pyx_n_tostring;


static PyObject *__pyx_f_4pypm_6Output_WriteSysEx(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_6Output_WriteSysEx[] = "\n        WriteSysEx(<timestamp>,<msg>)\n        writes a timestamped system-exclusive midi message.\n        <msg> can be a *list* or a *string*\n        example:\n            (assuming y is an input MIDI stream)\n            y.WriteSysEx(0,\'\\xF0\\x7D\\x10\\x11\\x12\\x13\\xF7\')\n                              is equivalent to\n            y.WriteSysEx(pypm.Time,\n            [0xF0, 0x7D, 0x10, 0x11, 0x12, 0x13, 0xF7])\n        ";
static PyObject *__pyx_f_4pypm_6Output_WriteSysEx(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_when = 0;
  PyObject *__pyx_v_msg = 0;
  PmError __pyx_v_err;
  unsigned char *__pyx_v_cmsg;
  PtTimestamp __pyx_v_CurTime;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  int __pyx_3;
  PyObject *__pyx_4 = 0;
  PmTimestamp __pyx_5;
  static char *__pyx_argnames[] = {"when","msg",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "OO", __pyx_argnames, &__pyx_v_when, &__pyx_v_msg)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_when);
  Py_INCREF(__pyx_v_msg);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":312 */
  __pyx_1 = PyTuple_New(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 312; goto __pyx_L1;}
  Py_INCREF(__pyx_v_msg);
  PyTuple_SET_ITEM(__pyx_1, 0, __pyx_v_msg);
  __pyx_2 = PyObject_CallObject(((PyObject *)(&PyType_Type)), __pyx_1); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 312; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  __pyx_3 = __pyx_2 == ((PyObject *)(&PyList_Type));
  Py_DECREF(__pyx_2); __pyx_2 = 0;
  if (__pyx_3) {
    __pyx_1 = __Pyx_GetName(__pyx_m, __pyx_n_array); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 313; goto __pyx_L1;}
    __pyx_2 = PyObject_GetAttr(__pyx_1, __pyx_n_array); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 313; goto __pyx_L1;}
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    __pyx_1 = PyTuple_New(2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 313; goto __pyx_L1;}
    Py_INCREF(__pyx_n_B);
    PyTuple_SET_ITEM(__pyx_1, 0, __pyx_n_B);
    Py_INCREF(__pyx_v_msg);
    PyTuple_SET_ITEM(__pyx_1, 1, __pyx_v_msg);
    __pyx_4 = PyObject_CallObject(__pyx_2, __pyx_1); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 313; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    __pyx_2 = PyObject_GetAttr(__pyx_4, __pyx_n_tostring); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 313; goto __pyx_L1;}
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    __pyx_1 = PyObject_CallObject(__pyx_2, 0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 313; goto __pyx_L1;}
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    Py_DECREF(__pyx_v_msg);
    __pyx_v_msg = __pyx_1;
    __pyx_1 = 0;
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":314 */
  __pyx_v_cmsg = ((unsigned char *)__pyx_v_msg);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":316 */
  __pyx_v_CurTime = Pt_Time();

  /* "/home/rsd/dev/pygame/src/pypm.pyx":317 */
  __pyx_5 = PyInt_AsLong(__pyx_v_when); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 317; goto __pyx_L1;}
  __pyx_v_err = Pm_WriteSysEx(((struct __pyx_obj_4pypm_Output *)__pyx_v_self)->midi,__pyx_5,__pyx_v_cmsg);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":318 */
  __pyx_3 = (__pyx_v_err < 0);
  if (__pyx_3) {
    __pyx_4 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 318; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_4, 0);
    Py_DECREF(__pyx_4); __pyx_4 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 318; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":319 */
  while (1) {
    __pyx_3 = (Pt_Time() == __pyx_v_CurTime);
    if (!__pyx_3) break;
  }

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("pypm.Output.WriteSysEx");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_when);
  Py_DECREF(__pyx_v_msg);
  return __pyx_r;
}

static PyObject *__pyx_k18p;

static char __pyx_k18[] = "MIDI input opened.";

static int __pyx_f_4pypm_5Input___init__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static int __pyx_f_4pypm_5Input___init__(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_InputDevice = 0;
  PyObject *__pyx_v_buffersize = 0;
  PmError __pyx_v_err;
  int __pyx_r;
  int __pyx_1;
  long __pyx_2;
  PyObject *__pyx_3 = 0;
  static char *__pyx_argnames[] = {"InputDevice","buffersize",0};
  __pyx_v_buffersize = __pyx_k6;
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O|O", __pyx_argnames, &__pyx_v_InputDevice, &__pyx_v_buffersize)) return -1;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_InputDevice);
  Py_INCREF(__pyx_v_buffersize);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":334 */
  __pyx_1 = PyInt_AsLong(__pyx_v_InputDevice); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 334; goto __pyx_L1;}
  ((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->i = __pyx_1;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":335 */
  ((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->debug = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":336 */
  __pyx_2 = PyInt_AsLong(__pyx_v_buffersize); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 336; goto __pyx_L1;}
  __pyx_v_err = Pm_OpenInput((&((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi),((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->i,NULL,__pyx_2,(&Pt_Time),NULL);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":337 */
  __pyx_1 = (__pyx_v_err < 0);
  if (__pyx_1) {
    __pyx_3 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 337; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_3, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 337; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":338 */
  __pyx_1 = ((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->debug;
  if (__pyx_1) {
    if (__Pyx_PrintItem(__pyx_k18p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 338; goto __pyx_L1;}
    if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 338; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  __pyx_r = 0;
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_3);
  __Pyx_AddTraceback("pypm.Input.__init__");
  __pyx_r = -1;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_InputDevice);
  Py_DECREF(__pyx_v_buffersize);
  return __pyx_r;
}

static PyObject *__pyx_k19p;

static char __pyx_k19[] = "Closing MIDI input stream and destroying instance";

static void __pyx_f_4pypm_5Input___dealloc__(PyObject *__pyx_v_self); /*proto*/
static void __pyx_f_4pypm_5Input___dealloc__(PyObject *__pyx_v_self) {
  PmError __pyx_v_err;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  Py_INCREF(__pyx_v_self);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":342 */
  __pyx_1 = ((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->debug;
  if (__pyx_1) {
    if (__Pyx_PrintItem(__pyx_k19p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 342; goto __pyx_L1;}
    if (__Pyx_PrintNewline() < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 342; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":346 */
  __pyx_v_err = Pm_Close(((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":347 */
  __pyx_1 = (__pyx_v_err < 0);
  if (__pyx_1) {
    __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 347; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_2, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 347; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("pypm.Input.__dealloc__");
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
}

static PyObject *__pyx_f_4pypm_5Input_SetFilter(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_5Input_SetFilter[] = "\n    SetFilter(<filters>) sets filters on an open input stream\n    to drop selected input types. By default, only active sensing\n    messages are filtered. To prohibit, say, active sensing and\n    sysex messages, call\n    SetFilter(stream, FILT_ACTIVE | FILT_SYSEX);\n\n    Filtering is useful when midi routing or midi thru functionality\n    is being provided by the user application.\n    For example, you may want to exclude timing messages\n    (clock, MTC, start/stop/continue), while allowing note-related\n    messages to pass. Or you may be using a sequencer or drum-machine\n    for MIDI clock information but want to exclude any notes\n    it may play.\n\n    Note: SetFilter empties the buffer after setting the filter,\n    just in case anything got through.\n        ";
static PyObject *__pyx_f_4pypm_5Input_SetFilter(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_filters = 0;
  PmEvent __pyx_v_buffer[1];
  PmError __pyx_v_err;
  PyObject *__pyx_r;
  long __pyx_1;
  int __pyx_2;
  PyObject *__pyx_3 = 0;
  static char *__pyx_argnames[] = {"filters",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_filters)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_filters);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":371 */
  __pyx_1 = PyInt_AsLong(__pyx_v_filters); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 371; goto __pyx_L1;}
  __pyx_v_err = Pm_SetFilter(((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi,__pyx_1);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":373 */
  __pyx_2 = (__pyx_v_err < 0);
  if (__pyx_2) {
    __pyx_3 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 373; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_3, 0);
    Py_DECREF(__pyx_3); __pyx_3 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 373; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":375 */
  while (1) {
    __pyx_2 = (Pm_Poll(((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi) != pmNoError);
    if (!__pyx_2) break;

    /* "/home/rsd/dev/pygame/src/pypm.pyx":377 */
    __pyx_v_err = Pm_Read(((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi,__pyx_v_buffer,1);

    /* "/home/rsd/dev/pygame/src/pypm.pyx":378 */
    __pyx_2 = (__pyx_v_err < 0);
    if (__pyx_2) {
      __pyx_3 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 378; goto __pyx_L1;}
      __Pyx_Raise(PyExc_Exception, __pyx_3, 0);
      Py_DECREF(__pyx_3); __pyx_3 = 0;
      {__pyx_filename = __pyx_f[0]; __pyx_lineno = 378; goto __pyx_L1;}
      goto __pyx_L5;
    }
    __pyx_L5:;
  }

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_3);
  __Pyx_AddTraceback("pypm.Input.SetFilter");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_filters);
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_5Input_SetChannelMask(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_5Input_SetChannelMask[] = "\n    SetChannelMask(<mask>) filters incoming messages based on channel.\n    The mask is a 16-bit bitfield corresponding to appropriate channels\n    Channel(<channel>) can assist in calling this function.\n    i.e. to set receive only input on channel 1, call with\n    SetChannelMask(Channel(1))\n    Multiple channels should be OR\'d together, like\n    SetChannelMask(Channel(10) | Channel(11))\n    note: PyPortMidi Channel function has been altered from\n          the original PortMidi c call to correct for what\n          seems to be a bug --- i.e. channel filters were\n          all numbered from 0 to 15 instead of 1 to 16.\n        ";
static PyObject *__pyx_f_4pypm_5Input_SetChannelMask(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_mask = 0;
  PmError __pyx_v_err;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  static char *__pyx_argnames[] = {"mask",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_mask)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_mask);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":395 */
  __pyx_1 = PyInt_AsLong(__pyx_v_mask); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 395; goto __pyx_L1;}
  __pyx_v_err = Pm_SetChannelMask(((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi,__pyx_1);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":396 */
  __pyx_1 = (__pyx_v_err < 0);
  if (__pyx_1) {
    __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 396; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_2, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 396; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("pypm.Input.SetChannelMask");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_mask);
  return __pyx_r;
}

static PyObject *__pyx_f_4pypm_5Input_Poll(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_5Input_Poll[] = "\n    Poll tests whether input is available,\n    returning TRUE, FALSE, or an error value.\n        ";
static PyObject *__pyx_f_4pypm_5Input_Poll(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PmError __pyx_v_err;
  PyObject *__pyx_r;
  int __pyx_1;
  PyObject *__pyx_2 = 0;
  static char *__pyx_argnames[] = {0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "", __pyx_argnames)) return 0;
  Py_INCREF(__pyx_v_self);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":404 */
  __pyx_v_err = Pm_Poll(((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":405 */
  __pyx_1 = (__pyx_v_err < 0);
  if (__pyx_1) {
    __pyx_2 = PyString_FromString(Pm_GetErrorText(__pyx_v_err)); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 405; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_2, 0);
    Py_DECREF(__pyx_2); __pyx_2 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 405; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":406 */
  __pyx_2 = PyInt_FromLong(__pyx_v_err); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 406; goto __pyx_L1;}
  __pyx_r = __pyx_2;
  __pyx_2 = 0;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_2);
  __Pyx_AddTraceback("pypm.Input.Poll");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_self);
  return __pyx_r;
}

static PyObject *__pyx_n_append;

static PyObject *__pyx_k20p;
static PyObject *__pyx_k21p;

static char __pyx_k20[] = "maximum buffer length is 1024";
static char __pyx_k21[] = "minimum buffer length is 1";

static PyObject *__pyx_f_4pypm_5Input_Read(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds); /*proto*/
static char __pyx_doc_4pypm_5Input_Read[] = "\nRead(length): returns up to <length> midi events stored in\nthe buffer and returns them as a list:\n[[[status,data1,data2,data3],timestamp],\n [[status,data1,data2,data3],timestamp],...]\nexample: Read(50) returns all the events in the buffer,\n         up to 50 events.\n        ";
static PyObject *__pyx_f_4pypm_5Input_Read(PyObject *__pyx_v_self, PyObject *__pyx_args, PyObject *__pyx_kwds) {
  PyObject *__pyx_v_length = 0;
  PmEvent __pyx_v_buffer[1024];
  PyObject *__pyx_v_x;
  PyObject *__pyx_v_NumEvents;
  PyObject *__pyx_v_loop;
  PyObject *__pyx_r;
  PyObject *__pyx_1 = 0;
  int __pyx_2;
  long __pyx_3;
  PmError __pyx_4;
  PyObject *__pyx_5 = 0;
  PyObject *__pyx_6 = 0;
  Py_ssize_t __pyx_7;
  PyObject *__pyx_8 = 0;
  PyObject *__pyx_9 = 0;
  PyObject *__pyx_10 = 0;
  PyObject *__pyx_11 = 0;
  static char *__pyx_argnames[] = {"length",0};
  if (!PyArg_ParseTupleAndKeywords(__pyx_args, __pyx_kwds, "O", __pyx_argnames, &__pyx_v_length)) return 0;
  Py_INCREF(__pyx_v_self);
  Py_INCREF(__pyx_v_length);
  __pyx_v_x = Py_None; Py_INCREF(Py_None);
  __pyx_v_NumEvents = Py_None; Py_INCREF(Py_None);
  __pyx_v_loop = Py_None; Py_INCREF(Py_None);

  /* "/home/rsd/dev/pygame/src/pypm.pyx":418 */
  __pyx_1 = PyList_New(0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 418; goto __pyx_L1;}
  Py_DECREF(__pyx_v_x);
  __pyx_v_x = __pyx_1;
  __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":420 */
  __pyx_1 = PyInt_FromLong(1024); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 420; goto __pyx_L1;}
  if (PyObject_Cmp(__pyx_v_length, __pyx_1, &__pyx_2) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 420; goto __pyx_L1;}
  __pyx_2 = __pyx_2 > 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  if (__pyx_2) {
    __Pyx_Raise(PyExc_IndexError, __pyx_k20p, 0);
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 420; goto __pyx_L1;}
    goto __pyx_L2;
  }
  __pyx_L2:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":421 */
  __pyx_1 = PyInt_FromLong(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 421; goto __pyx_L1;}
  if (PyObject_Cmp(__pyx_v_length, __pyx_1, &__pyx_2) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 421; goto __pyx_L1;}
  __pyx_2 = __pyx_2 < 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  if (__pyx_2) {
    __Pyx_Raise(PyExc_IndexError, __pyx_k21p, 0);
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 421; goto __pyx_L1;}
    goto __pyx_L3;
  }
  __pyx_L3:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":422 */
  __pyx_3 = PyInt_AsLong(__pyx_v_length); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 422; goto __pyx_L1;}
  __pyx_1 = PyInt_FromLong(Pm_Read(((struct __pyx_obj_4pypm_Input *)__pyx_v_self)->midi,__pyx_v_buffer,__pyx_3)); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 422; goto __pyx_L1;}
  Py_DECREF(__pyx_v_NumEvents);
  __pyx_v_NumEvents = __pyx_1;
  __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":423 */
  __pyx_1 = PyInt_FromLong(0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 423; goto __pyx_L1;}
  if (PyObject_Cmp(__pyx_v_NumEvents, __pyx_1, &__pyx_2) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 423; goto __pyx_L1;}
  __pyx_2 = __pyx_2 < 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  if (__pyx_2) {
    __pyx_4 = ((PmError)PyInt_AsLong(__pyx_v_NumEvents)); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 423; goto __pyx_L1;}
    __pyx_1 = PyString_FromString(Pm_GetErrorText(__pyx_4)); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 423; goto __pyx_L1;}
    __Pyx_Raise(PyExc_Exception, __pyx_1, 0);
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    {__pyx_filename = __pyx_f[0]; __pyx_lineno = 423; goto __pyx_L1;}
    goto __pyx_L4;
  }
  __pyx_L4:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":424 */
  __pyx_1 = PyList_New(0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 424; goto __pyx_L1;}
  Py_DECREF(__pyx_v_x);
  __pyx_v_x = __pyx_1;
  __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":425 */
  __pyx_1 = PyInt_FromLong(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 425; goto __pyx_L1;}
  if (PyObject_Cmp(__pyx_v_NumEvents, __pyx_1, &__pyx_2) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 425; goto __pyx_L1;}
  __pyx_2 = __pyx_2 >= 0;
  Py_DECREF(__pyx_1); __pyx_1 = 0;
  if (__pyx_2) {
    __pyx_1 = __Pyx_GetName(__pyx_b, __pyx_n_range); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 426; goto __pyx_L1;}
    __pyx_5 = PyTuple_New(1); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 426; goto __pyx_L1;}
    Py_INCREF(__pyx_v_NumEvents);
    PyTuple_SET_ITEM(__pyx_5, 0, __pyx_v_NumEvents);
    __pyx_6 = PyObject_CallObject(__pyx_1, __pyx_5); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 426; goto __pyx_L1;}
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    Py_DECREF(__pyx_5); __pyx_5 = 0;
    __pyx_1 = PyObject_GetIter(__pyx_6); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 426; goto __pyx_L1;}
    Py_DECREF(__pyx_6); __pyx_6 = 0;
    for (;;) {
      __pyx_5 = PyIter_Next(__pyx_1);
      if (!__pyx_5) {
        if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 426; goto __pyx_L1;}
        break;
      }
      Py_DECREF(__pyx_v_loop);
      __pyx_v_loop = __pyx_5;
      __pyx_5 = 0;
      __pyx_6 = PyObject_GetAttr(__pyx_v_x, __pyx_n_append); if (!__pyx_6) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_7 = PyInt_AsSsize_t(__pyx_v_loop); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_5 = PyInt_FromLong(((__pyx_v_buffer[__pyx_7]).message & 0xff)); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_7 = PyInt_AsSsize_t(__pyx_v_loop); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_8 = PyInt_FromLong((((__pyx_v_buffer[__pyx_7]).message >> 8) & 0xFF)); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_7 = PyInt_AsSsize_t(__pyx_v_loop); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_9 = PyInt_FromLong((((__pyx_v_buffer[__pyx_7]).message >> 16) & 0xFF)); if (!__pyx_9) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_7 = PyInt_AsSsize_t(__pyx_v_loop); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_10 = PyInt_FromLong((((__pyx_v_buffer[__pyx_7]).message >> 24) & 0xFF)); if (!__pyx_10) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_11 = PyList_New(4); if (!__pyx_11) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      PyList_SET_ITEM(__pyx_11, 0, __pyx_5);
      PyList_SET_ITEM(__pyx_11, 1, __pyx_8);
      PyList_SET_ITEM(__pyx_11, 2, __pyx_9);
      PyList_SET_ITEM(__pyx_11, 3, __pyx_10);
      __pyx_5 = 0;
      __pyx_8 = 0;
      __pyx_9 = 0;
      __pyx_10 = 0;
      __pyx_7 = PyInt_AsSsize_t(__pyx_v_loop); if (PyErr_Occurred()) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_5 = PyInt_FromLong((__pyx_v_buffer[__pyx_7]).timestamp); if (!__pyx_5) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      __pyx_8 = PyList_New(2); if (!__pyx_8) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      PyList_SET_ITEM(__pyx_8, 0, __pyx_11);
      PyList_SET_ITEM(__pyx_8, 1, __pyx_5);
      __pyx_11 = 0;
      __pyx_5 = 0;
      __pyx_9 = PyTuple_New(1); if (!__pyx_9) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      PyTuple_SET_ITEM(__pyx_9, 0, __pyx_8);
      __pyx_8 = 0;
      __pyx_10 = PyObject_CallObject(__pyx_6, __pyx_9); if (!__pyx_10) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 427; goto __pyx_L1;}
      Py_DECREF(__pyx_6); __pyx_6 = 0;
      Py_DECREF(__pyx_9); __pyx_9 = 0;
      Py_DECREF(__pyx_10); __pyx_10 = 0;
    }
    Py_DECREF(__pyx_1); __pyx_1 = 0;
    goto __pyx_L5;
  }
  __pyx_L5:;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":428 */
  Py_INCREF(__pyx_v_x);
  __pyx_r = __pyx_v_x;
  goto __pyx_L0;

  __pyx_r = Py_None; Py_INCREF(Py_None);
  goto __pyx_L0;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_5);
  Py_XDECREF(__pyx_6);
  Py_XDECREF(__pyx_8);
  Py_XDECREF(__pyx_9);
  Py_XDECREF(__pyx_10);
  Py_XDECREF(__pyx_11);
  __Pyx_AddTraceback("pypm.Input.Read");
  __pyx_r = 0;
  __pyx_L0:;
  Py_DECREF(__pyx_v_x);
  Py_DECREF(__pyx_v_NumEvents);
  Py_DECREF(__pyx_v_loop);
  Py_DECREF(__pyx_v_self);
  Py_DECREF(__pyx_v_length);
  return __pyx_r;
}

static __Pyx_InternTabEntry __pyx_intern_tab[] = {
  {&__pyx_n_B, "B"},
  {&__pyx_n_FALSE, "FALSE"},
  {&__pyx_n_FILT_ACTIVE, "FILT_ACTIVE"},
  {&__pyx_n_FILT_AFTERTOUCH, "FILT_AFTERTOUCH"},
  {&__pyx_n_FILT_CHANNEL_AFTERTOUCH, "FILT_CHANNEL_AFTERTOUCH"},
  {&__pyx_n_FILT_CLOCK, "FILT_CLOCK"},
  {&__pyx_n_FILT_CONTROL, "FILT_CONTROL"},
  {&__pyx_n_FILT_F9, "FILT_F9"},
  {&__pyx_n_FILT_FD, "FILT_FD"},
  {&__pyx_n_FILT_MTC, "FILT_MTC"},
  {&__pyx_n_FILT_NOTE, "FILT_NOTE"},
  {&__pyx_n_FILT_PITCHBEND, "FILT_PITCHBEND"},
  {&__pyx_n_FILT_PLAY, "FILT_PLAY"},
  {&__pyx_n_FILT_POLY_AFTERTOUCH, "FILT_POLY_AFTERTOUCH"},
  {&__pyx_n_FILT_PROGRAM, "FILT_PROGRAM"},
  {&__pyx_n_FILT_REALTIME, "FILT_REALTIME"},
  {&__pyx_n_FILT_RESET, "FILT_RESET"},
  {&__pyx_n_FILT_SONG_POSITION, "FILT_SONG_POSITION"},
  {&__pyx_n_FILT_SONG_SELECT, "FILT_SONG_SELECT"},
  {&__pyx_n_FILT_SYSEX, "FILT_SYSEX"},
  {&__pyx_n_FILT_TICK, "FILT_TICK"},
  {&__pyx_n_FILT_TUNE, "FILT_TUNE"},
  {&__pyx_n_FILT_UNDEFINED, "FILT_UNDEFINED"},
  {&__pyx_n_TRUE, "TRUE"},
  {&__pyx_n___version__, "__version__"},
  {&__pyx_n_append, "append"},
  {&__pyx_n_array, "array"},
  {&__pyx_n_range, "range"},
  {&__pyx_n_tostring, "tostring"},
  {0, 0}
};

static __Pyx_StringTabEntry __pyx_string_tab[] = {
  {&__pyx_k1p, __pyx_k1, sizeof(__pyx_k1)},
  {&__pyx_k7p, __pyx_k7, sizeof(__pyx_k7)},
  {&__pyx_k8p, __pyx_k8, sizeof(__pyx_k8)},
  {&__pyx_k9p, __pyx_k9, sizeof(__pyx_k9)},
  {&__pyx_k10p, __pyx_k10, sizeof(__pyx_k10)},
  {&__pyx_k11p, __pyx_k11, sizeof(__pyx_k11)},
  {&__pyx_k12p, __pyx_k12, sizeof(__pyx_k12)},
  {&__pyx_k13p, __pyx_k13, sizeof(__pyx_k13)},
  {&__pyx_k14p, __pyx_k14, sizeof(__pyx_k14)},
  {&__pyx_k15p, __pyx_k15, sizeof(__pyx_k15)},
  {&__pyx_k16p, __pyx_k16, sizeof(__pyx_k16)},
  {&__pyx_k18p, __pyx_k18, sizeof(__pyx_k18)},
  {&__pyx_k19p, __pyx_k19, sizeof(__pyx_k19)},
  {&__pyx_k20p, __pyx_k20, sizeof(__pyx_k20)},
  {&__pyx_k21p, __pyx_k21, sizeof(__pyx_k21)},
  {0, 0, 0}
};

static PyObject *__pyx_tp_new_4pypm_Output(PyTypeObject *t, PyObject *a, PyObject *k) {
  PyObject *o = (*t->tp_alloc)(t, 0);
  if (!o) return 0;
  return o;
}

static void __pyx_tp_dealloc_4pypm_Output(PyObject *o) {
  {
    PyObject *etype, *eval, *etb;
    PyErr_Fetch(&etype, &eval, &etb);
    ++o->ob_refcnt;
    __pyx_f_4pypm_6Output___dealloc__(o);
    if (PyErr_Occurred()) PyErr_WriteUnraisable(o);
    --o->ob_refcnt;
    PyErr_Restore(etype, eval, etb);
  }
  (*o->ob_type->tp_free)(o);
}

static struct PyMethodDef __pyx_methods_4pypm_Output[] = {
  {"Write", (PyCFunction)__pyx_f_4pypm_6Output_Write, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_6Output_Write},
  {"WriteShort", (PyCFunction)__pyx_f_4pypm_6Output_WriteShort, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_6Output_WriteShort},
  {"WriteSysEx", (PyCFunction)__pyx_f_4pypm_6Output_WriteSysEx, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_6Output_WriteSysEx},
  {0, 0, 0, 0}
};

static PyNumberMethods __pyx_tp_as_number_Output = {
  0, /*nb_add*/
  0, /*nb_subtract*/
  0, /*nb_multiply*/
  0, /*nb_divide*/
  0, /*nb_remainder*/
  0, /*nb_divmod*/
  0, /*nb_power*/
  0, /*nb_negative*/
  0, /*nb_positive*/
  0, /*nb_absolute*/
  0, /*nb_nonzero*/
  0, /*nb_invert*/
  0, /*nb_lshift*/
  0, /*nb_rshift*/
  0, /*nb_and*/
  0, /*nb_xor*/
  0, /*nb_or*/
  0, /*nb_coerce*/
  0, /*nb_int*/
  0, /*nb_long*/
  0, /*nb_float*/
  0, /*nb_oct*/
  0, /*nb_hex*/
  0, /*nb_inplace_add*/
  0, /*nb_inplace_subtract*/
  0, /*nb_inplace_multiply*/
  0, /*nb_inplace_divide*/
  0, /*nb_inplace_remainder*/
  0, /*nb_inplace_power*/
  0, /*nb_inplace_lshift*/
  0, /*nb_inplace_rshift*/
  0, /*nb_inplace_and*/
  0, /*nb_inplace_xor*/
  0, /*nb_inplace_or*/
  0, /*nb_floor_divide*/
  0, /*nb_true_divide*/
  0, /*nb_inplace_floor_divide*/
  0, /*nb_inplace_true_divide*/
  #if Py_TPFLAGS_DEFAULT & Py_TPFLAGS_HAVE_INDEX
  0, /*nb_index*/
  #endif
};

static PySequenceMethods __pyx_tp_as_sequence_Output = {
  0, /*sq_length*/
  0, /*sq_concat*/
  0, /*sq_repeat*/
  0, /*sq_item*/
  0, /*sq_slice*/
  0, /*sq_ass_item*/
  0, /*sq_ass_slice*/
  0, /*sq_contains*/
  0, /*sq_inplace_concat*/
  0, /*sq_inplace_repeat*/
};

static PyMappingMethods __pyx_tp_as_mapping_Output = {
  0, /*mp_length*/
  0, /*mp_subscript*/
  0, /*mp_ass_subscript*/
};

static PyBufferProcs __pyx_tp_as_buffer_Output = {
  0, /*bf_getreadbuffer*/
  0, /*bf_getwritebuffer*/
  0, /*bf_getsegcount*/
  0, /*bf_getcharbuffer*/
};

PyTypeObject __pyx_type_4pypm_Output = {
  PyObject_HEAD_INIT(0)
  0, /*ob_size*/
  "pypm.Output", /*tp_name*/
  sizeof(struct __pyx_obj_4pypm_Output), /*tp_basicsize*/
  0, /*tp_itemsize*/
  __pyx_tp_dealloc_4pypm_Output, /*tp_dealloc*/
  0, /*tp_print*/
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  &__pyx_tp_as_number_Output, /*tp_as_number*/
  &__pyx_tp_as_sequence_Output, /*tp_as_sequence*/
  &__pyx_tp_as_mapping_Output, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  &__pyx_tp_as_buffer_Output, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "\nclass Output:\n    define an output MIDI stream. Takes the form:\n        x = pypm.Output(MidiOutputDevice, latency)\n    latency is in ms.\n    If latency = 0 then timestamps for output are ignored.\n    ", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  __pyx_methods_4pypm_Output, /*tp_methods*/
  0, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  __pyx_f_4pypm_6Output___init__, /*tp_init*/
  0, /*tp_alloc*/
  __pyx_tp_new_4pypm_Output, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
};

static PyObject *__pyx_tp_new_4pypm_Input(PyTypeObject *t, PyObject *a, PyObject *k) {
  PyObject *o = (*t->tp_alloc)(t, 0);
  if (!o) return 0;
  return o;
}

static void __pyx_tp_dealloc_4pypm_Input(PyObject *o) {
  {
    PyObject *etype, *eval, *etb;
    PyErr_Fetch(&etype, &eval, &etb);
    ++o->ob_refcnt;
    __pyx_f_4pypm_5Input___dealloc__(o);
    if (PyErr_Occurred()) PyErr_WriteUnraisable(o);
    --o->ob_refcnt;
    PyErr_Restore(etype, eval, etb);
  }
  (*o->ob_type->tp_free)(o);
}

static struct PyMethodDef __pyx_methods_4pypm_Input[] = {
  {"SetFilter", (PyCFunction)__pyx_f_4pypm_5Input_SetFilter, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_5Input_SetFilter},
  {"SetChannelMask", (PyCFunction)__pyx_f_4pypm_5Input_SetChannelMask, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_5Input_SetChannelMask},
  {"Poll", (PyCFunction)__pyx_f_4pypm_5Input_Poll, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_5Input_Poll},
  {"Read", (PyCFunction)__pyx_f_4pypm_5Input_Read, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_5Input_Read},
  {0, 0, 0, 0}
};

static PyNumberMethods __pyx_tp_as_number_Input = {
  0, /*nb_add*/
  0, /*nb_subtract*/
  0, /*nb_multiply*/
  0, /*nb_divide*/
  0, /*nb_remainder*/
  0, /*nb_divmod*/
  0, /*nb_power*/
  0, /*nb_negative*/
  0, /*nb_positive*/
  0, /*nb_absolute*/
  0, /*nb_nonzero*/
  0, /*nb_invert*/
  0, /*nb_lshift*/
  0, /*nb_rshift*/
  0, /*nb_and*/
  0, /*nb_xor*/
  0, /*nb_or*/
  0, /*nb_coerce*/
  0, /*nb_int*/
  0, /*nb_long*/
  0, /*nb_float*/
  0, /*nb_oct*/
  0, /*nb_hex*/
  0, /*nb_inplace_add*/
  0, /*nb_inplace_subtract*/
  0, /*nb_inplace_multiply*/
  0, /*nb_inplace_divide*/
  0, /*nb_inplace_remainder*/
  0, /*nb_inplace_power*/
  0, /*nb_inplace_lshift*/
  0, /*nb_inplace_rshift*/
  0, /*nb_inplace_and*/
  0, /*nb_inplace_xor*/
  0, /*nb_inplace_or*/
  0, /*nb_floor_divide*/
  0, /*nb_true_divide*/
  0, /*nb_inplace_floor_divide*/
  0, /*nb_inplace_true_divide*/
  #if Py_TPFLAGS_DEFAULT & Py_TPFLAGS_HAVE_INDEX
  0, /*nb_index*/
  #endif
};

static PySequenceMethods __pyx_tp_as_sequence_Input = {
  0, /*sq_length*/
  0, /*sq_concat*/
  0, /*sq_repeat*/
  0, /*sq_item*/
  0, /*sq_slice*/
  0, /*sq_ass_item*/
  0, /*sq_ass_slice*/
  0, /*sq_contains*/
  0, /*sq_inplace_concat*/
  0, /*sq_inplace_repeat*/
};

static PyMappingMethods __pyx_tp_as_mapping_Input = {
  0, /*mp_length*/
  0, /*mp_subscript*/
  0, /*mp_ass_subscript*/
};

static PyBufferProcs __pyx_tp_as_buffer_Input = {
  0, /*bf_getreadbuffer*/
  0, /*bf_getwritebuffer*/
  0, /*bf_getsegcount*/
  0, /*bf_getcharbuffer*/
};

PyTypeObject __pyx_type_4pypm_Input = {
  PyObject_HEAD_INIT(0)
  0, /*ob_size*/
  "pypm.Input", /*tp_name*/
  sizeof(struct __pyx_obj_4pypm_Input), /*tp_basicsize*/
  0, /*tp_itemsize*/
  __pyx_tp_dealloc_4pypm_Input, /*tp_dealloc*/
  0, /*tp_print*/
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  &__pyx_tp_as_number_Input, /*tp_as_number*/
  &__pyx_tp_as_sequence_Input, /*tp_as_sequence*/
  &__pyx_tp_as_mapping_Input, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  &__pyx_tp_as_buffer_Input, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "\nclass Input:\n    define an input MIDI stream. Takes the form:\n        x = pypm.Input(MidiInputDevice)\n    ", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  __pyx_methods_4pypm_Input, /*tp_methods*/
  0, /*tp_members*/
  0, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  __pyx_f_4pypm_5Input___init__, /*tp_init*/
  0, /*tp_alloc*/
  __pyx_tp_new_4pypm_Input, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
};

static struct PyMethodDef __pyx_methods[] = {
  {"Initialize", (PyCFunction)__pyx_f_4pypm_Initialize, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_Initialize},
  {"Terminate", (PyCFunction)__pyx_f_4pypm_Terminate, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_Terminate},
  {"GetDefaultInputDeviceID", (PyCFunction)__pyx_f_4pypm_GetDefaultInputDeviceID, METH_VARARGS|METH_KEYWORDS, 0},
  {"GetDefaultOutputDeviceID", (PyCFunction)__pyx_f_4pypm_GetDefaultOutputDeviceID, METH_VARARGS|METH_KEYWORDS, 0},
  {"CountDevices", (PyCFunction)__pyx_f_4pypm_CountDevices, METH_VARARGS|METH_KEYWORDS, 0},
  {"GetDeviceInfo", (PyCFunction)__pyx_f_4pypm_GetDeviceInfo, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_GetDeviceInfo},
  {"Time", (PyCFunction)__pyx_f_4pypm_Time, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_Time},
  {"GetErrorText", (PyCFunction)__pyx_f_4pypm_GetErrorText, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_GetErrorText},
  {"Channel", (PyCFunction)__pyx_f_4pypm_Channel, METH_VARARGS|METH_KEYWORDS, __pyx_doc_4pypm_Channel},
  {0, 0, 0, 0}
};

static void __pyx_init_filenames(void); /*proto*/

PyMODINIT_FUNC initpypm(void); /*proto*/
PyMODINIT_FUNC initpypm(void) {
  PyObject *__pyx_1 = 0;
  PyObject *__pyx_2 = 0;
  PyObject *__pyx_3 = 0;
  PyObject *__pyx_4 = 0;
  __pyx_init_filenames();
  __pyx_m = Py_InitModule4("pypm", __pyx_methods, 0, 0, PYTHON_API_VERSION);
  if (!__pyx_m) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 7; goto __pyx_L1;};
  Py_INCREF(__pyx_m);
  __pyx_b = PyImport_AddModule("__builtin__");
  if (!__pyx_b) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 7; goto __pyx_L1;};
  if (PyObject_SetAttrString(__pyx_m, "__builtins__", __pyx_b) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 7; goto __pyx_L1;};
  if (__Pyx_InternStrings(__pyx_intern_tab) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 7; goto __pyx_L1;};
  if (__Pyx_InitStrings(__pyx_string_tab) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 7; goto __pyx_L1;};
  if (PyType_Ready(&__pyx_type_4pypm_Output) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 193; goto __pyx_L1;}
  if (PyObject_SetAttrString(__pyx_m, "Output", (PyObject *)&__pyx_type_4pypm_Output) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 193; goto __pyx_L1;}
  __pyx_ptype_4pypm_Output = &__pyx_type_4pypm_Output;
  if (PyType_Ready(&__pyx_type_4pypm_Input) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 322; goto __pyx_L1;}
  if (PyObject_SetAttrString(__pyx_m, "Input", (PyObject *)&__pyx_type_4pypm_Input) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 322; goto __pyx_L1;}
  __pyx_ptype_4pypm_Input = &__pyx_type_4pypm_Input;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":7 */
  if (PyObject_SetAttr(__pyx_m, __pyx_n___version__, __pyx_k1p) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 7; goto __pyx_L1;}

  /* "/home/rsd/dev/pygame/src/pypm.pyx":9 */
  __pyx_1 = __Pyx_Import(__pyx_n_array, 0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 9; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_array, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 9; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":100 */
  __pyx_1 = PyInt_FromLong(0x1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_ACTIVE, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 100; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":101 */
  __pyx_1 = PyInt_FromLong(0x2); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 101; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_SYSEX, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 101; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":102 */
  __pyx_1 = PyInt_FromLong(0x4); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 102; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_CLOCK, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 102; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":103 */
  __pyx_1 = PyInt_FromLong(0x8); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 103; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_PLAY, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 103; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":104 */
  __pyx_1 = PyInt_FromLong(0x10); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 104; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_F9, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 104; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":105 */
  __pyx_1 = PyInt_FromLong(0x10); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 105; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_TICK, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 105; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":106 */
  __pyx_1 = PyInt_FromLong(0x20); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_FD, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 106; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":107 */
  __pyx_1 = PyInt_FromLong(0x30); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 107; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_UNDEFINED, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 107; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":108 */
  __pyx_1 = PyInt_FromLong(0x40); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 108; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_RESET, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 108; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":109 */
  __pyx_1 = PyInt_FromLong(0x7F); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 109; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_REALTIME, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 109; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":110 */
  __pyx_1 = PyInt_FromLong(0x80); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 110; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_NOTE, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 110; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":111 */
  __pyx_1 = PyInt_FromLong(0x100); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 111; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_CHANNEL_AFTERTOUCH, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 111; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":112 */
  __pyx_1 = PyInt_FromLong(0x200); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 112; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_POLY_AFTERTOUCH, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 112; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":113 */
  __pyx_1 = PyInt_FromLong(0x300); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 113; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_AFTERTOUCH, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 113; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":114 */
  __pyx_1 = PyInt_FromLong(0x400); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 114; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_PROGRAM, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 114; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":115 */
  __pyx_1 = PyInt_FromLong(0x800); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 115; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_CONTROL, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 115; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":116 */
  __pyx_1 = PyInt_FromLong(0x1000); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 116; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_PITCHBEND, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 116; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":117 */
  __pyx_1 = PyInt_FromLong(0x2000); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 117; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_MTC, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 117; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":118 */
  __pyx_1 = PyInt_FromLong(0x4000); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_SONG_POSITION, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 118; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":119 */
  __pyx_1 = PyInt_FromLong(0x8000); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 119; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_SONG_SELECT, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 119; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":120 */
  __pyx_1 = PyInt_FromLong(0x10000); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 120; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FILT_TUNE, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 120; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":121 */
  __pyx_1 = PyInt_FromLong(0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 121; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_FALSE, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 121; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":122 */
  __pyx_1 = PyInt_FromLong(1); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 122; goto __pyx_L1;}
  if (PyObject_SetAttr(__pyx_m, __pyx_n_TRUE, __pyx_1) < 0) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 122; goto __pyx_L1;}
  Py_DECREF(__pyx_1); __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":205 */
  __pyx_1 = PyInt_FromLong(0); if (!__pyx_1) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 205; goto __pyx_L1;}
  __pyx_k3 = __pyx_1;
  __pyx_1 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":274 */
  __pyx_2 = PyInt_FromLong(0); if (!__pyx_2) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 274; goto __pyx_L1;}
  __pyx_k4 = __pyx_2;
  __pyx_2 = 0;
  __pyx_3 = PyInt_FromLong(0); if (!__pyx_3) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 274; goto __pyx_L1;}
  __pyx_k5 = __pyx_3;
  __pyx_3 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":332 */
  __pyx_4 = PyInt_FromLong(4096); if (!__pyx_4) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 332; goto __pyx_L1;}
  __pyx_k6 = __pyx_4;
  __pyx_4 = 0;

  /* "/home/rsd/dev/pygame/src/pypm.pyx":408 */
  return;
  __pyx_L1:;
  Py_XDECREF(__pyx_1);
  Py_XDECREF(__pyx_2);
  Py_XDECREF(__pyx_3);
  Py_XDECREF(__pyx_4);
  __Pyx_AddTraceback("pypm");
}

static char *__pyx_filenames[] = {
  "pypm.pyx",
};

/* Runtime support code */

static void __pyx_init_filenames(void) {
  __pyx_f = __pyx_filenames;
}

static PyObject *__Pyx_Import(PyObject *name, PyObject *from_list) {
    PyObject *__import__ = 0;
    PyObject *empty_list = 0;
    PyObject *module = 0;
    PyObject *global_dict = 0;
    PyObject *empty_dict = 0;
    PyObject *list;
    __import__ = PyObject_GetAttrString(__pyx_b, "__import__");
    if (!__import__)
        goto bad;
    if (from_list)
        list = from_list;
    else {
        empty_list = PyList_New(0);
        if (!empty_list)
            goto bad;
        list = empty_list;
    }
    global_dict = PyModule_GetDict(__pyx_m);
    if (!global_dict)
        goto bad;
    empty_dict = PyDict_New();
    if (!empty_dict)
        goto bad;
    module = PyObject_CallFunction(__import__, "OOOO",
        name, global_dict, empty_dict, list);
bad:
    Py_XDECREF(empty_list);
    Py_XDECREF(__import__);
    Py_XDECREF(empty_dict);
    return module;
}

static PyObject *__Pyx_GetStdout(void) {
    PyObject *f = PySys_GetObject("stdout");
    if (!f) {
        PyErr_SetString(PyExc_RuntimeError, "lost sys.stdout");
    }
    return f;
}

static int __Pyx_PrintItem(PyObject *v) {
    PyObject *f;
    
    if (!(f = __Pyx_GetStdout()))
        return -1;
    if (PyFile_SoftSpace(f, 1)) {
        if (PyFile_WriteString(" ", f) < 0)
            return -1;
    }
    if (PyFile_WriteObject(v, f, Py_PRINT_RAW) < 0)
        return -1;
    if (PyString_Check(v)) {
        char *s = PyString_AsString(v);
        Py_ssize_t len = PyString_Size(v);
        if (len > 0 &&
            isspace(Py_CHARMASK(s[len-1])) &&
            s[len-1] != ' ')
                PyFile_SoftSpace(f, 0);
    }
    return 0;
}

static int __Pyx_PrintNewline(void) {
    PyObject *f;
    
    if (!(f = __Pyx_GetStdout()))
        return -1;
    if (PyFile_WriteString("\n", f) < 0)
        return -1;
    PyFile_SoftSpace(f, 0);
    return 0;
}

static void __Pyx_Raise(PyObject *type, PyObject *value, PyObject *tb) {
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(tb);
    /* First, check the traceback argument, replacing None with NULL. */
    if (tb == Py_None) {
        Py_DECREF(tb);
        tb = 0;
    }
    else if (tb != NULL && !PyTraceBack_Check(tb)) {
        PyErr_SetString(PyExc_TypeError,
            "raise: arg 3 must be a traceback or None");
        goto raise_error;
    }
    /* Next, replace a missing value with None */
    if (value == NULL) {
        value = Py_None;
        Py_INCREF(value);
    }
    #if PY_VERSION_HEX < 0x02050000
    if (!PyClass_Check(type))
    #else
    if (!PyType_Check(type))
    #endif
    {
        /* Raising an instance.  The value should be a dummy. */
        if (value != Py_None) {
            PyErr_SetString(PyExc_TypeError,
                "instance exception may not have a separate value");
            goto raise_error;
        }
        /* Normalize to raise <class>, <instance> */
        Py_DECREF(value);
        value = type;
        #if PY_VERSION_HEX < 0x02050000
            if (PyInstance_Check(type)) {
                type = (PyObject*) ((PyInstanceObject*)type)->in_class;
                Py_INCREF(type);
            }
            else {
                PyErr_SetString(PyExc_TypeError,
                    "raise: exception must be an old-style class or instance");
                goto raise_error;
            }
        #else
            type = (PyObject*) type->ob_type;
            Py_INCREF(type);
            if (!PyType_IsSubtype((PyTypeObject *)type, (PyTypeObject *)PyExc_BaseException)) {
                PyErr_SetString(PyExc_TypeError,
                    "raise: exception class must be a subclass of BaseException");
                goto raise_error;
            }
        #endif
    }
    PyErr_Restore(type, value, tb);
    return;
raise_error:
    Py_XDECREF(value);
    Py_XDECREF(type);
    Py_XDECREF(tb);
    return;
}

static PyObject *__Pyx_GetName(PyObject *dict, PyObject *name) {
    PyObject *result;
    result = PyObject_GetAttr(dict, name);
    if (!result)
        PyErr_SetObject(PyExc_NameError, name);
    return result;
}

static int __Pyx_InternStrings(__Pyx_InternTabEntry *t) {
    while (t->p) {
        *t->p = PyString_InternFromString(t->s);
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

static int __Pyx_InitStrings(__Pyx_StringTabEntry *t) {
    while (t->p) {
        *t->p = PyString_FromStringAndSize(t->s, t->n - 1);
        if (!*t->p)
            return -1;
        ++t;
    }
    return 0;
}

#include "compile.h"
#include "frameobject.h"
#include "traceback.h"

static void __Pyx_AddTraceback(char *funcname) {
    PyObject *py_srcfile = 0;
    PyObject *py_funcname = 0;
    PyObject *py_globals = 0;
    PyObject *empty_tuple = 0;
    PyObject *empty_string = 0;
    PyCodeObject *py_code = 0;
    PyFrameObject *py_frame = 0;
    
    py_srcfile = PyString_FromString(__pyx_filename);
    if (!py_srcfile) goto bad;
    py_funcname = PyString_FromString(funcname);
    if (!py_funcname) goto bad;
    py_globals = PyModule_GetDict(__pyx_m);
    if (!py_globals) goto bad;
    empty_tuple = PyTuple_New(0);
    if (!empty_tuple) goto bad;
    empty_string = PyString_FromString("");
    if (!empty_string) goto bad;
    py_code = PyCode_New(
        0,            /*int argcount,*/
        0,            /*int nlocals,*/
        0,            /*int stacksize,*/
        0,            /*int flags,*/
        empty_string, /*PyObject *code,*/
        empty_tuple,  /*PyObject *consts,*/
        empty_tuple,  /*PyObject *names,*/
        empty_tuple,  /*PyObject *varnames,*/
        empty_tuple,  /*PyObject *freevars,*/
        empty_tuple,  /*PyObject *cellvars,*/
        py_srcfile,   /*PyObject *filename,*/
        py_funcname,  /*PyObject *name,*/
        __pyx_lineno,   /*int firstlineno,*/
        empty_string  /*PyObject *lnotab*/
    );
    if (!py_code) goto bad;
    py_frame = PyFrame_New(
        PyThreadState_Get(), /*PyThreadState *tstate,*/
        py_code,             /*PyCodeObject *code,*/
        py_globals,          /*PyObject *globals,*/
        0                    /*PyObject *locals*/
    );
    if (!py_frame) goto bad;
    py_frame->f_lineno = __pyx_lineno;
    PyTraceBack_Here(py_frame);
bad:
    Py_XDECREF(py_srcfile);
    Py_XDECREF(py_funcname);
    Py_XDECREF(empty_tuple);
    Py_XDECREF(empty_string);
    Py_XDECREF(py_code);
    Py_XDECREF(py_frame);
}
