#include <Python.h>
#include <stdio.h>
#include <string.h>
#include "setproctitle.h"

static PyMethodDef macosxMethods[]={{NULL,NULL,0,NULL}};
extern char *Py_GetProgramFullPath(void);
 
/*
  We do this because dyld does things when modules get linked
  This gives us a chance to do whateverthehellweneedtodo before
  AppKit and it's gang gets linked in and starts doing funky shit.
*/

void initmacosx(void)
{
	PyObject *module;
        char *nullstr = "(null)";
        /* pornography for the windowserver that doesn't know a damn about paths.. */
        printf("Py_GetProgramFullPath() = %s\n",Py_GetProgramFullPath());
        if (strcmp(nullstr,Py_GetProgramFullPath())!=0)
            setproctitle(Py_GetProgramFullPath());
        printf("Py_GetProgramFullPath() = %s\n",Py_GetProgramFullPath());
        module = Py_InitModule("macosx",macosxMethods);
}
